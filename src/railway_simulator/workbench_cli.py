"""Project/workbench CLI commands.

These commands mirror the Streamlit Project Workbench so studies can be run and
post-processed from a shell in a reproducible thesis/CI workflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import yaml

from railway_simulator.config.loader import (
    ConfigError,
    apply_collision_to_params,
    load_simulation_config,
)
from railway_simulator.domain.scenario import Scenario
from railway_simulator.domain.spectrum import SRSSettings
from railway_simulator.domain.study import StudyDefinition
from railway_simulator.persistence.database import ProjectDatabase, initialize_project_database
from railway_simulator.persistence.repositories import (
    ConfigSnapshotRepository,
    ProjectRepository,
    SpectrumRepository,
    StudyRepository,
)
from railway_simulator.services.project_service import ProjectService
from railway_simulator.services.simulation_service import SimulationService
from railway_simulator.spectrum.service import SpectrumService
from railway_simulator.studies.full_train import FullTrainStudyRunner, FullTrainStudySpec
from railway_simulator.studies.parametric_grid_cli import (
    GridExportFormat,
    format_records_table,
    preview_grid_from_yaml,
    run_grid_from_yaml,
    write_records,
)
from railway_simulator.studies.parametric_grid_io import ParametricGridYAMLError
from railway_simulator.studies.parametric_grid_persistence import (
    preview_parametric_grid_persistent_target,
    run_parametric_grid_persistent,
)
from railway_simulator.reporting import build_latex_chapter

project_app = typer.Typer(help="Create and inspect persistent project workspaces.")
study_app = typer.Typer(help="Run and inspect persistent parametric studies.")
srs_app = typer.Typer(help="Export and compare persisted Shock Response Spectrum curves.")
report_app = typer.Typer(help="Generate academic report bundles from persisted studies.")


@dataclass(frozen=True)
class ProjectContext:
    db: ProjectDatabase
    project_id: str
    project_root: Path


def register_workbench_commands(app: typer.Typer) -> None:
    """Register project/study/SRS command groups on the root Typer app."""
    app.add_typer(project_app, name="project")
    app.add_typer(study_app, name="study")
    app.add_typer(srs_app, name="srs")
    app.add_typer(report_app, name="report")


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise typer.BadParameter(f"Invalid YAML file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Expected a mapping at top level in {path}")
    return data


def _load_engine_config(path: Path) -> dict[str, Any]:
    try:
        cfg = load_simulation_config(path)
        return apply_collision_to_params(cfg, config_path=path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _parse_float_csv(value: str | None, *, default: Iterable[float] | None = None) -> tuple[float, ...]:
    if value is None or not str(value).strip():
        return tuple(default or ())
    tokens = str(value).replace(";", ",").replace(" ", ",").split(",")
    try:
        return tuple(float(tok) for tok in tokens if tok.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"Could not parse float list: {value!r}") from exc


def _parse_str_csv(value: str | None, *, default: Iterable[str] | None = None) -> tuple[str, ...]:
    if value is None or not str(value).strip():
        return tuple(default or ())
    return tuple(tok.strip() for tok in str(value).replace(";", ",").split(",") if tok.strip())


def _speeds_from_spec(raw: Any) -> tuple[float, ...]:
    if raw is None:
        return FullTrainStudySpec.speeds_kmh
    if isinstance(raw, (list, tuple)):
        return tuple(float(v) for v in raw)
    if isinstance(raw, dict):
        if "values" in raw:
            return tuple(float(v) for v in raw["values"])
        start = float(raw.get("start", 10.0))
        stop = float(raw.get("stop", start))
        step = float(raw.get("step", 10.0))
        if step <= 0:
            raise typer.BadParameter("speeds_kmh.step must be > 0")
        count = int(np.floor((stop - start) / step)) + 1
        values = [start + i * step for i in range(max(count, 0))]
        if values and values[-1] < stop and abs(values[-1] - stop) > 1.0e-9:
            values.append(stop)
        return tuple(float(v) for v in values)
    if isinstance(raw, str):
        return _parse_float_csv(raw)
    raise typer.BadParameter("Unsupported speeds_kmh specification")


def _tn_grid_from_spec(raw: Any) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return tuple(float(v) for v in raw)
    if isinstance(raw, str):
        return _parse_float_csv(raw)
    if isinstance(raw, dict):
        mode = str(raw.get("mode", "logspace")).lower()
        start = float(raw.get("min", raw.get("start", 10.0)))
        stop = float(raw.get("max", raw.get("stop", 3000.0)))
        n = int(raw.get("n", 80))
        if n <= 1:
            raise typer.BadParameter("Tn_grid_ms.n must be > 1")
        if mode == "linspace":
            return tuple(float(v) for v in np.linspace(start, stop, n))
        if mode == "logspace":
            if start <= 0 or stop <= 0:
                raise typer.BadParameter("logspace Tn grid needs positive min/max")
            return tuple(float(v) for v in np.geomspace(start, stop, n))
        raise typer.BadParameter(f"Unsupported Tn_grid_ms mode: {mode}")
    raise typer.BadParameter("Unsupported Tn_grid_ms specification")


def _resolve_db(db: Path) -> ProjectDatabase:
    if not db.exists():
        raise typer.BadParameter(f"Project database does not exist: {db}")
    return initialize_project_database(db)


def _project_context(db_path: Path, project_id: str | None = None) -> ProjectContext:
    db = _resolve_db(db_path)
    projects = ProjectRepository(db).list()
    if project_id is None:
        if not projects:
            raise typer.BadParameter("No project row found in database. Run 'project create' first.")
        if len(projects) > 1:
            raise typer.BadParameter("Multiple projects found. Pass --project-id explicitly.")
        project = projects[0]
    else:
        project = ProjectRepository(db).get(project_id)
    return ProjectContext(db=db, project_id=project.id, project_root=project.normalized_root())


def _build_full_train_spec(
    *,
    project_id: str,
    base_config_id: str,
    spec_path: Path | None,
    name: str | None,
    vehicles: str | None,
    modes: str | None,
    speeds: str | None,
    contact_models: str | None,
    mu_values: str | None,
    zeta_values: str | None,
    tn_grid: str | None,
) -> tuple[FullTrainStudySpec, Path | None]:
    raw: dict[str, Any] = _load_mapping(spec_path) if spec_path is not None else {}
    study_raw = raw.get("study", {}) if isinstance(raw.get("study", {}), dict) else {}
    base_raw = raw.get("base", {}) if isinstance(raw.get("base", {}), dict) else {}
    friction_raw = raw.get("friction", {}) if isinstance(raw.get("friction", {}), dict) else {}
    srs_raw = raw.get("srs", {}) if isinstance(raw.get("srs", {}), dict) else {}

    base_config_from_spec = base_raw.get("config")
    base_config_path = Path(base_config_from_spec) if base_config_from_spec else None
    if base_config_path is not None and spec_path is not None and not base_config_path.is_absolute():
        candidate = spec_path.parent / base_config_path
        if candidate.exists():
            base_config_path = candidate

    spec = FullTrainStudySpec(
        project_id=project_id,
        base_config_id=base_config_id,
        name=name or str(study_raw.get("name") or "train_consist_comparison"),
        vehicles=_parse_str_csv(vehicles, default=raw.get("vehicles") or ("traxx_br187",)),
        modes=_parse_str_csv(modes, default=raw.get("modes") or ("lok_solo", "zug_full")),
        speeds_kmh=_parse_float_csv(speeds) if speeds else _speeds_from_spec(raw.get("speeds_kmh")),
        contact_models=_parse_str_csv(
            contact_models,
            default=raw.get("contact_models") or ("anagnostopoulos", "flores", "lankarani-nikravesh"),
        ),
        mu_values=_parse_float_csv(mu_values, default=friction_raw.get("mu_values") or (0.30,)),
        zeta_srs=_parse_float_csv(zeta_values, default=srs_raw.get("zeta_values") or (0.05,)),
        Tn_grid_ms=_parse_float_csv(tn_grid) if tn_grid else _tn_grid_from_spec(srs_raw.get("Tn_grid_ms")),
        parameters={
            "source_spec_path": str(spec_path) if spec_path is not None else None,
            "contact_patch_version": base_raw.get("contact_patch_version", "contact_state_per_mass_v1"),
        },
    )
    return spec, base_config_path


def _write_plot(path: Path, frames: list[tuple[str, pd.DataFrame]], *, value_column: str, title: str) -> None:
    fig, ax = plt.subplots()
    for label, frame in frames:
        if "Tn_ms" not in frame.columns or value_column not in frame.columns:
            continue
        ax.plot(frame["Tn_ms"], frame[value_column], label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Tn [ms]")
    ax.set_ylabel(value_column)
    ax.set_title(title)
    ax.grid(True, which="both")
    if frames:
        ax.legend(fontsize="small")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_curve_records(
    *,
    spectrum_repo: SpectrumRepository,
    study_id: str,
    zeta: float | None = None,
    modes: set[str] | None = None,
    contact_models: set[str] | None = None,
    speed_kmh: float | None = None,
    mu: float | None = None,
) -> list[dict[str, Any]]:
    records = spectrum_repo.list_curve_records_for_study(study_id)
    out: list[dict[str, Any]] = []
    for rec in records:
        if zeta is not None and abs(float(rec.get("zeta", np.nan)) - float(zeta)) > 1.0e-12:
            continue
        if modes and rec.get("meta_mode") not in modes:
            continue
        if contact_models and rec.get("meta_contact_model") not in contact_models:
            continue
        if speed_kmh is not None and abs(float(rec.get("meta_speed_kmh", np.nan)) - float(speed_kmh)) > 1.0e-9:
            continue
        if mu is not None and abs(float(rec.get("meta_mu", np.nan)) - float(mu)) > 1.0e-12:
            continue
        out.append(rec)
    return out


def _curve_label(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("meta_mode") or record.get("scenario_name") or record.get("run_id")),
        f"v={record.get('meta_speed_kmh', '?')} km/h",
        str(record.get("meta_contact_model") or "contact=?"),
        f"mu={record.get('meta_mu', '?')}",
        f"zeta={record.get('zeta', '?')}",
    ]
    return " | ".join(parts)


def _curve_to_long_frame(record: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(record["curve_csv_path"])
    for key, value in record.items():
        if key.endswith("_json") or isinstance(value, (dict, list)):
            continue
        if key not in df.columns:
            df.insert(0, key, value)
    return df


@report_app.command("build-chapter")
def report_build_chapter(
    db: Path = typer.Option(..., "--db", exists=True, readable=True, help="Path to project.sqlite."),
    study_id: str = typer.Option(..., "--study-id", help="Study id to render."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory for the LaTeX bundle."),
    title: str = typer.Option(
        "Parametric Demand Study for Railway Vehicle Impact on Trackside Structures",
        "--title",
        help="Chapter title.",
    ),
    author: str = typer.Option("S. Labbe", "--author", help="Author string for standalone main.tex."),
    zeta: float | None = typer.Option(0.05, "--zeta", help="Optional damping-ratio filter for figures."),
) -> None:
    """Build a thesis-style LaTeX chapter bundle from a persisted study."""
    result = build_latex_chapter(
        db=_resolve_db(db),
        study_id=study_id,
        output_dir=output_dir,
        title=title,
        author=author,
        zeta=zeta,
    )
    typer.echo(f"Chapter: {result.chapter_tex}")
    typer.echo(f"Standalone main: {result.main_tex}")
    typer.echo(f"Bibliography: {result.bibliography_bib}")
    typer.echo(f"Metadata: {result.metadata_json}")
    typer.echo(f"Figures: {len(result.figures)}")
    typer.echo(f"Tables: {len(result.tables)}")


@project_app.command("create")
def project_create(
    name: str = typer.Option(..., "--name", "-n", help="Project name."),
    root: Path = typer.Option(..., "--root", "-r", help="Project folder to create."),
    description: str = typer.Option("", "--description", "-d", help="Optional project description."),
) -> None:
    """Create a project folder with project.sqlite and artifact subfolders."""
    project, _db = ProjectService().create_project(name=name, root_dir=root, description=description)
    typer.echo(f"Project created: {project.name}")
    typer.echo(f"Project id: {project.id}")
    typer.echo(f"Root: {project.normalized_root()}")
    typer.echo(f"Database: {project.normalized_root() / 'project.sqlite'}")


@project_app.command("list")
def project_list(
    db: Path = typer.Option(..., "--db", exists=True, readable=True, help="Path to project.sqlite."),
) -> None:
    """List project rows in a project database."""
    database = _resolve_db(db)
    rows = ProjectRepository(database).list()
    if not rows:
        typer.echo("No projects found.")
        return
    for project in rows:
        typer.echo(f"{project.id}\t{project.name}\t{project.normalized_root()}")


@study_app.command("list")
def study_list(
    db: Path = typer.Option(..., "--db", exists=True, readable=True, help="Path to project.sqlite."),
    project_id: str | None = typer.Option(None, "--project-id", help="Project id, required only for multi-project DBs."),
) -> None:
    """List studies stored in a project database."""
    ctx = _project_context(db, project_id)
    studies = StudyRepository(ctx.db).list_studies(ctx.project_id)
    if not studies:
        typer.echo("No studies found.")
        return
    for study in studies:
        typer.echo(f"{study.id}\t{study.status}\t{study.study_type}\t{study.name}")


@study_app.command("runs")
def study_runs(
    db: Path = typer.Option(..., "--db", exists=True, readable=True, help="Path to project.sqlite."),
    study_id: str = typer.Option(..., "--study-id", help="Study id."),
) -> None:
    """List persisted run records for one study."""
    database = _resolve_db(db)
    rows = StudyRepository(database).list_run_records_for_study(study_id)
    if not rows:
        typer.echo("No runs found.")
        return
    for row in rows:
        typer.echo(
            "\t".join(
                [
                    str(row.get("run_id")),
                    str(row.get("run_status")),
                    str(row.get("meta_mode", "")),
                    str(row.get("meta_speed_kmh", "")),
                    str(row.get("meta_contact_model", "")),
                    str(row.get("elapsed_s", "")),
                ]
            )
        )


@study_app.command("run-grid")
def study_run_grid(
    spec: Path = typer.Option(..., "--spec", "-s", help="Parametric grid YAML spec."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview scenarios without running the solver."),
    limit: int | None = typer.Option(None, "--limit", min=0, help="Run or preview only the first N scenarios."),
    strict: bool = typer.Option(False, "--strict", help="Fail immediately when one scenario fails."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Optional summary/preview output path."),
    db: Path | None = typer.Option(None, "--db", help="Optional project.sqlite path for persistent runs."),
    project_name: str | None = typer.Option(
        None,
        "--project-name",
        help="Project name to create or reuse when --db is provided.",
    ),
    study_name: str | None = typer.Option(
        None,
        "--study-name",
        help="Study name override when --db is provided.",
    ),
    output_format: GridExportFormat = typer.Option(
        GridExportFormat.csv,
        "--format",
        case_sensitive=False,
        help="Output format when --out is provided: csv, json, or both.",
    ),
    base_config: Path | None = typer.Option(
        None,
        "--base-config",
        "-c",
        help="Override base.config from the YAML spec.",
    ),
) -> None:
    """Preview or run a YAML-defined parametric grid, optionally persisted."""

    try:
        if dry_run:
            definition, base_config_path, rows = preview_grid_from_yaml(
                spec,
                limit=limit,
                base_config_override=base_config,
            )
            mode = "Dry run"
            display_name = study_name or definition.grid.name
            target = (
                preview_parametric_grid_persistent_target(
                    db_path=db,
                    project_name=project_name,
                    study_name=display_name,
                    definition=definition,
                )
                if db is not None
                else None
            )
        elif db is not None:
            result = run_parametric_grid_persistent(
                spec,
                db_path=db,
                base_config_override=base_config,
                project_name=project_name,
                study_name=study_name,
                limit=limit,
                strict=strict,
            )
            rows = result.rows
            base_config_path = result.base_config_path
            mode = "Run"
            display_name = result.study.name
            target = None
        else:
            definition, base_config_path, rows = run_grid_from_yaml(
                spec,
                base_config_override=base_config,
                limit=limit,
                strict=strict,
            )
            mode = "Run"
            display_name = definition.grid.name
            target = None
    except ParametricGridYAMLError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        raise typer.BadParameter(f"Grid run failed: {exc}") from exc

    typer.echo(f"{mode}: {display_name}")
    typer.echo(f"Spec: {Path(spec)}")
    if base_config_path is not None:
        typer.echo(f"Base config: {base_config_path}")
    if db is not None:
        typer.echo(f"Database: {Path(db)}")
        if target is not None:
            project_suffix = f" ({target.project_id})" if target.project_id is not None else " (new)"
            typer.echo(f"Project: {target.project_name}{project_suffix}")
            typer.echo(f"Study: {target.study_name} (dry-run, not written)")
        elif "result" in locals():
            typer.echo(f"Project: {result.project.name} ({result.project.id})")
            typer.echo(f"Study id: {result.study.id}")
    typer.echo(f"Scenarios: {len(rows)}")
    typer.echo(format_records_table(rows))

    if out is not None:
        written = write_records(rows, out, export_format=output_format)
        for path in written:
            typer.echo(f"Wrote: {path}")


@study_app.command("run-full-train")
def study_run_full_train(
    db: Path = typer.Option(..., "--db", exists=True, readable=True, help="Path to project.sqlite."),
    base_config: Path | None = typer.Option(None, "--base-config", "-c", exists=True, readable=True, help="Base engine YAML/JSON config."),
    spec_file: Path | None = typer.Option(None, "--spec", "-s", exists=True, readable=True, help="Full train study YAML spec."),
    project_id: str | None = typer.Option(None, "--project-id", help="Project id, required only for multi-project DBs."),
    name: str | None = typer.Option(None, "--name", help="Study name override."),
    vehicles: str | None = typer.Option(None, "--vehicles", help="Comma-separated vehicles, currently traxx_br187."),
    modes: str | None = typer.Option(None, "--modes", help="Comma-separated modes: lok_solo,zug_full."),
    speeds: str | None = typer.Option(None, "--speeds", help="Comma-separated speeds in km/h, e.g. 10,20,30."),
    contact_models: str | None = typer.Option(None, "--contact-models", help="Comma-separated contact models."),
    mu_values: str | None = typer.Option(None, "--mu-values", help="Comma-separated friction values."),
    zeta_values: str | None = typer.Option(None, "--zeta-values", help="Comma-separated SRS damping ratios."),
    tn_grid: str | None = typer.Option(None, "--tn-grid-ms", help="Explicit comma-separated SRS periods in ms."),
    no_srs: bool = typer.Option(False, "--no-srs", help="Run histories only; do not compute persisted SRS curves."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build scenarios and print count without writing DB or running solver."),
) -> None:
    """Run the persistent Lok-solo vs full-consist parametric study."""
    ctx = _project_context(db, project_id)
    provisional_base_id = "cfg_dry_run" if dry_run else "cfg_pending"
    spec, base_from_spec = _build_full_train_spec(
        project_id=ctx.project_id,
        base_config_id=provisional_base_id,
        spec_path=spec_file,
        name=name,
        vehicles=vehicles,
        modes=modes,
        speeds=speeds,
        contact_models=contact_models,
        mu_values=mu_values,
        zeta_values=zeta_values,
        tn_grid=tn_grid,
    )
    resolved_base_config = base_config or base_from_spec
    if resolved_base_config is None:
        raise typer.BadParameter("Pass --base-config or provide base.config in --spec.")
    if not resolved_base_config.exists():
        raise typer.BadParameter(f"Base config does not exist: {resolved_base_config}")

    base = _load_engine_config(resolved_base_config)

    if dry_run:
        dry_study = StudyDefinition(
            project_id=ctx.project_id,
            name=spec.name,
            study_type="full_train",
            base_config_id="cfg_dry_run",
        )
        runner = FullTrainStudyRunner(simulation_service=SimulationService())
        scenarios = runner.build_scenarios(study=dry_study, base_config=base, spec=spec)
        typer.echo(f"Dry run: {len(scenarios)} scenarios would be built.")
        typer.echo(json.dumps(spec.to_study_definition().parameters, indent=2, sort_keys=True))
        return

    cfg_snapshot = ConfigSnapshotRepository(ctx.db).create(ctx.project_id, base, source_path=resolved_base_config)
    spec = FullTrainStudySpec(
        project_id=spec.project_id,
        base_config_id=cfg_snapshot.id,
        name=spec.name,
        vehicles=spec.vehicles,
        modes=spec.modes,
        speeds_kmh=spec.speeds_kmh,
        contact_models=spec.contact_models,
        mu_values=spec.mu_values,
        zeta_srs=spec.zeta_srs,
        Tn_grid_ms=spec.Tn_grid_ms,
        parameters=spec.parameters,
    )
    study_repo = StudyRepository(ctx.db)
    spectrum_repo = SpectrumRepository(ctx.db)
    sim_service = SimulationService(repository=study_repo, runs_dir=ctx.project_root / "runs")
    spectrum_service = SpectrumService(repository=spectrum_repo, spectra_dir=ctx.project_root / "spectra")
    runner = FullTrainStudyRunner(
        simulation_service=sim_service,
        study_repository=study_repo,
        spectrum_service=spectrum_service,
    )

    def _progress(index: int, total: int, scenario: Scenario, run: Any) -> None:
        typer.echo(f"[{index}/{total}] {scenario.name} -> {run.status}")

    result = runner.run(spec=spec, base_config=base, compute_srs=not no_srs, progress_callback=_progress)
    study = result["study"]
    ok = sum(1 for run in result["runs"] if run.status == "ok")
    failed = len(result["runs"]) - ok
    typer.echo(f"Study id: {study.id}")
    typer.echo(f"Runs: {ok} ok, {failed} failed")
    typer.echo(f"SRS curves: {len(result['srs_curves'])}")


@srs_app.command("list")
def srs_list(
    db: Path = typer.Option(..., "--db", exists=True, readable=True, help="Path to project.sqlite."),
    study_id: str = typer.Option(..., "--study-id", help="Study id."),
) -> None:
    """List SRS curves persisted for one study."""
    records = SpectrumRepository(_resolve_db(db)).list_curve_records_for_study(study_id)
    if not records:
        typer.echo("No SRS curves found.")
        return
    for rec in records:
        typer.echo(
            "\t".join(
                [
                    str(rec.get("curve_id")),
                    str(rec.get("zeta")),
                    str(rec.get("meta_mode", "")),
                    str(rec.get("meta_speed_kmh", "")),
                    str(rec.get("meta_contact_model", "")),
                    str(rec.get("curve_csv_path")),
                ]
            )
        )


@srs_app.command("export")
def srs_export(
    db: Path = typer.Option(..., "--db", exists=True, readable=True, help="Path to project.sqlite."),
    study_id: str = typer.Option(..., "--study-id", help="Study id."),
    output: Path = typer.Option(..., "--output", "-o", help="Output long-format CSV."),
    zeta: float | None = typer.Option(None, "--zeta", help="Optional damping ratio filter."),
    modes: str | None = typer.Option(None, "--modes", help="Optional comma-separated mode filter."),
    contact_models: str | None = typer.Option(None, "--contact-models", help="Optional comma-separated contact model filter."),
    speed_kmh: float | None = typer.Option(None, "--speed-kmh", help="Optional speed filter."),
    mu: float | None = typer.Option(None, "--mu", help="Optional friction filter."),
) -> None:
    """Export selected SRS curves to one long-format CSV."""
    repo = SpectrumRepository(_resolve_db(db))
    records = _load_curve_records(
        spectrum_repo=repo,
        study_id=study_id,
        zeta=zeta,
        modes=set(_parse_str_csv(modes)) if modes else None,
        contact_models=set(_parse_str_csv(contact_models)) if contact_models else None,
        speed_kmh=speed_kmh,
        mu=mu,
    )
    if not records:
        raise typer.BadParameter("No SRS curves match the requested filters.")
    frames = [_curve_to_long_frame(rec) for rec in records]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    typer.echo(f"Exported {len(records)} curves to {output}")


@srs_app.command("compare")
def srs_compare(
    db: Path = typer.Option(..., "--db", exists=True, readable=True, help="Path to project.sqlite."),
    study_id: str = typer.Option(..., "--study-id", help="Study id."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory for CSV/PNG comparison artifacts."),
    zeta: float = typer.Option(0.05, "--zeta", help="Damping ratio."),
    contact_models: str | None = typer.Option(None, "--contact-models", help="Optional comma-separated contact model filter."),
    speed_kmh: float | None = typer.Option(None, "--speed-kmh", help="Optional speed filter."),
    mu: float | None = typer.Option(None, "--mu", help="Optional friction filter."),
) -> None:
    """Create reproducible SRS overlay, envelope and full-train/lok ratio artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = SpectrumRepository(_resolve_db(db))
    records = _load_curve_records(
        spectrum_repo=repo,
        study_id=study_id,
        zeta=zeta,
        contact_models=set(_parse_str_csv(contact_models)) if contact_models else None,
        speed_kmh=speed_kmh,
        mu=mu,
    )
    if not records:
        raise typer.BadParameter("No SRS curves match the requested filters.")

    labelled = [(_curve_label(rec), pd.read_csv(rec["curve_csv_path"])) for rec in records]
    long_df = pd.concat([_curve_to_long_frame(rec) for rec in records], ignore_index=True)
    long_path = output_dir / "srs_curves_long.csv"
    long_df.to_csv(long_path, index=False)

    value_column = "Feq_MN" if "Feq_MN" in labelled[0][1].columns else "Feq"
    envelope = SpectrumService.envelope([frame for _label, frame in labelled], value_column=value_column)
    envelope_path = output_dir / "srs_envelope.csv"
    envelope.to_csv(envelope_path, index=False)

    _write_plot(output_dir / "srs_overlay.png", labelled, value_column=value_column, title="SRS comparison")
    env_col = f"{value_column}_envelope"
    _write_plot(output_dir / "srs_envelope.png", [("envelope", envelope)], value_column=env_col, title="SRS envelope")

    ratios: list[pd.DataFrame] = []
    groups: dict[tuple[Any, Any, Any, Any], dict[str, dict[str, Any]]] = {}
    for rec in records:
        key = (rec.get("meta_speed_kmh"), rec.get("meta_contact_model"), rec.get("meta_mu"), rec.get("zeta"))
        mode = str(rec.get("meta_mode"))
        groups.setdefault(key, {})[mode] = rec
    for key, pair in groups.items():
        if "zug_full" not in pair or "lok_solo" not in pair:
            continue
        ratio = SpectrumService.ratio(
            pd.read_csv(pair["zug_full"]["curve_csv_path"]),
            pd.read_csv(pair["lok_solo"]["curve_csv_path"]),
            value_column=value_column,
        )
        ratio.insert(0, "speed_kmh", key[0])
        ratio.insert(1, "contact_model", key[1])
        ratio.insert(2, "mu", key[2])
        ratio.insert(3, "zeta", key[3])
        ratios.append(ratio)
    if ratios:
        ratio_df = pd.concat(ratios, ignore_index=True)
        ratio_path = output_dir / "srs_full_train_vs_lok_ratio.csv"
        ratio_df.to_csv(ratio_path, index=False)
        ratio_value_col = f"{value_column}_ratio"
        ratio_frames = []
        for (speed, contact, mu_val, zeta_val), grp in ratio_df.groupby(["speed_kmh", "contact_model", "mu", "zeta"]):
            ratio_frames.append((f"v={speed} {contact} mu={mu_val} zeta={zeta_val}", grp))
        _write_plot(output_dir / "srs_full_train_vs_lok_ratio.png", ratio_frames, value_column=ratio_value_col, title="Full train / Lok solo SRS ratio")
    else:
        ratio_path = None

    typer.echo(f"Long CSV: {long_path}")
    typer.echo(f"Envelope CSV: {envelope_path}")
    if ratio_path is not None:
        typer.echo(f"Ratio CSV: {ratio_path}")
    typer.echo(f"Plots: {output_dir}")
