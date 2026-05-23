"""SQLite persistence for YAML-defined generic parametric grids."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from railway_simulator.domain.common import utc_now_iso
from railway_simulator.domain.project import Project
from railway_simulator.domain.result import RunMetric, SimulationRun
from railway_simulator.domain.scenario import Scenario
from railway_simulator.domain.study import StudyDefinition
from railway_simulator.persistence.database import ProjectDatabase, initialize_project_database
from railway_simulator.persistence.repositories import (
    ConfigSnapshotRepository,
    ProjectRepository,
    StudyRepository,
)
from railway_simulator.services.simulation_service import SimulationService, extract_run_metrics
from railway_simulator.studies.parametric_grid import (
    ParametricScenario,
    apply_scenario_to_config,
    build_parametric_scenarios,
)
from railway_simulator.studies.parametric_grid_cli import (
    extract_grid_result_warnings,
    find_result_dataframe,
    load_base_config_for_grid,
    load_grid_definition,
    summarize_parametric_grid_record,
)
from railway_simulator.studies.parametric_grid_io import (
    ParametricGridDefinition,
    ParametricGridYAMLError,
)


RunCaseFn = Callable[[dict[str, Any], ParametricScenario], Any]
DEFAULT_PROJECT_NAME = "impact_workbench"


@dataclass(frozen=True)
class PersistentGridRunResult:
    """Result bundle returned after a persisted parametric grid run."""

    db: ProjectDatabase
    project: Project
    study: StudyDefinition
    scenarios: list[Scenario]
    runs: list[SimulationRun]
    rows: list[dict[str, Any]]
    base_config_path: Path


@dataclass(frozen=True)
class PersistentGridTargetPreview:
    """No-write description of the DB target a persistent grid run would use."""

    db_path: Path
    project_name: str
    project_id: str | None
    study_name: str
    db_exists: bool


class _WarningCollector(logging.Handler):
    """Collect warning log records emitted while one scenario is running."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


def preview_parametric_grid_persistent_target(
    *,
    db_path: str | Path,
    project_name: str | None = None,
    study_name: str | None = None,
    definition: ParametricGridDefinition | None = None,
) -> PersistentGridTargetPreview:
    """Describe the persistent target without creating or modifying SQLite."""

    path = Path(db_path).expanduser()
    resolved_study_name = study_name or (definition.grid.name if definition is not None else "")
    project_id: str | None = None
    resolved_project_name = project_name or DEFAULT_PROJECT_NAME

    if path.exists():
        for row in _read_project_rows_no_write(path):
            if project_name is not None and row["name"] == project_name:
                project_id = row["id"]
                resolved_project_name = row["name"]
                break
            if project_name is None and project_id is None:
                project_id = row["id"]
                resolved_project_name = row["name"]

    return PersistentGridTargetPreview(
        db_path=path,
        project_name=resolved_project_name,
        project_id=project_id,
        study_name=resolved_study_name,
        db_exists=path.exists(),
    )


def run_parametric_grid_persistent(
    spec_path: str | Path,
    *,
    db_path: str | Path,
    base_config_override: str | Path | None = None,
    project_name: str | None = None,
    study_name: str | None = None,
    limit: int | None = None,
    strict: bool = False,
    run_case_fn: RunCaseFn | None = None,
) -> PersistentGridRunResult:
    """Run a YAML parametric grid and persist project/study/scenario/run rows."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative.")

    definition, resolved_base_config_path = load_grid_definition(
        spec_path,
        base_config_override=base_config_override,
    )
    if resolved_base_config_path is None:
        raise ParametricGridYAMLError("Pass --base-config or provide base.config in the spec.")
    base_config = load_base_config_for_grid(resolved_base_config_path)

    db = initialize_project_database(db_path)
    project = _get_or_create_project(db, db_path=Path(db_path), project_name=project_name)
    _ensure_project_dirs(project.normalized_root())

    config_snapshot = ConfigSnapshotRepository(db).create(
        project.id,
        base_config,
        source_path=resolved_base_config_path,
    )
    study = StudyDefinition(
        project_id=project.id,
        name=study_name or definition.grid.name,
        study_type="parametric_grid",
        base_config_id=config_snapshot.id,
        parameters=_study_parameters(
            definition=definition,
            spec_path=Path(spec_path),
            base_config_path=resolved_base_config_path,
        ),
    )

    study_repo = StudyRepository(db)
    study_repo.create_study(study)
    simulation_service = SimulationService(
        repository=study_repo,
        runs_dir=project.normalized_root() / "runs",
    )

    grid_scenarios = build_parametric_scenarios(definition.grid)
    if limit is not None:
        grid_scenarios = grid_scenarios[:limit]

    scenarios: list[Scenario] = []
    runs: list[SimulationRun] = []
    rows: list[dict[str, Any]] = []

    failed = False
    for grid_scenario in grid_scenarios:
        scenario = _create_persistent_scenario(
            study=study,
            base_config=base_config,
            grid_scenario=grid_scenario,
            spec_path=Path(spec_path),
            base_config_path=resolved_base_config_path,
        )
        study_repo.add_scenario(scenario)
        scenarios.append(scenario)

        run, result, warnings_text, error_text = _execute_and_persist_scenario(
            scenario=scenario,
            grid_scenario=grid_scenario,
            simulation_service=simulation_service,
            study_repo=study_repo,
            runs_dir=project.normalized_root() / "runs",
            run_case_fn=run_case_fn,
        )
        runs.append(run)
        if run.status != "ok":
            failed = True

        if warnings_text:
            scenario.meta["solver_warnings"] = warnings_text
            study_repo.update_scenario_meta(scenario.id, scenario.meta)

        row = summarize_parametric_grid_record(
            {
                "scenario": grid_scenario,
                "status": run.status,
                "result": result,
                "warnings": warnings_text,
                "error": error_text or run.error_message,
                "error_type": None,
            }
        )
        rows.append(row)

        if strict and run.status != "ok":
            study_repo.update_study_status(study.id, "failed", finished_at=utc_now_iso())
            raise RuntimeError(run.error_message or f"Scenario failed: {grid_scenario.label}")

    status = "completed_with_failures" if failed else "completed"
    study_repo.update_study_status(study.id, status, finished_at=utc_now_iso())

    return PersistentGridRunResult(
        db=db,
        project=project,
        study=study,
        scenarios=scenarios,
        runs=runs,
        rows=rows,
        base_config_path=resolved_base_config_path,
    )


def _get_or_create_project(
    db: ProjectDatabase,
    *,
    db_path: Path,
    project_name: str | None,
) -> Project:
    projects = ProjectRepository(db)
    existing = projects.list()

    if project_name is not None:
        for project in existing:
            if project.name == project_name:
                return project
        project = Project(name=project_name, root_dir=Path(db_path).expanduser().resolve().parent)
        projects.create(project)
        return project

    if len(existing) == 1:
        return existing[0]

    for project in existing:
        if project.name == DEFAULT_PROJECT_NAME:
            return project

    if existing:
        names = ", ".join(project.name for project in existing)
        raise ValueError(f"Multiple projects found ({names}). Pass --project-name.")

    project = Project(name=DEFAULT_PROJECT_NAME, root_dir=Path(db_path).expanduser().resolve().parent)
    projects.create(project)
    return project


def _ensure_project_dirs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for child in ["configs", "runs", "studies", "spectra", "artifacts"]:
        (root / child).mkdir(exist_ok=True)


def _study_parameters(
    *,
    definition: ParametricGridDefinition,
    spec_path: Path,
    base_config_path: Path,
) -> dict[str, Any]:
    return {
        "study_type": "parametric_grid",
        "source_spec_path": str(spec_path),
        "base_config_path": str(base_config_path),
        "description": definition.grid.description,
        "quantities": list(definition.grid.quantities),
        "dimensions": [
            {
                "name": dimension.name,
                "kind": dimension.kind,
                "path": dimension.path,
                "values": list(dimension.values),
            }
            for dimension in definition.grid.dimensions
        ],
        "srs": definition.srs,
    }


def _create_persistent_scenario(
    *,
    study: StudyDefinition,
    base_config: dict[str, Any],
    grid_scenario: ParametricScenario,
    spec_path: Path,
    base_config_path: Path,
) -> Scenario:
    meta = _scenario_metadata(
        grid_scenario=grid_scenario,
        spec_path=spec_path,
        base_config_path=base_config_path,
    )
    try:
        params = apply_scenario_to_config(base_config, grid_scenario)
    except Exception as exc:
        params = dict(base_config)
        meta["configuration_error"] = str(exc)
    return Scenario(
        study_id=study.id,
        name=grid_scenario.label,
        params=params,
        meta=meta,
    )


def _scenario_metadata(
    *,
    grid_scenario: ParametricScenario,
    spec_path: Path,
    base_config_path: Path,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "scenario_index": grid_scenario.index,
        "scenario_label": grid_scenario.label,
        "study_type": "parametric_grid",
        "spec_path": str(spec_path),
        "base_config_path": str(base_config_path),
    }
    for key, value in grid_scenario.metadata.items():
        meta[key] = _metadata_scalar(value)
    return meta


def _metadata_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _execute_and_persist_scenario(
    *,
    scenario: Scenario,
    grid_scenario: ParametricScenario,
    simulation_service: SimulationService,
    study_repo: StudyRepository,
    runs_dir: Path,
    run_case_fn: RunCaseFn | None,
) -> tuple[SimulationRun, Any, str, str | None]:
    if scenario.meta.get("configuration_error"):
        run = SimulationRun(
            scenario_id=scenario.id,
            status="failed",
            config_hash=scenario.config_hash,
            error_message=str(scenario.meta["configuration_error"]),
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
        )
        study_repo.add_run(run)
        return run, None, "", run.error_message

    if run_case_fn is None:
        return _execute_with_simulation_service(
            scenario=scenario,
            simulation_service=simulation_service,
        )
    return _execute_with_injected_runner(
        scenario=scenario,
        grid_scenario=grid_scenario,
        study_repo=study_repo,
        runs_dir=runs_dir,
        run_case_fn=run_case_fn,
    )


def _execute_with_simulation_service(
    *,
    scenario: Scenario,
    simulation_service: SimulationService,
) -> tuple[SimulationRun, pd.DataFrame | None, str, str | None]:
    collector = _WarningCollector()
    collector.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("railway_simulator.core")
    logger.addHandler(collector)
    try:
        run = simulation_service.run_scenario(scenario)
    finally:
        logger.removeHandler(collector)

    result: pd.DataFrame | None = None
    if run.status == "ok" and run.result_csv_path is not None and run.result_csv_path.exists():
        result = pd.read_csv(run.result_csv_path)

    # TODO: If the engine grows structured solver warnings, persist them in a
    # first-class table/artifact. For now the summary exposes captured log text.
    return run, result, "; ".join(collector.messages), run.error_message


def _execute_with_injected_runner(
    *,
    scenario: Scenario,
    grid_scenario: ParametricScenario,
    study_repo: StudyRepository,
    runs_dir: Path,
    run_case_fn: RunCaseFn,
) -> tuple[SimulationRun, Any, str, str | None]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    start = time.perf_counter()
    try:
        result = run_case_fn(dict(scenario.params), grid_scenario)
        elapsed = time.perf_counter() - start
        df = find_result_dataframe(result)
        result_csv_path = runs_dir / f"{scenario.id}.csv" if df is not None else None
        if df is not None and result_csv_path is not None:
            df.to_csv(result_csv_path, index=False)
        run = SimulationRun(
            scenario_id=scenario.id,
            status="ok",
            config_hash=scenario.config_hash,
            result_csv_path=result_csv_path,
            elapsed_s=elapsed,
            started_at=started_at,
            finished_at=utc_now_iso(),
        )
        study_repo.add_run(run)
        if df is not None:
            for name, value, unit in extract_run_metrics(df):
                study_repo.add_metric(RunMetric(run_id=run.id, name=name, value=value, unit=unit))
        return run, result, extract_grid_result_warnings(result), None
    except Exception as exc:
        elapsed = time.perf_counter() - start
        run = SimulationRun(
            scenario_id=scenario.id,
            status="failed",
            config_hash=scenario.config_hash,
            elapsed_s=elapsed,
            error_message=repr(exc),
            started_at=started_at,
            finished_at=utc_now_iso(),
        )
        study_repo.add_run(run)
        return run, None, "", str(exc)


def _read_project_rows_no_write(path: Path) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT id, name FROM projects ORDER BY created_at, name").fetchall()
    except sqlite3.Error:
        return []
    return [{"id": row["id"], "name": row["name"]} for row in rows]


def load_parametric_grid_study_summary(
    db: ProjectDatabase,
    study_id: str,
) -> list[dict[str, Any]]:
    """Reconstruct a parametric-grid summary from persisted runs.

    The live runner returns a rich summary immediately after execution. This
    reader rebuilds the same UI-oriented table later from the SQLite metadata
    and persisted run CSV files, so saved studies can be browsed without
    re-running the solver. Solver warnings are read from scenario metadata when
    available. Older studies created before warning persistence simply show an
    empty warning field.
    """
    records = StudyRepository(db).list_run_records_for_study(study_id)
    rows: list[dict[str, Any]] = []
    for record in records:
        row = _summary_row_from_run_record(record)
        rows.append(row)
    return rows


def _summary_row_from_run_record(record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario_index": record.get("meta_scenario_index"),
        "scenario_label": record.get("meta_scenario_label") or record.get("scenario_name"),
        "status": record.get("run_status"),
    }

    for key, value in record.items():
        if not key.startswith("meta_"):
            continue
        clean_key = key.removeprefix("meta_")
        if clean_key in {
            "scenario_index",
            "scenario_label",
            "study_type",
            "spec_path",
            "base_config_path",
            "solver_warnings",
        }:
            continue
        row[clean_key] = value

    result = _read_persisted_result_dataframe(record.get("result_csv_path"))
    metrics = summarize_parametric_grid_record(
        {
            "scenario": _SummaryScenario(
                index=_coerce_int(row.get("scenario_index")),
                label=str(row.get("scenario_label") or record.get("scenario_name") or "scenario"),
                metadata={k: v for k, v in row.items() if k not in {"scenario_index", "scenario_label", "status"}},
            ),
            "status": row.get("status"),
            "result": result,
            "warnings": record.get("meta_solver_warnings") or "",
            "error": record.get("error_message"),
        }
    )
    # Preserve DB status/error even if the generic summarizer injected metric keys.
    metrics["status"] = row.get("status")
    metrics["error"] = record.get("error_message")
    return metrics


@dataclass(frozen=True)
class _SummaryScenario:
    """Minimal scenario shape accepted by summarize_parametric_grid_record."""

    index: int
    label: str
    metadata: dict[str, Any]


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return -1


def _read_persisted_result_dataframe(path_value: Any) -> pd.DataFrame | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


__all__ = [
    "DEFAULT_PROJECT_NAME",
    "PersistentGridRunResult",
    "PersistentGridTargetPreview",
    "load_parametric_grid_study_summary",
    "preview_parametric_grid_persistent_target",
    "run_parametric_grid_persistent",
]
