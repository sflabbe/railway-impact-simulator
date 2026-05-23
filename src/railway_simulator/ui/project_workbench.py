"""Project workbench UI for persisted parametric studies and SRS comparison.

The legacy Streamlit tabs execute ad-hoc studies in memory.  This module wires
those workflows to the new domain/persistence/services architecture so a study
can be re-opened and compared later without re-running the solver.
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
from railway_simulator.studies.parametric_grid import ParametricScenario
from railway_simulator.studies.parametric_grid_cli import run_grid_from_yaml
from railway_simulator.studies.parametric_grid_io import (
    ParametricGridDefinition,
    load_parametric_grid_yaml,
    parametric_grid_definition_to_preview,
)
from railway_simulator.studies.parametric_grid_persistence import (
    PersistentGridRunResult,
    load_parametric_grid_study_summary,
    run_parametric_grid_persistent,
)
from railway_simulator.reporting import ChapterBuildResult, build_latex_chapter
from railway_simulator.ui.st_compat import safe_button, safe_download_button, safe_plotly_chart


DEFAULT_PERIOD_GRID_MS = tuple(float(v) for v in np.logspace(math.log10(10.0), math.log10(3000.0), 30))
DEFAULT_PARAMETRIC_SPEC_PATH = Path("configs/studies/impact_parametric_mini.yml")
DEFAULT_PARAMETRIC_DB_PATH = Path("projects/impact_workbench/project.sqlite")
DEFAULT_PARAMETRIC_PROJECT_NAME = "impact_workbench"
PARAMETRIC_GRID_SUMMARY_COLUMNS = (
    "scenario_index",
    "scenario_label",
    "status",
    "peak_Impact_Force_MN",
    "peak_Penetration_mm",
    "peak_Acceleration_g",
    "t_end",
    "n_steps",
    "warnings",
    "error",
)

ParametricGridRunCaseFn = Callable[[dict[str, Any], ParametricScenario], Any]


@dataclass(frozen=True)
class ParametricGridUIRunResult:
    """Small UI-facing result bundle for generic parametric grids."""

    summary: pd.DataFrame
    definition: ParametricGridDefinition
    base_config_path: Path | None
    persistent_result: PersistentGridRunResult | None = None


def parse_float_csv(text: str, *, default: tuple[float, ...] = ()) -> tuple[float, ...]:
    """Parse a comma/semicolon separated list of finite floats."""
    if text is None:
        return default
    normalized = str(text).replace(";", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if not parts:
        return default
    values: list[float] = []
    for part in parts:
        value = float(part)
        if not math.isfinite(value):
            raise ValueError(f"non-finite value: {part}")
        values.append(value)
    return tuple(values)


def make_log_period_grid_ms(min_ms: float, max_ms: float, n: int) -> tuple[float, ...]:
    """Build a positive logarithmic period grid in milliseconds."""
    min_value = float(min_ms)
    max_value = float(max_ms)
    count = int(n)
    if min_value <= 0.0 or max_value <= 0.0:
        raise ValueError("period bounds must be positive")
    if max_value <= min_value:
        raise ValueError("Tn max must be greater than Tn min")
    if count < 2:
        raise ValueError("period grid must contain at least 2 points")
    return tuple(float(v) for v in np.logspace(math.log10(min_value), math.log10(max_value), count))


def curve_label(record: dict[str, Any]) -> str:
    """Human-readable label for a persisted SRS curve record."""
    parts = []
    mode = record.get("meta_mode")
    speed = record.get("meta_speed_kmh")
    contact = record.get("meta_contact_model")
    mu = record.get("meta_mu")
    if mode is not None:
        parts.append(str(mode))
    if speed is not None:
        try:
            parts.append(f"{float(speed):g} km/h")
        except Exception:
            parts.append(str(speed))
    if contact is not None:
        parts.append(str(contact))
    if mu is not None:
        try:
            parts.append(f"mu={float(mu):g}")
        except Exception:
            parts.append(f"mu={mu}")
    if not parts:
        parts.append(str(record.get("scenario_name", record.get("run_id", "curve"))))
    return " | ".join(parts)


def load_curve_dataframe(record: dict[str, Any]) -> pd.DataFrame:
    """Load one SRS curve CSV and attach metadata columns."""
    path = Path(str(record["curve_csv_path"]))
    df = pd.read_csv(path)
    df.insert(0, "curve_label", curve_label(record))
    for key in ("run_id", "curve_id", "scenario_name", "zeta", "force_column"):
        if key in record and key not in df.columns:
            df.insert(0, key, record[key])
    return df


def selected_curve_records(
    records: list[dict[str, Any]],
    *,
    zeta: float | None = None,
    modes: set[str] | None = None,
    contact_models: set[str] | None = None,
    speeds_kmh: set[float] | None = None,
) -> list[dict[str, Any]]:
    """Filter SRS record dictionaries for plotting/export."""
    out: list[dict[str, Any]] = []
    for rec in records:
        if zeta is not None and not math.isclose(float(rec.get("zeta", math.nan)), float(zeta), rel_tol=0.0, abs_tol=1e-12):
            continue
        if modes and str(rec.get("meta_mode")) not in modes:
            continue
        if contact_models and str(rec.get("meta_contact_model")) not in contact_models:
            continue
        if speeds_kmh:
            try:
                speed = float(rec.get("meta_speed_kmh"))
            except Exception:
                continue
            if not any(math.isclose(speed, s, rel_tol=0.0, abs_tol=1e-9) for s in speeds_kmh):
                continue
        out.append(rec)
    return out


def build_srs_overlay_figure(records: list[dict[str, Any]], *, value_column: str = "Feq_MN", max_curves: int = 30) -> go.Figure:
    """Build a Plotly overlay of selected SRS curves."""
    fig = go.Figure()
    for rec in records[:max_curves]:
        df = load_curve_dataframe(rec)
        if value_column not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["Tn_ms"],
                y=df[value_column],
                mode="lines",
                name=curve_label(rec),
            )
        )
    fig.update_layout(
        title="Shock Response Spectrum comparison",
        xaxis_title="Natural period Tn [ms]",
        yaxis_title=value_column,
        height=560,
        xaxis_type="log",
    )
    return fig


def build_srs_envelope_figure(records: list[dict[str, Any]], *, value_column: str = "Feq_MN") -> go.Figure:
    """Build envelope curve over selected SRS curves."""
    frames = [load_curve_dataframe(rec) for rec in records]
    env = SpectrumService.envelope(frames, value_column=value_column)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=env["Tn_ms"],
            y=env[f"{value_column}_envelope"],
            mode="lines",
            name="Envelope",
        )
    )
    fig.update_layout(
        title="SRS envelope over selected runs",
        xaxis_title="Natural period Tn [ms]",
        yaxis_title=f"{value_column} envelope",
        height=520,
        xaxis_type="log",
    )
    return fig


def build_full_vs_lok_ratio_figure(records: list[dict[str, Any]], *, value_column: str = "Feq_MN") -> go.Figure:
    """Pair Zug_full and Lok_solo curves and plot SRS ratios.

    Pairing key: speed, contact model, mu and zeta.  Missing pairs are skipped.
    """
    buckets: dict[tuple[Any, Any, Any, Any], dict[str, dict[str, Any]]] = {}
    for rec in records:
        mode = str(rec.get("meta_mode"))
        if mode not in {"lok_solo", "zug_full"}:
            continue
        key = (
            rec.get("meta_speed_kmh"),
            rec.get("meta_contact_model"),
            rec.get("meta_mu"),
            rec.get("zeta"),
        )
        buckets.setdefault(key, {})[mode] = rec

    fig = go.Figure()
    for key, pair in buckets.items():
        if "zug_full" not in pair or "lok_solo" not in pair:
            continue
        num = load_curve_dataframe(pair["zug_full"])
        den = load_curve_dataframe(pair["lok_solo"])
        ratio = SpectrumService.ratio(num, den, value_column=value_column)
        speed, contact, mu, zeta = key
        fig.add_trace(
            go.Scatter(
                x=ratio["Tn_ms"],
                y=ratio[f"{value_column}_ratio"],
                mode="lines",
                name=f"{speed:g} km/h | {contact} | mu={mu:g} | zeta={zeta:g}",
            )
        )
    fig.update_layout(
        title="SRS ratio: Zug_full / Lok_solo",
        xaxis_title="Natural period Tn [ms]",
        yaxis_title=f"{value_column} ratio [-]",
        height=520,
        xaxis_type="log",
    )
    return fig


def _open_db_context(db_path: str | Path) -> tuple[ProjectDatabase, Any]:
    db = initialize_project_database(db_path)
    projects = ProjectRepository(db).list()
    if not projects:
        raise ValueError("No project row found in this database. Create a project first.")
    return db, projects[0]


def _available_values(records: list[dict[str, Any]], key: str) -> list[Any]:
    values = []
    for rec in records:
        value = rec.get(key)
        if value is not None and value not in values:
            values.append(value)
    try:
        return sorted(values)
    except TypeError:
        return values


def _studies_dataframe(studies: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": study.id,
                "name": study.name,
                "type": study.study_type,
                "status": study.status,
                "created_at": study.created_at,
                "finished_at": study.finished_at,
            }
            for study in studies
        ]
    )


def _run_records_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    keep_order = [
        "run_id",
        "scenario_name",
        "run_status",
        "elapsed_s",
        "meta_mode",
        "meta_speed_kmh",
        "meta_contact_model",
        "meta_mu",
        "result_csv_path",
        "error_message",
    ]
    df = pd.DataFrame(records)
    cols = [c for c in keep_order if c in df.columns] + [c for c in df.columns if c not in keep_order]
    return df[cols]


def _curve_records_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    keep_order = [
        "curve_id",
        "scenario_name",
        "zeta",
        "meta_mode",
        "meta_speed_kmh",
        "meta_contact_model",
        "meta_mu",
        "curve_csv_path",
    ]
    df = pd.DataFrame(records)
    cols = [c for c in keep_order if c in df.columns] + [c for c in df.columns if c not in keep_order]
    return df[cols]


def normalize_parametric_limit(value: int | None) -> int | None:
    """Normalize UI limit semantics: None/0 means run every scenario."""
    if value is None:
        return None
    limit = int(value)
    if limit < 0:
        raise ValueError("limit scenarios must be non-negative")
    return None if limit == 0 else limit


def load_parametric_grid_ui_spec(spec_path: str | Path) -> ParametricGridDefinition:
    """Load a YAML parametric grid spec for the Workbench UI."""
    return load_parametric_grid_yaml(spec_path)


def parametric_grid_spec_metadata(definition: ParametricGridDefinition) -> dict[str, Any]:
    """Return basic display metadata for a loaded parametric grid spec."""
    return {
        "study_name": definition.grid.name,
        "description": definition.grid.description or "",
        "base_config_path": str(definition.base_config_path) if definition.base_config_path is not None else "",
        "dimensions": len(definition.grid.dimensions),
        "quantities": ", ".join(definition.grid.quantities),
        "srs_enabled": bool(definition.srs.get("enabled", False)),
    }


def preview_parametric_grid_ui_scenarios(spec_path: str | Path) -> pd.DataFrame:
    """Preview YAML scenarios without executing the solver or touching SQLite."""
    definition = load_parametric_grid_ui_spec(spec_path)
    rows = parametric_grid_definition_to_preview(definition)
    preview_rows = []
    for row in rows:
        preview_rows.append(
            {
                "scenario_index": row.get("index"),
                "scenario_label": row.get("label"),
                **{key: value for key, value in row.items() if key not in {"index", "label"}},
            }
        )
    return pd.DataFrame(preview_rows)


def parametric_grid_summary_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the Workbench summary table with stable front columns."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    front = [col for col in PARAMETRIC_GRID_SUMMARY_COLUMNS if col in df.columns]
    metric_cols = set(PARAMETRIC_GRID_SUMMARY_COLUMNS)
    metadata_cols = [
        col
        for col in df.columns
        if col not in metric_cols and col not in {"scenario_index", "scenario_label", "status"}
    ]
    ordered = [
        col
        for col in ("scenario_index", "scenario_label", "status")
        if col in df.columns
    ]
    ordered.extend(metadata_cols)
    ordered.extend(col for col in front if col not in ordered)
    ordered.extend(col for col in df.columns if col not in ordered)
    return df[ordered]


def parametric_grid_warning_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return rows with non-empty warnings for prominent UI display."""
    if summary.empty or "warnings" not in summary.columns:
        return pd.DataFrame()
    warnings = summary["warnings"].fillna("").astype(str).str.strip()
    mask = warnings.ne("") & warnings.str.lower().ne("nan")
    cols = [
        col
        for col in ("scenario_index", "scenario_label", "status", "warnings")
        if col in summary.columns
    ]
    return summary.loc[mask, cols].copy()


def parametric_grid_peak_force_chart_data(summary: pd.DataFrame) -> pd.DataFrame:
    """Return chart-ready peak force data, or an empty frame when unavailable."""
    required = {"scenario_label", "peak_Impact_Force_MN"}
    if summary.empty or not required.issubset(summary.columns):
        return pd.DataFrame(columns=["scenario_label", "peak_Impact_Force_MN"])
    out = summary[["scenario_label", "peak_Impact_Force_MN"]].copy()
    out["peak_Impact_Force_MN"] = pd.to_numeric(out["peak_Impact_Force_MN"], errors="coerce")
    return out.dropna(subset=["peak_Impact_Force_MN"])


def build_parametric_peak_force_figure(summary: pd.DataFrame) -> go.Figure | None:
    """Build the basic peak force bar chart shown after a grid run."""
    chart_data = parametric_grid_peak_force_chart_data(summary)
    if chart_data.empty:
        return None
    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_data["scenario_label"],
                y=chart_data["peak_Impact_Force_MN"],
                name="Peak force",
            )
        ]
    )
    fig.update_layout(
        title="Peak impact force by scenario",
        xaxis_title="Scenario",
        yaxis_title="Peak Impact_Force_MN",
        height=420,
    )
    return fig


def parametric_grid_summary_csv_bytes(summary: pd.DataFrame) -> bytes:
    """Generate UTF-8 CSV bytes for the summary download button."""
    return summary.to_csv(index=False).encode("utf-8")


def parametric_grid_peak_force_csv_bytes(summary: pd.DataFrame) -> bytes:
    """Generate CSV bytes for the peak-force plot data."""
    return parametric_grid_peak_force_chart_data(summary).to_csv(index=False).encode("utf-8")


def parametric_grid_peak_force_html_bytes(summary: pd.DataFrame) -> bytes:
    """Generate a standalone Plotly HTML export for the peak-force chart."""
    fig = build_parametric_peak_force_figure(summary)
    if fig is None:
        return b""
    return fig.to_html(include_plotlyjs=True, full_html=True).encode("utf-8")


def load_saved_parametric_grid_summary(db: ProjectDatabase, study_id: str) -> pd.DataFrame:
    """Load a saved parametric-grid study summary without re-running cases."""
    return parametric_grid_summary_dataframe(load_parametric_grid_study_summary(db, study_id))


def run_parametric_grid_ui_study(
    spec_path: str | Path,
    *,
    db_path: str | Path | None = None,
    project_name: str | None = DEFAULT_PARAMETRIC_PROJECT_NAME,
    study_name: str | None = None,
    limit: int | None = 1,
    strict: bool = False,
    persist_results: bool = True,
    run_case_fn: ParametricGridRunCaseFn | None = None,
) -> ParametricGridUIRunResult:
    """Run a generic parametric grid for the Workbench UI.

    Persistent runs use the same SQLite runner as the CLI. Non-persistent runs
    use the existing in-memory orchestration.
    """
    normalized_limit = normalize_parametric_limit(limit)
    resolved_study_name = study_name.strip() if isinstance(study_name, str) and study_name.strip() else None

    if persist_results:
        if db_path is None:
            raise ValueError("db_path is required when persist_results=True")
        persistent = run_parametric_grid_persistent(
            spec_path,
            db_path=db_path,
            project_name=project_name or None,
            study_name=resolved_study_name,
            limit=normalized_limit,
            strict=strict,
            run_case_fn=run_case_fn,
        )
        return ParametricGridUIRunResult(
            summary=parametric_grid_summary_dataframe(persistent.rows),
            definition=load_parametric_grid_ui_spec(spec_path),
            base_config_path=persistent.base_config_path,
            persistent_result=persistent,
        )

    definition, base_config_path, rows = run_grid_from_yaml(
        spec_path,
        limit=normalized_limit,
        strict=strict,
        run_case_fn=run_case_fn,
    )
    return ParametricGridUIRunResult(
        summary=parametric_grid_summary_dataframe(rows),
        definition=definition,
        base_config_path=base_config_path,
    )


def list_parametric_grid_studies(db: ProjectDatabase, project_id: str) -> list[Any]:
    """List saved generic parametric grid studies, newest first."""
    studies = [
        study
        for study in StudyRepository(db).list_studies(project_id)
        if study.study_type == "parametric_grid"
    ]
    return list(reversed(studies))


def sanitize_path_component(value: str, *, fallback: str = "study") -> str:
    """Return a filesystem-safe, human-readable path component."""
    raw = str(value or "").strip().lower()
    chars: list[str] = []
    previous_sep = False
    for char in raw:
        if char.isalnum():
            chars.append(char)
            previous_sep = False
        elif char in {"_", "-", "."}:
            chars.append(char)
            previous_sep = False
        elif not previous_sep:
            chars.append("_")
            previous_sep = True
    out = "".join(chars).strip("._-")
    return out or fallback


def default_chapter_output_dir(project_root: Path, *, study_name: str, study_id: str) -> Path:
    """Default output folder for a thesis chapter bundle generated from the UI."""
    slug = sanitize_path_component(study_name)
    short_id = sanitize_path_component(study_id, fallback="study")[:8]
    return project_root / "reports" / f"chapter_{slug}_{short_id}"


def build_workbench_chapter_bundle(
    *,
    db: ProjectDatabase,
    study_id: str,
    output_dir: str | Path,
    title: str,
    author: str,
    zeta: float | None = 0.05,
) -> ChapterBuildResult:
    """Build the LaTeX chapter bundle used by the Streamlit workbench."""
    return build_latex_chapter(
        db=db,
        study_id=study_id,
        output_dir=Path(output_dir),
        title=title,
        author=author,
        zeta=zeta,
    )


def zip_report_bundle_bytes(output_dir: str | Path) -> bytes:
    """Zip a generated report bundle and return the archive bytes for download."""
    root = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"report bundle directory does not exist: {root}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(root).as_posix())
    return buffer.getvalue()


def _render_parametric_grid_outputs(st: Any, summary: pd.DataFrame, *, key_prefix: str) -> None:
    """Render summary, warnings, chart, and CSV download for a grid result."""
    if summary.empty:
        st.info("No summary rows available.")
        return

    st.markdown("#### Summary")
    st.dataframe(summary)

    warning_rows = parametric_grid_warning_rows(summary)
    if not warning_rows.empty:
        st.warning("Warnings were reported for one or more scenarios.")
        st.dataframe(warning_rows)

    st.markdown("#### Peak force")
    fig = build_parametric_peak_force_figure(summary)
    if fig is None:
        st.info("No peak force metric available for this run.")
    else:
        safe_plotly_chart(st, fig, width="stretch")
        col_plot_data, col_plot_html = st.columns(2)
        with col_plot_data:
            safe_download_button(
                st,
                label="Download peak-force plot data CSV",
                data=parametric_grid_peak_force_csv_bytes(summary),
                file_name="parametric_grid_peak_force_data.csv",
                mime="text/csv",
                width="stretch",
                key=f"{key_prefix}_peak_force_data_csv",
            )
        with col_plot_html:
            safe_download_button(
                st,
                label="Download peak-force plot HTML",
                data=parametric_grid_peak_force_html_bytes(summary),
                file_name="parametric_grid_peak_force.html",
                mime="text/html",
                width="stretch",
                key=f"{key_prefix}_peak_force_html",
            )

    safe_download_button(
        st,
        label="Download summary CSV",
        data=parametric_grid_summary_csv_bytes(summary),
        file_name="parametric_grid_summary.csv",
        mime="text/csv",
        width="stretch",
        key=f"{key_prefix}_summary_csv",
    )


def _render_custom_parametric_grid_section(
    *,
    active_db_path: str | Path | None = None,
    active_db: ProjectDatabase | None = None,
    active_project_id: str | None = None,
) -> None:
    """Render the generic YAML-defined parametric grid UI."""
    import streamlit as st

    st.markdown("#### Custom parametric grid")

    default_db = str(active_db_path or DEFAULT_PARAMETRIC_DB_PATH)
    col_left, col_right = st.columns(2)
    with col_left:
        spec_path_text = st.text_input(
            "Spec path",
            value=st.session_state.get("wb_grid_spec_path", str(DEFAULT_PARAMETRIC_SPEC_PATH)),
            key="wb_grid_spec_path_input",
        )
        project_name = st.text_input(
            "Project name",
            value=st.session_state.get("wb_grid_project_name", DEFAULT_PARAMETRIC_PROJECT_NAME),
            key="wb_grid_project_name_input",
        )
        limit_value = st.number_input(
            "Limit scenarios",
            min_value=0,
            value=1,
            step=1,
            key="wb_grid_limit",
            help="Use 0 to run all scenarios.",
        )
    with col_right:
        db_path_text = st.text_input(
            "DB path",
            value=st.session_state.get("wb_grid_db_path", default_db),
            key="wb_grid_db_path_input",
        )
        study_name = st.text_input(
            "Study name",
            value=st.session_state.get("wb_grid_study_name", ""),
            key="wb_grid_study_name_input",
            help="Leave blank to use study.name from YAML.",
        )
        strict_mode = st.checkbox("Strict mode", value=False, key="wb_grid_strict")
        persist_results = st.checkbox("Persist results", value=True, key="wb_grid_persist")

    col_load, col_preview, col_run = st.columns(3)
    with col_load:
        load_clicked = safe_button(st, "Load spec", width="stretch", key="wb_grid_load")
    with col_preview:
        preview_clicked = safe_button(st, "Preview scenarios", width="stretch", key="wb_grid_preview")
    with col_run:
        run_clicked = safe_button(st, "Run study", type="primary", width="stretch", key="wb_grid_run")

    if load_clicked:
        try:
            definition = load_parametric_grid_ui_spec(spec_path_text)
            st.session_state["wb_grid_metadata"] = parametric_grid_spec_metadata(definition)
            st.session_state["wb_grid_spec_path"] = spec_path_text
            st.success(f"Loaded spec: {definition.grid.name}")
        except Exception as exc:
            st.session_state.pop("wb_grid_metadata", None)
            st.error(f"Could not load parametric grid spec: {exc}")

    metadata = st.session_state.get("wb_grid_metadata")
    if metadata:
        st.markdown("#### Spec metadata")
        st.dataframe(pd.DataFrame([metadata]))

    if preview_clicked:
        try:
            preview_df = preview_parametric_grid_ui_scenarios(spec_path_text)
            st.session_state["wb_grid_preview_df"] = preview_df
            st.success(f"Previewed {len(preview_df)} scenario(s).")
        except Exception as exc:
            st.session_state.pop("wb_grid_preview_df", None)
            st.error(f"Could not preview scenarios: {exc}")

    preview_df = st.session_state.get("wb_grid_preview_df")
    if isinstance(preview_df, pd.DataFrame) and not preview_df.empty:
        st.markdown("#### Scenario preview")
        st.dataframe(preview_df)

    if run_clicked:
        try:
            with st.spinner("Running parametric grid study..."):
                result = run_parametric_grid_ui_study(
                    spec_path_text,
                    db_path=db_path_text,
                    project_name=project_name,
                    study_name=study_name,
                    limit=int(limit_value),
                    strict=bool(strict_mode),
                    persist_results=bool(persist_results),
                )
            st.session_state["wb_grid_summary_df"] = result.summary
            st.session_state["wb_grid_db_path"] = db_path_text
            st.session_state["wb_grid_project_name"] = project_name
            st.session_state["wb_grid_study_name"] = study_name
            if result.persistent_result is not None:
                st.session_state["wb_db_path"] = str(result.persistent_result.db.path)
                st.session_state["wb_project_id"] = result.persistent_result.project.id
                st.session_state["wb_last_study_id"] = result.persistent_result.study.id
                st.success(
                    f"Study stored: {result.persistent_result.study.name} "
                    f"({len(result.summary)} scenario(s))"
                )
            else:
                st.success(f"Study completed in memory: {len(result.summary)} scenario(s)")
        except Exception as exc:
            st.error(f"Could not run parametric grid study: {exc}")

    summary_df = st.session_state.get("wb_grid_summary_df")
    if isinstance(summary_df, pd.DataFrame):
        _render_parametric_grid_outputs(st, summary_df, key_prefix="wb_grid")

    if active_db is not None and active_project_id is not None:
        st.markdown("#### Saved parametric grid studies")
        studies = list_parametric_grid_studies(active_db, active_project_id)
        if not studies:
            st.info("No saved parametric grid studies in this project yet.")
        else:
            study_options = {f"{s.created_at} | {s.name} | {s.status} | {s.id}": s for s in studies}
            label = st.selectbox(
                "Saved study",
                list(study_options),
                key="wb_grid_saved_study",
            )
            selected_study = study_options[label]
            summary = load_saved_parametric_grid_summary(active_db, selected_study.id)
            if summary.empty:
                st.info("Selected study has no persisted runs yet.")
            else:
                _render_parametric_grid_outputs(st, summary, key_prefix=f"wb_grid_saved_{selected_study.id}")
            with st.expander("Raw persisted run records", expanded=False):
                records = StudyRepository(active_db).list_run_records_for_study(selected_study.id)
                st.dataframe(_run_records_dataframe(records))


def render_project_workbench(params: dict[str, Any]) -> None:
    """Render the persisted project/study/SRS workflow inside Streamlit."""
    import streamlit as st

    st.title("Project Workbench")
    st.markdown(
        "Persist parametric studies in a project SQLite database, compute SRS curves, "
        "and compare saved runs without launching the solver again."
    )

    # ------------------------------------------------------------------
    # Project open/create
    # ------------------------------------------------------------------
    st.markdown("### Project database")
    col_new, col_open = st.columns(2)

    with col_new:
        st.markdown("#### Create project")
        default_root = Path.cwd() / "projects" / "impact_workbench"
        project_name = st.text_input("Project name", value="impact_workbench", key="wb_project_name")
        project_desc = st.text_area("Description", value="Persisted railway-impact study workspace", key="wb_project_desc")
        project_root = st.text_input("Project folder", value=str(default_root), key="wb_project_root")
        if safe_button(st, "Create / initialize project", type="primary", width="stretch", key="wb_create_project"):
            project, db = ProjectService().create_project(
                name=project_name,
                root_dir=project_root,
                description=project_desc,
            )
            st.session_state["wb_db_path"] = str(db.path)
            st.session_state["wb_project_id"] = project.id
            st.success(f"Project ready: {project.name} ({db.path})")

    with col_open:
        st.markdown("#### Open project")
        db_path_text = st.text_input(
            "Existing project.sqlite",
            value=st.session_state.get("wb_db_path", str(default_root / "project.sqlite")),
            key="wb_open_db_path",
        )
        if safe_button(st, "Open database", width="stretch", key="wb_open_project"):
            try:
                db, project = _open_db_context(db_path_text)
                st.session_state["wb_db_path"] = str(db.path)
                st.session_state["wb_project_id"] = project.id
                st.success(f"Opened project: {project.name}")
            except Exception as exc:
                st.error(f"Could not open project database: {exc}")

    db_path = st.session_state.get("wb_db_path")
    if not db_path:
        st.info("Create or open a project database to enable persisted studies.")
        st.markdown("### Parametric studies")
        _render_custom_parametric_grid_section(active_db_path=DEFAULT_PARAMETRIC_DB_PATH)
        return

    try:
        db, project = _open_db_context(db_path)
    except Exception as exc:
        st.error(f"Active project database is not usable: {exc}")
        st.markdown("### Parametric studies")
        _render_custom_parametric_grid_section(active_db_path=db_path)
        return

    project_root = project.normalized_root()
    st.caption(f"Active project: **{project.name}** · `{db.path}`")

    study_repo = StudyRepository(db)
    config_repo = ConfigSnapshotRepository(db)
    spectrum_repo = SpectrumRepository(db)

    sub_grid, sub_launch, sub_compare, sub_report, sub_browser = st.tabs(
        ["Parametric studies", "Launch full-train study", "SRS comparison", "Report bundle", "Database browser"]
    )

    # ------------------------------------------------------------------
    # Generic YAML parametric grid launcher
    # ------------------------------------------------------------------
    with sub_grid:
        st.markdown("### Parametric studies")
        _render_custom_parametric_grid_section(
            active_db_path=db.path,
            active_db=db,
            active_project_id=project.id,
        )

    # ------------------------------------------------------------------
    # Full-train launcher
    # ------------------------------------------------------------------
    with sub_launch:
        st.markdown("### Full-train parametric study")
        st.write(
            "This launcher snapshots the current sidebar configuration and then varies "
            "vehicle mode, speed, contact model and friction. Results are persisted under "
            "the active project."
        )

        left, right = st.columns([1, 1])
        with left:
            study_name = st.text_input("Study name", value="consist_srs_comparison", key="wb_full_train_name")
            modes = st.multiselect(
                "Vehicle modes",
                options=["lok_solo", "zug_full"],
                default=["lok_solo", "zug_full"],
                key="wb_full_train_modes",
            )
            speeds_text = st.text_input("Speeds [km/h]", value="20,40", key="wb_full_train_speeds")
            mu_text = st.text_input("Friction values mu", value="0.30", key="wb_full_train_mu")
            contact_models = st.multiselect(
                "Contact models",
                options=["anagnostopoulos", "flores", "lankarani-nikravesh", "ye", "hooke", "hertz"],
                default=["anagnostopoulos"],
                key="wb_full_train_contacts",
            )
        with right:
            compute_srs = st.checkbox("Compute and store SRS curves", value=True, key="wb_full_train_compute_srs")
            zeta_text = st.text_input("SRS damping ratios zeta", value="0.05", key="wb_full_train_zeta")
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                Tn_min = st.number_input("Tn min [ms]", min_value=1.0, value=10.0, key="wb_Tn_min")
            with col_t2:
                Tn_max = st.number_input("Tn max [ms]", min_value=2.0, value=3000.0, key="wb_Tn_max")
            with col_t3:
                Tn_n = st.number_input("Tn points", min_value=2, max_value=200, value=30, step=1, key="wb_Tn_n")

        try:
            speeds = parse_float_csv(speeds_text)
            mus = parse_float_csv(mu_text)
            zetas = parse_float_csv(zeta_text)
            periods = make_log_period_grid_ms(Tn_min, Tn_max, int(Tn_n))
            n_scen = len(modes) * len(speeds) * len(mus) * len(contact_models)
            st.info(f"Study size: {n_scen} simulations · {len(zetas) if compute_srs else 0} SRS curve(s) per successful run")
            valid_spec = bool(modes and speeds and mus and contact_models and zetas)
        except Exception as exc:
            st.error(f"Invalid study definition: {exc}")
            valid_spec = False
            speeds = mus = zetas = periods = ()
            n_scen = 0

        if n_scen > 40:
            st.warning(
                "This is a large synchronous Streamlit run. For heavy production sweeps, prefer the CLI or a smaller first pass."
            )

        if safe_button(st, "Run persisted full-train study", type="primary", width="stretch", key="wb_run_full_train"):
            if not valid_spec:
                st.error("Fix the study definition before running.")
            else:
                snapshot = config_repo.create(project.id, dict(params), source_path=None)
                spec = FullTrainStudySpec(
                    project_id=project.id,
                    base_config_id=snapshot.id,
                    name=study_name,
                    modes=tuple(modes),
                    speeds_kmh=tuple(float(v) for v in speeds),
                    contact_models=tuple(contact_models),
                    mu_values=tuple(float(v) for v in mus),
                    zeta_srs=tuple(float(v) for v in zetas),
                    Tn_grid_ms=tuple(float(v) for v in periods),
                )
                progress = st.progress(0.0)
                status_box = st.empty()

                def _progress(i: int, total: int, scenario: Any, run: Any) -> None:
                    progress.progress(i / max(total, 1))
                    status_box.write(f"{i}/{total}: {scenario.name} → {run.status}")

                runner = FullTrainStudyRunner(
                    simulation_service=SimulationService(
                        repository=study_repo,
                        runs_dir=project_root / "runs",
                    ),
                    study_repository=study_repo,
                    spectrum_service=SpectrumService(
                        repository=spectrum_repo,
                        spectra_dir=project_root / "spectra",
                    ),
                )
                with st.spinner("Running persisted full-train study..."):
                    result = runner.run(
                        spec=spec,
                        base_config=dict(params),
                        compute_srs=compute_srs,
                        progress_callback=_progress,
                    )
                study = result["study"]
                runs = result["runs"]
                failed = sum(1 for run in runs if run.status != "ok")
                st.session_state["wb_last_study_id"] = study.id
                st.success(f"Study stored: {study.name} · {len(runs)} run(s), {failed} failed")

    # ------------------------------------------------------------------
    # SRS comparison
    # ------------------------------------------------------------------
    with sub_compare:
        st.markdown("### SRS comparison from saved studies")
        studies = study_repo.list_studies(project.id)
        if not studies:
            st.info("No studies in this project yet.")
        else:
            study_options = {f"{s.name} · {s.status} · {s.id}": s for s in studies}
            labels = list(study_options)
            default_index = 0
            last_id = st.session_state.get("wb_last_study_id")
            if last_id:
                for i, label in enumerate(labels):
                    if study_options[label].id == last_id:
                        default_index = i
                        break
            selected_label = st.selectbox("Study", labels, index=default_index, key="wb_srs_study")
            selected_study = study_options[selected_label]
            records = spectrum_repo.list_curve_records_for_study(selected_study.id)
            if not records:
                st.warning("This study has no persisted SRS curves. Re-run it with SRS enabled.")
            else:
                rec_df = _curve_records_dataframe(records)
                with st.expander("Persisted SRS curves", expanded=False):
                    st.dataframe(rec_df)

                zeta_values = _available_values(records, "zeta")
                mode_values = [str(v) for v in _available_values(records, "meta_mode")]
                contact_values = [str(v) for v in _available_values(records, "meta_contact_model")]
                speed_values = [float(v) for v in _available_values(records, "meta_speed_kmh")]

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    zeta_sel = st.selectbox("zeta", zeta_values, index=0, key="wb_srs_zeta")
                    mode_sel = st.multiselect("Modes", mode_values, default=mode_values, key="wb_srs_modes")
                with col_b:
                    contact_sel = st.multiselect(
                        "Contact models",
                        contact_values,
                        default=contact_values,
                        key="wb_srs_contacts",
                    )
                    speed_sel = st.multiselect(
                        "Speeds [km/h]",
                        speed_values,
                        default=speed_values,
                        key="wb_srs_speeds",
                    )
                with col_c:
                    max_curves = int(st.number_input("Max overlay curves", min_value=1, max_value=100, value=30, key="wb_srs_max_curves"))
                    show_envelope = st.checkbox("Show envelope", value=True, key="wb_srs_show_envelope")
                    show_ratio = st.checkbox("Show Zug/Lok ratio", value=True, key="wb_srs_show_ratio")

                filtered = selected_curve_records(
                    records,
                    zeta=float(zeta_sel),
                    modes=set(mode_sel),
                    contact_models=set(contact_sel),
                    speeds_kmh=set(float(v) for v in speed_sel),
                )
                st.caption(f"Selected curves: {len(filtered)}")
                if filtered:
                    fig = build_srs_overlay_figure(filtered, max_curves=max_curves)
                    safe_plotly_chart(st, fig, width="stretch")
                    if show_envelope and len(filtered) >= 1:
                        try:
                            safe_plotly_chart(st, build_srs_envelope_figure(filtered), width="stretch")
                        except Exception as exc:
                            st.warning(f"Envelope not available: {exc}")
                    if show_ratio:
                        try:
                            ratio_fig = build_full_vs_lok_ratio_figure(filtered)
                            if ratio_fig.data:
                                safe_plotly_chart(st, ratio_fig, width="stretch")
                            else:
                                st.info("No matching Zug_full/Lok_solo SRS pairs found for the current filters.")
                        except Exception as exc:
                            st.warning(f"Ratio plot not available: {exc}")

                    try:
                        long_df = pd.concat([load_curve_dataframe(rec) for rec in filtered], ignore_index=True)
                        safe_download_button(
                            st,
                            label="Export selected SRS curves as CSV",
                            data=long_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"srs_curves__{selected_study.id}.csv",
                            mime="text/csv",
                            width="stretch",
                            key="wb_srs_export_csv",
                        )
                    except Exception as exc:
                        st.warning(f"CSV export not available: {exc}")
                else:
                    st.info("No curves match the current filters.")

    # ------------------------------------------------------------------
    # LaTeX report bundle
    # ------------------------------------------------------------------
    with sub_report:
        st.markdown("### Thesis chapter bundle")
        st.write(
            "Generate a reproducible LaTeX chapter bundle from one persisted study. "
            "The bundle contains `main.tex`, the chapter source, bibliography, figures, "
            "CSV summaries and metadata."
        )
        studies = study_repo.list_studies(project.id)
        if not studies:
            st.info("No studies in this project yet.")
        else:
            study_options = {f"{s.name} · {s.status} · {s.id}": s for s in studies}
            labels = list(study_options)
            default_index = 0
            last_id = st.session_state.get("wb_last_study_id")
            if last_id:
                for i, label in enumerate(labels):
                    if study_options[label].id == last_id:
                        default_index = i
                        break
            report_label = st.selectbox("Study", labels, index=default_index, key="wb_report_study")
            report_study = study_options[report_label]
            default_out = default_chapter_output_dir(
                project_root,
                study_name=report_study.name,
                study_id=report_study.id,
            )

            report_title = st.text_input(
                "Chapter title",
                value="Parametric Demand Study for Railway Vehicle Impact on Trackside Structures",
                key="wb_report_title",
            )
            report_author = st.text_input("Author", value="S. Labbe", key="wb_report_author")
            zeta_values = _available_values(spectrum_repo.list_curve_records_for_study(report_study.id), "zeta")
            zeta_default = float(zeta_values[0]) if zeta_values else 0.05
            report_zeta = st.number_input(
                "SRS damping ratio for figures",
                min_value=0.0,
                value=zeta_default,
                step=0.01,
                format="%.4f",
                key="wb_report_zeta",
            )
            report_output_dir = st.text_input(
                "Output folder",
                value=str(default_out),
                key="wb_report_output_dir",
            )

            cols = st.columns([1, 1])
            with cols[0]:
                build_clicked = safe_button(
                    st,
                    "Generate LaTeX chapter bundle",
                    type="primary",
                    width="stretch",
                    key="wb_report_build",
                )
            with cols[1]:
                st.caption("Use the CLI for batch generation: `railway-sim report build-chapter ...`")

            if build_clicked:
                try:
                    with st.spinner("Generating LaTeX chapter bundle..."):
                        report_result = build_workbench_chapter_bundle(
                            db=db,
                            study_id=report_study.id,
                            output_dir=report_output_dir,
                            title=report_title,
                            author=report_author,
                            zeta=float(report_zeta),
                        )
                    st.session_state["wb_last_report_dir"] = str(report_result.output_dir)
                    st.success(f"Report bundle generated: {report_result.output_dir}")
                    st.write(
                        {
                            "chapter": str(report_result.chapter_tex),
                            "main": str(report_result.main_tex),
                            "bibliography": str(report_result.bibliography_bib),
                            "metadata": str(report_result.metadata_json),
                            "figures": len(report_result.figures),
                            "tables": len(report_result.tables),
                        }
                    )
                except Exception as exc:
                    st.error(f"Could not generate report bundle: {exc}")

            last_report_dir = st.session_state.get("wb_last_report_dir") or report_output_dir
            try:
                report_dir_path = Path(last_report_dir)
                if report_dir_path.is_dir():
                    st.markdown("#### Existing bundle files")
                    files = sorted(
                        str(path.relative_to(report_dir_path))
                        for path in report_dir_path.rglob("*")
                        if path.is_file()
                    )
                    st.dataframe(pd.DataFrame({"file": files}))
                    safe_download_button(
                        st,
                        label="Download report bundle as ZIP",
                        data=zip_report_bundle_bytes(report_dir_path),
                        file_name=f"{report_dir_path.name}.zip",
                        mime="application/zip",
                        width="stretch",
                        key="wb_report_download_zip",
                    )
            except Exception as exc:
                st.warning(f"Report ZIP not available: {exc}")

    # ------------------------------------------------------------------
    # Browser
    # ------------------------------------------------------------------
    with sub_browser:
        st.markdown("### Database browser")
        studies = study_repo.list_studies(project.id)
        st.markdown("#### Studies")
        st.dataframe(_studies_dataframe(studies))
        if studies:
            selected = st.selectbox(
                "Inspect study",
                [f"{s.name} · {s.id}" for s in studies],
                key="wb_browser_study",
            )
            study_id = selected.rsplit(" · ", 1)[-1]
            run_records = study_repo.list_run_records_for_study(study_id)
            curve_records = spectrum_repo.list_curve_records_for_study(study_id)
            st.markdown("#### Runs")
            st.dataframe(_run_records_dataframe(run_records))
            st.markdown("#### SRS curves")
            st.dataframe(_curve_records_dataframe(curve_records))
