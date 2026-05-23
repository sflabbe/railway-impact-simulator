"""CLI-facing helpers for YAML-defined parametric grids.

This module keeps the reusable orchestration independent from Typer so tests can
exercise dry-runs and injected case runners without touching persistence or UI
layers.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from railway_simulator.config.loader import (
    ConfigError,
    apply_collision_to_params,
    load_simulation_config,
)
from railway_simulator.core.engine import run_simulation

from .parametric_grid import (
    ParametricScenario,
    apply_scenario_to_config,
    build_parametric_scenarios,
    preview_parametric_grid,
    run_parametric_grid_in_memory,
)
from .parametric_grid_io import (
    ParametricGridDefinition,
    ParametricGridYAMLError,
    load_parametric_grid_yaml,
)


class GridExportFormat(str, Enum):
    """Supported tabular export formats for grid previews and summaries."""

    csv = "csv"
    json = "json"
    both = "both"


RunCaseFn = Callable[[dict[str, Any], ParametricScenario], Any]

KNOWN_QUANTITIES = (
    "Impact_Force_MN",
    "Penetration_mm",
    "Acceleration_g",
)


def load_grid_definition(
    spec_path: str | Path,
    *,
    base_config_override: str | Path | None = None,
) -> tuple[ParametricGridDefinition, Path | None]:
    """Load a YAML grid spec and resolve the effective base config path."""

    definition = load_parametric_grid_yaml(spec_path)
    if base_config_override is None:
        return definition, definition.base_config_path
    return definition, Path(base_config_override).expanduser().resolve()


def preview_grid_from_yaml(
    spec_path: str | Path,
    *,
    limit: int | None = None,
    base_config_override: str | Path | None = None,
) -> tuple[ParametricGridDefinition, Path | None, list[dict[str, Any]]]:
    """Load a grid YAML and return preview rows without loading or running cases."""

    _validate_limit(limit)
    definition, base_config_path = load_grid_definition(
        spec_path,
        base_config_override=base_config_override,
    )
    rows = [_preview_row(row) for row in preview_parametric_grid(definition.grid)]
    if limit is not None:
        rows = rows[:limit]
    return definition, base_config_path, rows


def run_grid_from_yaml(
    spec_path: str | Path,
    *,
    base_config_override: str | Path | None = None,
    limit: int | None = None,
    strict: bool = False,
    run_case_fn: RunCaseFn | None = None,
) -> tuple[ParametricGridDefinition, Path, list[dict[str, Any]]]:
    """Run a YAML-defined grid in memory and return summary rows."""

    _validate_limit(limit)
    definition, base_config_path = load_grid_definition(
        spec_path,
        base_config_override=base_config_override,
    )
    if base_config_path is None:
        raise ParametricGridYAMLError("Pass --base-config or provide base.config in the spec.")

    base_config = _load_base_config(base_config_path)
    case_runner = run_case_fn or _default_run_case
    raw_results = run_parametric_grid_in_memory(
        definition.grid,
        base_config,
        case_runner,
        strict=strict,
        limit=limit,
    )
    return definition, base_config_path, [_summary_row(record) for record in raw_results]


def write_records(
    rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    export_format: GridExportFormat | str = GridExportFormat.csv,
) -> list[Path]:
    """Write records to CSV, JSON, or both and return written paths."""

    fmt = GridExportFormat(export_format)
    out = Path(out_path)
    paths = _output_paths(out, fmt)
    for path, path_format in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path_format == GridExportFormat.csv:
            pd.DataFrame(rows).to_csv(path, index=False)
        elif path_format == GridExportFormat.json:
            path.write_text(
                json.dumps(rows, indent=2, default=_json_default),
                encoding="utf-8",
            )
        else:
            raise ValueError(f"Unsupported export format: {path_format}")
    return [path for path, _fmt in paths]


def format_records_table(rows: list[dict[str, Any]]) -> str:
    """Return a readable console table for preview or summary rows."""

    if not rows:
        return "(no scenarios)"
    return pd.DataFrame(rows).to_string(index=False)


def _load_base_config(path: Path) -> dict[str, Any]:
    try:
        config = load_simulation_config(path)
        return apply_collision_to_params(config, config_path=path)
    except ConfigError as exc:
        raise ParametricGridYAMLError(str(exc)) from exc


def _default_run_case(config: dict[str, Any], _scenario: ParametricScenario) -> pd.DataFrame:
    return run_simulation(config, emit_peak_diagnostics=False)


def _preview_row(row: dict[str, Any]) -> dict[str, Any]:
    out: "OrderedDict[str, Any]" = OrderedDict()
    out["scenario_index"] = row.get("index")
    out["scenario_label"] = row.get("label")
    for key, value in row.items():
        if key not in {"index", "label"}:
            out[key] = value
    return dict(out)


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    scenario = record["scenario"]
    out: "OrderedDict[str, Any]" = OrderedDict()
    out["scenario_index"] = scenario.index
    out["scenario_label"] = scenario.label
    out["status"] = record.get("status")
    for key, value in scenario.metadata.items():
        out[key] = value

    metrics = _extract_result_metrics(record.get("result"))
    for key in (
        "peak_Impact_Force_MN",
        "peak_Penetration_mm",
        "peak_Acceleration_g",
        "t_end",
        "n_steps",
    ):
        out[key] = metrics.get(key)
    out["error"] = record.get("error")
    return dict(out)


def _extract_result_metrics(result: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    df = _find_dataframe(result)

    if isinstance(result, dict):
        for key in (
            "peak_Impact_Force_MN",
            "peak_Penetration_mm",
            "peak_Acceleration_g",
            "t_end",
            "n_steps",
        ):
            if key in result:
                metrics[key] = result[key]

    if df is None:
        return metrics

    for quantity in KNOWN_QUANTITIES:
        key = f"peak_{quantity}"
        if key not in metrics and quantity in df.columns:
            metrics[key] = _safe_nanmax(df[quantity])

    if "t_end" not in metrics:
        if "Time_s" in df.columns and len(df):
            metrics["t_end"] = _safe_last(df["Time_s"])
        elif "Time_ms" in df.columns and len(df):
            value = _safe_last(df["Time_ms"])
            metrics["t_end"] = None if value is None else float(value) / 1000.0

    if "n_steps" not in metrics:
        metrics["n_steps"] = max(int(len(df)) - 1, 0) if len(df) else 0

    return metrics


def _find_dataframe(result: Any) -> pd.DataFrame | None:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        for key in ("df", "results", "results_df", "time_history", "time_series"):
            value = result.get(key)
            if isinstance(value, pd.DataFrame):
                return value
    if isinstance(result, tuple):
        for value in result:
            if isinstance(value, pd.DataFrame):
                return value
    return None


def _safe_nanmax(series: Any) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return None
    return float(np.nanmax(values))


def _safe_last(series: Any) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if values.size == 0 or np.isnan(values[-1]):
        return None
    return float(values[-1])


def _output_paths(
    out_path: Path,
    export_format: GridExportFormat,
) -> list[tuple[Path, GridExportFormat]]:
    if export_format == GridExportFormat.both:
        return [
            (_with_suffix(out_path, ".csv"), GridExportFormat.csv),
            (_with_suffix(out_path, ".json"), GridExportFormat.json),
        ]
    return [(out_path, export_format)]


def _with_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path.with_suffix(suffix)
    return Path(f"{path}{suffix}")


def _validate_limit(limit: int | None) -> None:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative.")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return str(value)


__all__ = [
    "GridExportFormat",
    "format_records_table",
    "load_grid_definition",
    "preview_grid_from_yaml",
    "run_grid_from_yaml",
    "write_records",
]
