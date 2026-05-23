"""Pure helpers for generic parametric study grids.

This module deliberately stays independent from the solver, Workbench services,
SQLite persistence, Streamlit, and CLI entrypoints.  It only builds scenarios,
applies them to in-memory configuration dictionaries, and optionally delegates
execution to an injected callable.
"""

from __future__ import annotations

import copy
import itertools
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from . import get_by_path, set_by_path

_PATHLESS_KINDS = {"metadata", "derived"}
_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class SweepDimension:
    """One parameter, category, or metadata dimension in a cartesian sweep."""

    name: str
    kind: str
    path: str | None
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        kind = str(self.kind).strip().lower()
        path = self.path.strip() if isinstance(self.path, str) else self.path
        if path == "":
            path = None
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True)
class ParametricGridSpec:
    """Specification for a reusable cartesian parametric grid."""

    name: str
    dimensions: tuple[SweepDimension, ...]
    description: str | None = None
    quantities: tuple[str, ...] = ("Impact_Force_MN",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "quantities", tuple(self.quantities))


@dataclass(frozen=True)
class ParametricScenario:
    """One concrete scenario generated from a parametric grid."""

    index: int
    label: str
    parameters: dict[str, Any]
    metadata: dict[str, Any]


RunCaseFn = Callable[[dict[str, Any], ParametricScenario], Any]


def build_parametric_scenarios(spec: ParametricGridSpec) -> list[ParametricScenario]:
    """Build deterministic cartesian scenarios from a grid specification."""

    _validate_spec(spec)
    if not spec.dimensions:
        return [
            _make_scenario(
                index=0,
                label="base",
                parameters={},
                metadata={},
                dimensions=(),
            )
        ]

    scenarios: list[ParametricScenario] = []
    value_sets = [dimension.values for dimension in spec.dimensions]
    for index, values in enumerate(itertools.product(*value_sets)):
        parameters = {
            dimension.name: value for dimension, value in zip(spec.dimensions, values)
        }
        metadata = dict(parameters)
        label = "__".join(
            f"{_format_label_key(dimension.name)}={_format_label_value(value)}"
            for dimension, value in zip(spec.dimensions, values)
        )
        scenarios.append(
            _make_scenario(
                index=index,
                label=label,
                parameters=parameters,
                metadata=metadata,
                dimensions=spec.dimensions,
            )
        )
    return scenarios


def apply_scenario_to_config(
    base_config: dict[str, Any],
    scenario: ParametricScenario,
) -> dict[str, Any]:
    """Return a modified deep copy of ``base_config`` for one scenario."""

    if not isinstance(base_config, dict):
        raise TypeError("base_config must be a dictionary.")

    dimensions: tuple[SweepDimension, ...] = getattr(scenario, "_dimensions", ())
    if not dimensions and scenario.parameters:
        raise ValueError(
            "Scenario does not include sweep dimension definitions; "
            "build it with build_parametric_scenarios before applying it."
        )

    config = copy.deepcopy(base_config)
    for dimension in dimensions:
        if dimension.name not in scenario.parameters:
            raise ValueError(
                f"Scenario '{scenario.label}' is missing value for dimension "
                f"'{dimension.name}'."
            )
        value = scenario.parameters[dimension.name]
        if dimension.path is None:
            if dimension.kind in _PATHLESS_KINDS:
                continue
            raise ValueError(
                f"Dimension '{dimension.name}' has no path. Only metadata or "
                "derived dimensions may omit a path."
            )

        try:
            get_by_path(config, dimension.path)
        except Exception as exc:
            raise ValueError(
                f"Cannot apply dimension '{dimension.name}' to path "
                f"'{dimension.path}': path does not exist or cannot be read "
                f"({exc})."
            ) from exc

        try:
            config = set_by_path(config, dimension.path, value)
        except Exception as exc:
            raise ValueError(
                f"Cannot apply dimension '{dimension.name}' to path "
                f"'{dimension.path}': value could not be written ({exc})."
            ) from exc

    return config


def preview_parametric_grid(spec: ParametricGridSpec) -> list[dict[str, Any]]:
    """Return a simple tabular preview without running simulations or writing files."""

    return [
        {"index": scenario.index, "label": scenario.label, **scenario.metadata}
        for scenario in build_parametric_scenarios(spec)
    ]


def run_parametric_grid_in_memory(
    spec: ParametricGridSpec,
    base_config: dict[str, Any],
    run_case_fn: RunCaseFn,
    *,
    strict: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run a grid using an injected callable and return in-memory result records."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None.")

    scenarios = build_parametric_scenarios(spec)
    if limit is not None:
        scenarios = scenarios[:limit]

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        try:
            config = apply_scenario_to_config(base_config, scenario)
            result = run_case_fn(config, scenario)
        except Exception as exc:
            if strict:
                raise
            results.append(
                {
                    "scenario": scenario,
                    "status": "failed",
                    "result": None,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            continue

        results.append(
            {
                "scenario": scenario,
                "status": "ok",
                "result": result,
                "error": None,
                "error_type": None,
            }
        )

    return results


def _validate_spec(spec: ParametricGridSpec) -> None:
    if not spec.name:
        raise ValueError("ParametricGridSpec.name must not be empty.")

    seen: set[str] = set()
    for dimension in spec.dimensions:
        if not dimension.name:
            raise ValueError("SweepDimension.name must not be empty.")
        if dimension.name in seen:
            raise ValueError(f"Duplicate sweep dimension name: '{dimension.name}'.")
        seen.add(dimension.name)
        if not dimension.values:
            raise ValueError(f"SweepDimension '{dimension.name}' has no values.")


def _make_scenario(
    *,
    index: int,
    label: str,
    parameters: dict[str, Any],
    metadata: dict[str, Any],
    dimensions: tuple[SweepDimension, ...],
) -> ParametricScenario:
    scenario = ParametricScenario(
        index=index,
        label=label,
        parameters=parameters,
        metadata=metadata,
    )
    object.__setattr__(scenario, "_dimensions", dimensions)
    return scenario


def _format_label_key(name: str) -> str:
    token = _LABEL_SAFE_RE.sub("_", str(name).strip()).strip("_")
    return token or "dimension"


def _format_label_value(value: Any) -> str:
    if value is None:
        raw = "none"
    elif isinstance(value, bool):
        raw = "true" if value else "false"
    elif isinstance(value, int) and not isinstance(value, bool):
        raw = str(value)
    elif isinstance(value, float):
        raw = format(value, "g") if math.isfinite(value) else str(value).lower()
    else:
        raw = str(value)
    token = _LABEL_SAFE_RE.sub("_", raw.strip()).strip("_")
    return token or "value"


__all__ = [
    "ParametricGridSpec",
    "ParametricScenario",
    "SweepDimension",
    "apply_scenario_to_config",
    "build_parametric_scenarios",
    "preview_parametric_grid",
    "run_parametric_grid_in_memory",
]
