"""YAML IO helpers for generic parametric grid study definitions.

This module only translates declarative YAML into the pure
``ParametricGridSpec`` model. It does not run simulations, write files, or
interact with persistence, UI, or CLI layers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .parametric_grid import ParametricGridSpec, SweepDimension, preview_parametric_grid

_DEFAULT_QUANTITIES = ("Impact_Force_MN",)


class ParametricGridYAMLError(ValueError):
    """Raised when a parametric grid YAML definition is invalid."""


@dataclass(frozen=True)
class ParametricGridDefinition:
    """Loaded YAML definition plus the pure grid spec used for previews."""

    grid: ParametricGridSpec
    base_config_path: Path | None
    srs: dict[str, Any]
    raw: dict[str, Any]


def load_parametric_grid_yaml(path: str | Path) -> ParametricGridDefinition:
    """Load a YAML parametric grid definition without running any cases."""

    spec_path = Path(path)
    try:
        data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ParametricGridYAMLError(f"YAML file not found: {spec_path}") from exc
    except yaml.YAMLError as exc:
        raise ParametricGridYAMLError(f"{spec_path}: invalid YAML: {exc}") from exc

    raw = _as_mapping(data, "root")
    study = _required_mapping(raw, "study")
    study_name = _required_text(study, "study.name")
    study_type = _required_text(study, "study.type")
    if study_type != "parametric_grid":
        raise ParametricGridYAMLError(
            "study.type must be 'parametric_grid' for a parametric grid definition."
        )

    description = study.get("description")
    if description is not None:
        description = str(description)

    dimensions_raw = _required_sequence(raw, "dimensions")
    dimensions = tuple(_parse_dimension(item, index) for index, item in enumerate(dimensions_raw))
    quantities = _parse_quantities(raw.get("outputs"))

    srs = raw.get("srs", {})
    if srs is None:
        srs = {}
    if not isinstance(srs, Mapping):
        raise ParametricGridYAMLError("srs must be a mapping when provided.")

    return ParametricGridDefinition(
        grid=ParametricGridSpec(
            name=study_name,
            description=description,
            dimensions=dimensions,
            quantities=quantities,
        ),
        base_config_path=_parse_base_config_path(raw.get("base"), spec_path),
        srs=dict(srs),
        raw=raw,
    )


def parametric_grid_definition_to_preview(
    definition: ParametricGridDefinition,
) -> list[dict[str, Any]]:
    """Return a tabular preview for a loaded definition without running the solver."""

    return preview_parametric_grid(definition.grid)


def _parse_dimension(item: Any, index: int) -> SweepDimension:
    field = f"dimensions[{index}]"
    dimension = _as_mapping(item, field)
    values = _required_sequence(dimension, f"{field}.values")
    if not values:
        raise ParametricGridYAMLError(f"{field}.values must contain at least one value.")

    path = dimension.get("path")
    if path is not None and not isinstance(path, str):
        raise ParametricGridYAMLError(f"{field}.path must be a string when provided.")

    return SweepDimension(
        name=_required_text(dimension, f"{field}.name"),
        kind=_required_text(dimension, f"{field}.kind"),
        path=path,
        values=tuple(values),
    )


def _parse_quantities(outputs: Any) -> tuple[str, ...]:
    if outputs is None:
        return _DEFAULT_QUANTITIES
    outputs_map = _as_mapping(outputs, "outputs")
    quantities = outputs_map.get("quantities")
    if quantities is None:
        return _DEFAULT_QUANTITIES
    quantities_seq = _as_sequence(quantities, "outputs.quantities")
    if not quantities_seq:
        raise ParametricGridYAMLError("outputs.quantities must contain at least one quantity.")
    return tuple(str(quantity) for quantity in quantities_seq)


def _parse_base_config_path(base: Any, spec_path: Path) -> Path | None:
    if base is None:
        return None
    base_map = _as_mapping(base, "base")
    config_value = base_map.get("config")
    if config_value is None:
        return None
    if not isinstance(config_value, str) or not config_value.strip():
        raise ParametricGridYAMLError("base.config must be a non-empty string when provided.")

    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = spec_path.parent / config_path
    return config_path.resolve()


def _required_mapping(parent: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field not in parent:
        raise ParametricGridYAMLError(f"Missing required field: {field}.")
    return _as_mapping(parent[field], field)


def _required_sequence(parent: Mapping[str, Any], field: str) -> tuple[Any, ...]:
    key = field.rsplit(".", maxsplit=1)[-1]
    if key not in parent:
        raise ParametricGridYAMLError(f"Missing required field: {field}.")
    return _as_sequence(parent[key], field)


def _required_text(parent: Mapping[str, Any], field: str) -> str:
    key = field.rsplit(".", maxsplit=1)[-1]
    if key not in parent:
        raise ParametricGridYAMLError(f"Missing required field: {field}.")
    value = parent[key]
    if not isinstance(value, str) or not value.strip():
        raise ParametricGridYAMLError(f"{field} must be a non-empty string.")
    return value.strip()


def _as_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ParametricGridYAMLError(f"{field} must be a mapping.")
    return dict(value)


def _as_sequence(value: Any, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ParametricGridYAMLError(f"{field} must be a sequence.")
    return tuple(value)


__all__ = [
    "ParametricGridDefinition",
    "ParametricGridYAMLError",
    "load_parametric_grid_yaml",
    "parametric_grid_definition_to_preview",
]
