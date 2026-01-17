from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .laws import ForceDisplacementLaw, build_law_from_spec, compute_absorbed_energy
from pydantic import TypeAdapter

from .models import (
    EnPresetPartner,
    InterfaceLawSpec,
    InterfacePresetRef,
    PiecewiseLinearLaw,
    SimulationConfig,
    format_validation_error,
)
from .presets import (
    load_en15227_presets,
    resolve_interface_preset,
    resolve_partner_preset,
    resolve_scenario_preset,
)


class ConfigError(ValueError):
    pass


def _obj_to_dict(obj: Any) -> Any:
    """Convert common config objects to plain Python types.

    Supports dict, Pydantic models (v2 and v1), and dataclasses.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except Exception:
            pass
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def load_simulation_config(path: Path) -> Dict[str, Any]:
    raw = _load_raw_config(path)
    normalized = normalize_config_dict(raw, filename=path.name)
    return normalized


def _load_raw_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = yaml.safe_load(text)
    else:
        raise ConfigError(f"Unsupported config extension '{path.suffix}'.")
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: configuration must be a mapping")
    return data


def normalize_config_dict(config: Dict[str, Any], *, filename: str) -> Dict[str, Any]:
    raw = deepcopy(config)
    raw = migrate_config_dict(raw)
    try:
        SimulationConfig.model_validate(raw)
    except Exception as exc:
        if hasattr(exc, "errors"):
            raise ConfigError(format_validation_error(exc, filename=filename)) from exc
        raise
    return raw


def migrate_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    data = deepcopy(config)
    if "collision" not in data:
        import warnings
        warnings.warn(
            "Config is missing 'collision' section; auto-migrating legacy contact keys.",
            DeprecationWarning,
        )
        collision: Dict[str, Any] = {"partner": {"type": "rigid_wall"}}
        if "k_wall" in data:
            collision["interface"] = {
                "type": "linear",
                "k_N_per_m": data["k_wall"],
            }
        data["collision"] = collision
        legacy = deepcopy(config)
        legacy.pop("legacy", None)
        legacy.pop("collision", None)
        data["legacy"] = legacy
    return data


def resolve_collision(
    config: Dict[str, Any],
    *,
    config_path: Path,
) -> Dict[str, Any]:
    cfg = SimulationConfig.model_validate(config)
    collision = cfg.collision
    if collision is None:
        return {}

    base_dir = config_path.parent
    presets = load_en15227_presets(base_dir)
    partner = collision.partner
    interface_spec: Optional[InterfaceLawSpec] = collision.interface

    if isinstance(partner, EnPresetPartner):
        preset = resolve_partner_preset(presets, partner.preset_id)
        if not preset:
            raise ConfigError(f"{config_path.name}: unknown partner preset '{partner.preset_id}'")
        partner = preset
    if isinstance(partner, dict) and partner.get("type") == "en15227_preset":
        preset_id = partner.get("preset_id")
        preset = resolve_partner_preset(presets, preset_id)
        if not preset:
            raise ConfigError(f"{config_path.name}: unknown partner preset '{preset_id}'")
        partner = preset

    if collision.scenario:
        scenario = resolve_scenario_preset(presets, collision.scenario)
        if not scenario:
            raise ConfigError(f"{config_path.name}: unknown scenario preset '{collision.scenario}'")
        if "partner" in scenario:
            partner = scenario["partner"]
        if "interface" in scenario:
            interface_spec = scenario["interface"]

    interface_meta: Dict[str, Any] = {}
    law: Optional[ForceDisplacementLaw] = None
    if interface_spec is not None:
        if isinstance(interface_spec, InterfacePresetRef):
            preset = resolve_interface_preset(presets, interface_spec.preset_id)
            if not preset:
                raise ConfigError(
                    f"{config_path.name}: unknown interface preset '{interface_spec.preset_id}'"
                )
            interface_spec = preset
        if isinstance(interface_spec, dict):
            interface_spec = deepcopy(interface_spec)
            if interface_spec.get("type") == "en15227_preset":
                preset_id = interface_spec.get("preset_id")
                preset = resolve_interface_preset(presets, preset_id)
                if not preset:
                    raise ConfigError(f"{config_path.name}: unknown interface preset '{preset_id}'")
                interface_spec = preset
            metadata = interface_spec.pop("metadata", None) if isinstance(interface_spec, dict) else None
            if isinstance(metadata, dict):
                interface_meta.update(metadata)
            interface_spec = TypeAdapter(InterfaceLawSpec).validate_python(interface_spec)
        law = build_law_from_spec(interface_spec, config_dir=base_dir)
        interface_meta["absorbed_energy_J"] = law.absorbed_energy()

    if isinstance(interface_spec, PiecewiseLinearLaw):
        interface_meta["tabulated_energy_J"] = compute_absorbed_energy(interface_spec.points)

    for key, value in list(interface_meta.items()):
        if isinstance(value, str):
            try:
                interface_meta[key] = float(value)
            except ValueError:
                continue

    return {
        "partner": _obj_to_dict(partner),
        "interface": interface_spec.model_dump() if interface_spec is not None else None,
        "interface_law": law,
        "interface_meta": interface_meta,
    }


def apply_collision_to_params(
    config: Dict[str, Any],
    *,
    config_path: Path,
) -> Dict[str, Any]:
    params = deepcopy(config)
    collision = resolve_collision(params, config_path=config_path)
    if not collision:
        return params
    interface = collision.get("interface")
    interface_law = collision.get("interface_law")
    interface_meta = collision.get("interface_meta", {})
    if interface and interface.get("type") == "linear":
        params["k_wall"] = interface.get("k_N_per_m")
    if interface_law is not None and interface and interface.get("type") != "linear":
        params["contact_law"] = interface_law
        params.setdefault("contact_model", "tabulated")
    if interface_meta:
        params.setdefault("collision_meta", {})
        params["collision_meta"].update(interface_meta)
    if collision.get("partner") is not None:
        params.setdefault("collision_partner", collision["partner"])
    return params
