from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class PresetBundle:
    partners: Dict[str, Dict[str, Any]]
    interfaces: Dict[str, Dict[str, Any]]
    scenarios: Dict[str, Dict[str, Any]]


def load_en15227_presets(base_dir: Path) -> PresetBundle:
    data_dir = base_dir / "data" / "en15227"
    if not data_dir.is_dir():
        repo_root = Path(__file__).resolve().parents[3]
        data_dir = repo_root / "data" / "en15227"
    partners = _load_yaml(data_dir / "partners.yaml")
    interfaces = _load_yaml(data_dir / "interfaces.yaml")
    scenarios = _load_yaml(data_dir / "scenarios.yaml")
    return PresetBundle(
        partners=partners.get("partners", {}),
        interfaces=interfaces.get("interfaces", {}),
        scenarios=scenarios.get("scenarios", {}),
    )


def resolve_partner_preset(presets: PresetBundle, preset_id: str) -> Optional[Dict[str, Any]]:
    return presets.partners.get(preset_id)


def resolve_interface_preset(presets: PresetBundle, preset_id: str) -> Optional[Dict[str, Any]]:
    return presets.interfaces.get(preset_id)


def resolve_scenario_preset(presets: PresetBundle, preset_id: str) -> Optional[Dict[str, Any]]:
    return presets.scenarios.get(preset_id)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
