from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, "src")

import numpy as np
import yaml

from railway_simulator.config.loader import (
    ConfigError,
    apply_collision_to_params,
    migrate_config_dict,
    normalize_config_dict,
)
from railway_simulator.core.engine import run_simulation


def test_invalid_partner_validation() -> None:
    cfg = {
        "collision": {
            "partner": {"type": "reference_mass_1d"},
        }
    }
    try:
        normalize_config_dict(cfg, filename="bad.yml")
    except ConfigError as exc:
        assert "mass_kg" in str(exc)
    else:
        raise AssertionError("Expected ConfigError for missing mass_kg")


def test_en15227_c2_energy_computation(tmp_path: Path) -> None:
    cfg_path = tmp_path / "case.yml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "collision": {
                    "partner": {
                        "type": "en15227_preset",
                        "preset_id": "EN15227_2011_C2_regional_129t",
                    },
                    "interface": {
                        "type": "en15227_preset",
                        "preset_id": "EN15227_2011_C2_central_coupler",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    params = apply_collision_to_params(
        normalize_config_dict(yaml.safe_load(cfg_path.read_text()), filename=cfg_path.name),
        config_path=cfg_path,
    )
    energy = params["collision_meta"]["absorbed_energy_J"]
    expected = params["collision_meta"]["expected_energy_J"]
    assert energy > 0.0
    assert abs(energy - expected) / expected < 0.25


def test_migrated_config_matches_legacy_results() -> None:
    config_path = Path("configs/ice1_aluminum.yml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["T_max"] = 0.02
    raw["h_init"] = 1.0e-4
    raw["step"] = int(np.ceil(raw["T_max"] / raw["h_init"]))

    legacy_results = run_simulation(raw)

    migrated = migrate_config_dict(raw)
    migrated_params = apply_collision_to_params(migrated, config_path=config_path)
    migrated_results = run_simulation(migrated_params)

    legacy_force = legacy_results["Impact_Force_MN"].to_numpy()
    migrated_force = migrated_results["Impact_Force_MN"].to_numpy()

    np.testing.assert_allclose(legacy_force, migrated_force, rtol=1e-4, atol=1e-4)
