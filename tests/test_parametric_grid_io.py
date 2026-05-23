from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from railway_simulator.studies import get_by_path
from railway_simulator.studies.parametric_grid import (
    apply_scenario_to_config,
    build_parametric_scenarios,
)
from railway_simulator.studies.parametric_grid_io import (
    load_parametric_grid_yaml,
    parametric_grid_definition_to_preview,
)


MINI_SPEC_PATH = Path("configs/studies/impact_parametric_mini.yml")
FIXTURE_DIR = Path("build/test_parametric_grid_io")


def _valid_payload() -> dict:
    return {
        "study": {
            "name": "mini",
            "type": "parametric_grid",
        },
        "dimensions": [
            {
                "name": "contact_law",
                "kind": "parameter",
                "path": "contact_model",
                "values": ["hooke"],
            }
        ],
    }


def _write_yaml(name: str, payload: dict) -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_impact_parametric_mini_yaml_can_be_loaded() -> None:
    definition = load_parametric_grid_yaml(MINI_SPEC_PATH)

    assert definition.grid.name == "impact_parametric_mini"
    assert definition.raw["study"]["type"] == "parametric_grid"
    assert definition.srs == {"enabled": False}


def test_impact_parametric_mini_builds_two_by_two_grid_and_preview() -> None:
    definition = load_parametric_grid_yaml(MINI_SPEC_PATH)

    scenarios = build_parametric_scenarios(definition.grid)
    preview = parametric_grid_definition_to_preview(definition)

    assert len(scenarios) == 4
    assert len(preview) == 4


def test_impact_parametric_mini_preserves_base_config_and_quantities() -> None:
    definition = load_parametric_grid_yaml(MINI_SPEC_PATH)

    assert definition.raw["base"]["config"] == "../en15227/traxx_freight__EN15227_C1.yml"
    assert definition.base_config_path == (
        MINI_SPEC_PATH.parent / "../en15227/traxx_freight__EN15227_C1.yml"
    ).resolve()
    assert definition.grid.quantities == (
        "Impact_Force_MN",
        "Penetration_mm",
        "Acceleration_g",
    )


def test_impact_parametric_mini_paths_exist_in_base_and_apply_first_scenario() -> None:
    definition = load_parametric_grid_yaml(MINI_SPEC_PATH)
    assert definition.base_config_path is not None
    base_config = yaml.safe_load(definition.base_config_path.read_text(encoding="utf-8"))

    for dimension in definition.grid.dimensions:
        assert dimension.path is not None
        get_by_path(base_config, dimension.path)

    scenario = build_parametric_scenarios(definition.grid)[0]
    modified = apply_scenario_to_config(base_config, scenario)

    for dimension in definition.grid.dimensions:
        assert dimension.path is not None
        assert get_by_path(modified, dimension.path) == scenario.parameters[dimension.name]


def test_load_parametric_grid_yaml_raises_clear_error_for_missing_study_name() -> None:
    payload = _valid_payload()
    del payload["study"]["name"]

    with pytest.raises(ValueError, match="study.name"):
        load_parametric_grid_yaml(_write_yaml("missing_study_name.yml", payload))


def test_load_parametric_grid_yaml_raises_clear_error_for_missing_dimensions() -> None:
    payload = _valid_payload()
    del payload["dimensions"]

    with pytest.raises(ValueError, match="dimensions"):
        load_parametric_grid_yaml(_write_yaml("missing_dimensions.yml", payload))


def test_load_parametric_grid_yaml_raises_clear_error_for_wrong_study_type() -> None:
    payload = _valid_payload()
    payload["study"]["type"] = "full_train"

    with pytest.raises(ValueError, match="study.type.*parametric_grid"):
        load_parametric_grid_yaml(_write_yaml("wrong_study_type.yml", payload))


def test_load_parametric_grid_yaml_raises_clear_error_for_dimension_without_values() -> None:
    payload = _valid_payload()
    del payload["dimensions"][0]["values"]

    with pytest.raises(ValueError, match=r"dimensions\[0\]\.values"):
        load_parametric_grid_yaml(_write_yaml("dimension_without_values.yml", payload))
