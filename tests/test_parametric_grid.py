import contextlib
import io
from pathlib import Path

import pytest

from railway_simulator.core.engine import run_simulation
from railway_simulator.studies import normalize_contact_law_after_contact_model_override
from railway_simulator.studies.parametric_grid import (
    ParametricGridSpec,
    SweepDimension,
    apply_scenario_to_config,
    build_parametric_scenarios,
    preview_parametric_grid,
    run_parametric_grid_in_memory,
)
from railway_simulator.studies.parametric_grid_cli import load_base_config_for_grid


REPO_ROOT = Path(__file__).resolve().parents[1]


def _two_by_two_spec() -> ParametricGridSpec:
    return ParametricGridSpec(
        name="demo_grid",
        dimensions=(
            SweepDimension(
                name="speed_kmh",
                kind="numeric",
                path="speed.kmh",
                values=(20, 40),
            ),
            SweepDimension(
                name="contact_model",
                kind="categorical",
                path="contact.model",
                values=("hooke", "hertz"),
            ),
        ),
    )


def test_build_parametric_scenarios_builds_two_by_two_grid() -> None:
    scenarios = build_parametric_scenarios(_two_by_two_spec())

    assert len(scenarios) == 4
    assert [scenario.index for scenario in scenarios] == [0, 1, 2, 3]


def test_build_parametric_scenarios_preserves_stable_order() -> None:
    scenarios = build_parametric_scenarios(_two_by_two_spec())

    assert [scenario.parameters for scenario in scenarios] == [
        {"speed_kmh": 20, "contact_model": "hooke"},
        {"speed_kmh": 20, "contact_model": "hertz"},
        {"speed_kmh": 40, "contact_model": "hooke"},
        {"speed_kmh": 40, "contact_model": "hertz"},
    ]


def test_build_parametric_scenarios_uses_deterministic_labels() -> None:
    spec = _two_by_two_spec()
    labels_a = [scenario.label for scenario in build_parametric_scenarios(spec)]
    labels_b = [scenario.label for scenario in build_parametric_scenarios(spec)]

    assert labels_a == labels_b
    assert labels_a == [
        "speed_kmh=20__contact_model=hooke",
        "speed_kmh=20__contact_model=hertz",
        "speed_kmh=40__contact_model=hooke",
        "speed_kmh=40__contact_model=hertz",
    ]


def test_build_parametric_scenarios_stores_flat_metadata() -> None:
    scenario = build_parametric_scenarios(_two_by_two_spec())[0]

    assert scenario.metadata == {"speed_kmh": 20, "contact_model": "hooke"}
    assert all(not isinstance(value, dict) for value in scenario.metadata.values())


def test_preview_parametric_grid_does_not_execute_cases() -> None:
    executed = False

    def run_case_fn(_config, _scenario):
        nonlocal executed
        executed = True

    preview = preview_parametric_grid(_two_by_two_spec())

    assert len(preview) == 4
    assert preview[0] == {
        "index": 0,
        "label": "speed_kmh=20__contact_model=hooke",
        "speed_kmh": 20,
        "contact_model": "hooke",
    }
    assert executed is False
    assert callable(run_case_fn)


def test_apply_scenario_to_config_does_not_mutate_base_config() -> None:
    base = {"speed": {"kmh": 0}, "contact": {"model": "base"}}
    scenario = build_parametric_scenarios(_two_by_two_spec())[0]

    modified = apply_scenario_to_config(base, scenario)

    assert base == {"speed": {"kmh": 0}, "contact": {"model": "base"}}
    assert modified == {"speed": {"kmh": 20}, "contact": {"model": "hooke"}}
    assert modified is not base


def test_apply_scenario_to_config_applies_simple_path() -> None:
    spec = ParametricGridSpec(
        name="simple",
        dimensions=(
            SweepDimension(
                name="speed_kmh",
                kind="numeric",
                path="speed_kmh",
                values=(20,),
            ),
        ),
    )
    scenario = build_parametric_scenarios(spec)[0]

    modified = apply_scenario_to_config({"speed_kmh": 0}, scenario)

    assert modified["speed_kmh"] == 20


def test_apply_scenario_to_config_raises_clear_error_for_missing_path() -> None:
    spec = ParametricGridSpec(
        name="missing_path",
        dimensions=(
            SweepDimension(
                name="speed_kmh",
                kind="numeric",
                path="missing.speed_kmh",
                values=(20,),
            ),
        ),
    )
    scenario = build_parametric_scenarios(spec)[0]

    with pytest.raises(ValueError, match="speed_kmh.*missing.speed_kmh"):
        apply_scenario_to_config({"speed_kmh": 0}, scenario)


def test_run_parametric_grid_in_memory_respects_limit() -> None:
    calls = []

    def run_case_fn(config, scenario):
        calls.append(scenario.index)
        return {"label": scenario.label, "speed": config["speed"]["kmh"]}

    results = run_parametric_grid_in_memory(
        _two_by_two_spec(),
        {"speed": {"kmh": 0}, "contact": {"model": "base"}},
        run_case_fn,
        limit=1,
    )

    assert calls == [0]
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert results[0]["result"] == {
        "label": "speed_kmh=20__contact_model=hooke",
        "speed": 20,
    }


def test_run_parametric_grid_in_memory_continues_after_error_when_not_strict() -> None:
    spec = ParametricGridSpec(
        name="error_grid",
        dimensions=(
            SweepDimension(
                name="speed_kmh",
                kind="numeric",
                path="speed_kmh",
                values=(20, 40),
            ),
        ),
    )

    def run_case_fn(_config, scenario):
        if scenario.index == 0:
            raise RuntimeError("boom")
        return scenario.label

    results = run_parametric_grid_in_memory(spec, {"speed_kmh": 0}, run_case_fn)

    assert [result["status"] for result in results] == ["failed", "ok"]
    assert results[0]["error_type"] == "RuntimeError"
    assert "boom" in results[0]["error"]
    assert results[1]["result"] == "speed_kmh=40"


def test_run_parametric_grid_in_memory_raises_first_error_when_strict() -> None:
    spec = ParametricGridSpec(
        name="strict_grid",
        dimensions=(
            SweepDimension(
                name="speed_kmh",
                kind="numeric",
                path="speed_kmh",
                values=(20, 40),
            ),
        ),
    )
    calls = []

    def run_case_fn(_config, scenario):
        calls.append(scenario.index)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_parametric_grid_in_memory(
            spec,
            {"speed_kmh": 0},
            run_case_fn,
            strict=True,
        )

    assert calls == [0]


def test_build_parametric_scenarios_rejects_empty_dimension_values() -> None:
    spec = ParametricGridSpec(
        name="empty",
        dimensions=(
            SweepDimension(
                name="speed_kmh",
                kind="numeric",
                path="speed_kmh",
                values=(),
            ),
        ),
    )

    with pytest.raises(ValueError, match="speed_kmh.*no values"):
        build_parametric_scenarios(spec)


def test_apply_scenario_to_config_ignores_pathless_metadata_dimension() -> None:
    spec = ParametricGridSpec(
        name="metadata_only",
        dimensions=(
            SweepDimension(
                name="case_group",
                kind="metadata",
                path=None,
                values=("baseline",),
            ),
            SweepDimension(
                name="speed_kmh",
                kind="numeric",
                path="speed_kmh",
                values=(20,),
            ),
        ),
    )
    scenario = build_parametric_scenarios(spec)[0]

    modified = apply_scenario_to_config({"speed_kmh": 0}, scenario)

    assert modified == {"speed_kmh": 20}
    assert scenario.metadata == {"case_group": "baseline", "speed_kmh": 20}


def _contact_model_override_scenario(value: str):
    spec = ParametricGridSpec(
        name="contact_override",
        dimensions=(
            SweepDimension(
                name="contact_model",
                kind="parameter",
                path="contact_model",
                values=(value,),
            ),
        ),
    )
    return build_parametric_scenarios(spec)[0]


def test_contact_law_normalization_removes_law_without_mutating_input() -> None:
    base = {
        "contact_model": "hooke",
        "contact_law": {"points": [[0.0, 0.0], [0.1, 1.0]]},
        "collision_meta": {"preset": "en15227"},
    }

    modified = normalize_contact_law_after_contact_model_override(
        base,
        contact_model_overridden=True,
    )

    assert modified["contact_law"] is None
    assert modified["collision_meta"]["contact_law_removed_due_to_contact_model_override"] is True
    assert base == {
        "contact_model": "hooke",
        "contact_law": {"points": [[0.0, 0.0], [0.1, 1.0]]},
        "collision_meta": {"preset": "en15227"},
    }


def test_contact_law_normalization_preserves_tabulated_law_without_mutating_input() -> None:
    base = {
        "contact_model": "tabulated",
        "contact_law": {"points": [[0.0, 0.0], [0.1, 1.0]]},
        "collision_meta": {"preset": "en15227"},
    }

    modified = normalize_contact_law_after_contact_model_override(
        base,
        contact_model_overridden=True,
    )

    assert modified["contact_law"] == {"points": [[0.0, 0.0], [0.1, 1.0]]}
    assert "contact_law_removed_due_to_contact_model_override" not in modified["collision_meta"]
    assert base == {
        "contact_model": "tabulated",
        "contact_law": {"points": [[0.0, 0.0], [0.1, 1.0]]},
        "collision_meta": {"preset": "en15227"},
    }


def test_contact_law_normalization_isolates_output_metadata() -> None:
    base = {
        "contact_model": "tabulated",
        "contact_law": {"points": [[0.0, 0.0], [0.1, 1.0]]},
        "collision_meta": {"preset": "en15227"},
    }

    modified = normalize_contact_law_after_contact_model_override(
        base,
        contact_model_overridden=True,
    )
    modified["collision_meta"]["new_key"] = "output-only"

    assert base["collision_meta"] == {"preset": "en15227"}


def test_contact_model_override_removes_inherited_tabulated_contact_law() -> None:
    base = load_base_config_for_grid(REPO_ROOT / "configs" / "mi_en15227_c1.yml")
    assert base["contact_model"] == "tabulated"
    assert base["contact_law"] is not None

    modified = apply_scenario_to_config(base, _contact_model_override_scenario("hooke"))

    assert modified["contact_model"] == "hooke"
    assert modified["contact_law"] is None
    assert modified["collision_meta"]["contact_law_removed_due_to_contact_model_override"] is True

    modified.update({"T_max": 1.0e-4, "h_init": 1.0e-4, "step": 1, "T_int": (0.0, 1.0e-4)})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        run_simulation(modified, emit_peak_diagnostics=False)


def test_tabulated_contact_model_override_preserves_inherited_contact_law() -> None:
    base = load_base_config_for_grid(REPO_ROOT / "configs" / "mi_en15227_c1.yml")

    modified = apply_scenario_to_config(base, _contact_model_override_scenario("tabulated"))

    assert modified["contact_model"] == "tabulated"
    assert modified["contact_law"] is not None
    assert "contact_law_removed_due_to_contact_model_override" not in modified.get("collision_meta", {})
