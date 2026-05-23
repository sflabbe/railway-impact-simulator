import pytest

from railway_simulator.studies.parametric_grid import (
    ParametricGridSpec,
    SweepDimension,
    apply_scenario_to_config,
    build_parametric_scenarios,
    preview_parametric_grid,
    run_parametric_grid_in_memory,
)


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
