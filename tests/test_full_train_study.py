from __future__ import annotations

from railway_simulator.domain.study import StudyDefinition
from railway_simulator.studies.full_train import FullTrainStudyRunner, FullTrainStudySpec


class _DummySimulationService:
    pass


def test_full_train_runner_builds_lok_and_zug_scenarios() -> None:
    base = {
        "n_masses": 3,
        "masses": [10_000.0, 20_000.0, 10_000.0],
        "x_init": [0.0, 5.0, 10.0],
        "y_init": [0.0, 0.0, 0.0],
        "fy": [1.0e6, 1.0e6],
        "uy": [0.1, 0.1],
        "k_train": [1.0e7, 1.0e7],
    }
    spec = FullTrainStudySpec(
        project_id="p1",
        base_config_id="cfg1",
        speeds_kmh=(10.0,),
        contact_models=("hooke",),
        mu_values=(0.3,),
    )
    study = StudyDefinition(project_id="p1", name="s", study_type="full_train", base_config_id="cfg1")
    runner = FullTrainStudyRunner(simulation_service=_DummySimulationService())  # type: ignore[arg-type]

    scenarios = runner.build_scenarios(study=study, base_config=base, spec=spec)

    assert len(scenarios) == 2
    lok = [s for s in scenarios if s.meta["mode"] == "lok_solo"][0]
    zug = [s for s in scenarios if s.meta["mode"] == "zug_full"][0]
    assert lok.params["n_masses"] == 3
    assert zug.params["n_masses"] == 15
    assert lok.params["v0_init"] == zug.params["v0_init"] == -10.0 / 3.6
    assert zug.meta["contact_patch_version"] == "contact_state_per_mass_v1"
