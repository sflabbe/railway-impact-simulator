from __future__ import annotations

import pandas as pd

from railway_simulator.domain.result import SimulationRun
from railway_simulator.domain.scenario import Scenario
from railway_simulator.domain.spectrum import SRSCurve
from railway_simulator.domain.study import StudyDefinition
from railway_simulator.persistence.repositories import (
    ConfigSnapshotRepository,
    SpectrumRepository,
    StudyRepository,
)
from railway_simulator.services.project_service import ProjectService


def test_repository_joined_read_models_include_scenario_metadata(tmp_path) -> None:
    project, db = ProjectService().create_project(name="demo", root_dir=tmp_path / "demo")
    cfg = ConfigSnapshotRepository(db).create(project.id, {"n_masses": 1})
    study_repo = StudyRepository(db)
    spectrum_repo = SpectrumRepository(db)

    study = StudyDefinition(
        project_id=project.id,
        name="study",
        study_type="full_train",
        base_config_id=cfg.id,
    )
    study_repo.create_study(study)
    scenario = Scenario(
        study_id=study.id,
        name="lok",
        params={"n_masses": 1},
        meta={"mode": "lok_solo", "speed_kmh": 40.0, "contact_model": "flores", "mu": 0.3},
    )
    study_repo.add_scenario(scenario)

    run_csv = tmp_path / "run.csv"
    pd.DataFrame({"Time_s": [0.0, 0.1], "Impact_Force_MN": [0.0, 1.0]}).to_csv(run_csv, index=False)
    run = SimulationRun(scenario_id=scenario.id, status="ok", config_hash=scenario.config_hash, result_csv_path=run_csv)
    study_repo.add_run(run)

    curve_csv = tmp_path / "curve.csv"
    pd.DataFrame({"Tn_ms": [10.0], "Feq_MN": [1.0]}).to_csv(curve_csv, index=False)
    curve = SRSCurve(run_id=run.id, zeta=0.05, force_column="Impact_Force_MN", curve_csv_path=curve_csv)
    spectrum_repo.add_curve(curve)

    run_records = study_repo.list_run_records_for_study(study.id)
    curve_records = spectrum_repo.list_curve_records_for_study(study.id)

    assert run_records[0]["meta_mode"] == "lok_solo"
    assert run_records[0]["meta_speed_kmh"] == 40.0
    assert curve_records[0]["meta_contact_model"] == "flores"
    assert curve_records[0]["curve_csv_path"] == str(curve_csv)
    assert study_repo.get_study(study.id).name == "study"
