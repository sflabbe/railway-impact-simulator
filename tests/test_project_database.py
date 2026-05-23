from __future__ import annotations

from pathlib import Path

from railway_simulator.domain.project import Project
from railway_simulator.domain.scenario import Scenario
from railway_simulator.domain.study import StudyDefinition
from railway_simulator.persistence.database import initialize_project_database
from railway_simulator.persistence.repositories import (
    ConfigSnapshotRepository,
    ProjectRepository,
    StudyRepository,
)


def test_project_database_roundtrip(tmp_path: Path) -> None:
    db = initialize_project_database(tmp_path / "project.sqlite")
    projects = ProjectRepository(db)
    configs = ConfigSnapshotRepository(db)
    studies = StudyRepository(db)

    project = Project(name="impact_workbench", root_dir=tmp_path, description="test project")
    projects.create(project)

    loaded = projects.get(project.id)
    assert loaded.name == "impact_workbench"
    assert loaded.root_dir == tmp_path.resolve()

    snapshot = configs.create(project.id, {"v0_init": -10.0, "contact_model": "hooke"})
    assert configs.get(snapshot.id).config_hash == snapshot.config_hash

    study = StudyDefinition(
        project_id=project.id,
        name="mini",
        study_type="full_train",
        base_config_id=snapshot.id,
        parameters={"speeds_kmh": [10.0]},
    )
    studies.create_study(study)
    scenario = Scenario(study_id=study.id, name="one", params={"v0_init": -10.0})
    studies.add_scenario(scenario)

    assert studies.list_studies(project.id)[0].name == "mini"
    assert studies.list_scenarios(study.id)[0].name == "one"
