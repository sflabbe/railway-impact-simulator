from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from railway_simulator.cli import app
from railway_simulator.domain.project import Project
from railway_simulator.domain.result import SimulationRun
from railway_simulator.domain.scenario import Scenario
from railway_simulator.domain.spectrum import SRSCurve
from railway_simulator.domain.study import StudyDefinition
from railway_simulator.persistence.database import initialize_project_database
from railway_simulator.persistence.repositories import ConfigSnapshotRepository, ProjectRepository, SpectrumRepository, StudyRepository


def _mini_config(path: Path) -> Path:
    path.write_text(
        """
case_name: mini
n_masses: 3
masses: [10000, 20000, 10000]
x_init: [0, 5, 10]
y_init: [0, 0, 0]
fy: [1000000, 1000000]
uy: [0.1, 0.1]
k_train: [10000000, 10000000]
""".strip(),
        encoding="utf-8",
    )
    return path


def test_project_cli_create_list_and_full_train_dry_run(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "impact_workbench"

    result = runner.invoke(app, ["project", "create", "--name", "impact_workbench", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert (root / "project.sqlite").is_file()
    assert (root / "runs").is_dir()
    assert "Project id:" in result.output

    result = runner.invoke(app, ["project", "list", "--db", str(root / "project.sqlite")])
    assert result.exit_code == 0, result.output
    assert "impact_workbench" in result.output

    cfg = _mini_config(tmp_path / "mini.yml")
    result = runner.invoke(
        app,
        [
            "study",
            "run-full-train",
            "--db",
            str(root / "project.sqlite"),
            "--base-config",
            str(cfg),
            "--speeds",
            "10,20",
            "--contact-models",
            "anagnostopoulos",
            "--mu-values",
            "0.3",
            "--zeta-values",
            "0.05",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run: 4 scenarios" in result.output
    assert "contact_state_per_mass_v1" in result.output


def test_srs_cli_export_and_compare_from_persisted_curves(tmp_path: Path) -> None:
    runner = CliRunner()
    db = initialize_project_database(tmp_path / "project.sqlite")
    project = Project(name="impact_workbench", root_dir=tmp_path)
    ProjectRepository(db).create(project)
    snapshot = ConfigSnapshotRepository(db).create(project.id, {"n_masses": 3})
    study = StudyDefinition(project_id=project.id, name="s", study_type="full_train", base_config_id=snapshot.id)
    studies = StudyRepository(db)
    studies.create_study(study)

    lok = Scenario(study_id=study.id, name="lok", params={"v0_init": -10.0}, meta={"mode": "lok_solo", "speed_kmh": 10.0, "contact_model": "anagnostopoulos", "mu": 0.3})
    zug = Scenario(study_id=study.id, name="zug", params={"v0_init": -10.0}, meta={"mode": "zug_full", "speed_kmh": 10.0, "contact_model": "anagnostopoulos", "mu": 0.3})
    studies.add_scenario(lok)
    studies.add_scenario(zug)
    lok_run = SimulationRun(scenario_id=lok.id, status="ok", config_hash="h1", result_csv_path=tmp_path / "lok_results.csv")
    zug_run = SimulationRun(scenario_id=zug.id, status="ok", config_hash="h2", result_csv_path=tmp_path / "zug_results.csv")
    studies.add_run(lok_run)
    studies.add_run(zug_run)

    periods = [10.0, 100.0, 1000.0]
    lok_curve = tmp_path / "lok_srs.csv"
    zug_curve = tmp_path / "zug_srs.csv"
    pd.DataFrame({"zeta": 0.05, "force_column": "Impact_Force_MN", "Tn_ms": periods, "Feq_MN": [1.0, 2.0, 1.5]}).to_csv(lok_curve, index=False)
    pd.DataFrame({"zeta": 0.05, "force_column": "Impact_Force_MN", "Tn_ms": periods, "Feq_MN": [2.0, 4.0, 3.0]}).to_csv(zug_curve, index=False)
    spectra = SpectrumRepository(db)
    spectra.add_curve(SRSCurve(run_id=lok_run.id, zeta=0.05, force_column="Impact_Force_MN", curve_csv_path=lok_curve))
    spectra.add_curve(SRSCurve(run_id=zug_run.id, zeta=0.05, force_column="Impact_Force_MN", curve_csv_path=zug_curve))

    exported = tmp_path / "export.csv"
    result = runner.invoke(app, ["srs", "export", "--db", str(db.path), "--study-id", study.id, "--output", str(exported), "--zeta", "0.05"])
    assert result.exit_code == 0, result.output
    assert exported.is_file()
    assert set(pd.read_csv(exported)["meta_mode"]) == {"lok_solo", "zug_full"}

    out = tmp_path / "compare"
    result = runner.invoke(app, ["srs", "compare", "--db", str(db.path), "--study-id", study.id, "--output-dir", str(out), "--zeta", "0.05"])
    assert result.exit_code == 0, result.output
    ratio = pd.read_csv(out / "srs_full_train_vs_lok_ratio.csv")
    assert (out / "srs_overlay.png").is_file()
    assert (out / "srs_envelope.csv").is_file()
    assert ratio["Feq_MN_ratio"].tolist() == [2.0, 2.0, 2.0]
