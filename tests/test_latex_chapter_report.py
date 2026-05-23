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
from railway_simulator.reporting import build_latex_chapter


def _make_report_fixture(tmp_path: Path):
    db = initialize_project_database(tmp_path / "project.sqlite")
    project = Project(name="impact_workbench", root_dir=tmp_path)
    ProjectRepository(db).create(project)
    snapshot = ConfigSnapshotRepository(db).create(project.id, {"n_masses": 3, "contact_model": "anagnostopoulos"})
    study = StudyDefinition(project_id=project.id, name="train_consist_comparison", study_type="full_train", base_config_id=snapshot.id)
    studies = StudyRepository(db)
    studies.create_study(study)

    periods = [10.0, 100.0, 1000.0]
    spectra = SpectrumRepository(db)
    for mode, values in [("lok_solo", [1.0, 2.0, 1.5]), ("zug_full", [1.1, 2.2, 1.8])]:
        scenario = Scenario(
            study_id=study.id,
            name=mode,
            params={"v0_init": -10.0},
            meta={
                "vehicle": "traxx_br187",
                "mode": mode,
                "speed_kmh": 80.0,
                "contact_model": "anagnostopoulos",
                "mu": 0.3,
            },
        )
        studies.add_scenario(scenario)
        result_csv = tmp_path / f"{mode}_results.csv"
        pd.DataFrame({"Time_s": [0.0, 0.01], "Impact_Force_MN": [0.0, max(values)]}).to_csv(result_csv, index=False)
        run = SimulationRun(scenario_id=scenario.id, status="ok", config_hash=f"hash-{mode}", result_csv_path=result_csv)
        studies.add_run(run)
        curve_csv = tmp_path / f"{mode}_srs.csv"
        pd.DataFrame(
            {
                "zeta": 0.05,
                "force_column": "Impact_Force_MN",
                "Tn_ms": periods,
                "Feq_MN": values,
            }
        ).to_csv(curve_csv, index=False)
        spectra.add_curve(SRSCurve(run_id=run.id, zeta=0.05, force_column="Impact_Force_MN", curve_csv_path=curve_csv))
    return db, study


def test_build_latex_chapter_bundle_from_persisted_study(tmp_path: Path) -> None:
    db, study = _make_report_fixture(tmp_path)
    result = build_latex_chapter(db=db, study_id=study.id, output_dir=tmp_path / "chapter", author="Tester")

    assert result.chapter_tex.is_file()
    assert result.main_tex.is_file()
    assert result.bibliography_bib.is_file()
    assert result.metadata_json.is_file()
    assert (result.output_dir / "tables" / "srs_curves_long.csv").is_file()
    assert (result.output_dir / "figures" / "srs_overlay.png").is_file()
    assert (result.output_dir / "figures" / "srs_full_train_vs_lok_ratio.png").is_file()

    tex = result.chapter_tex.read_text(encoding="utf-8")
    assert "Contact-state verification patch" in tex
    assert "F_{\\mathrm{EN}} = 6" in tex
    assert "\\bibliography{references}" in result.main_tex.read_text(encoding="utf-8")


def test_report_cli_build_chapter(tmp_path: Path) -> None:
    db, study = _make_report_fixture(tmp_path)
    out = tmp_path / "cli_chapter"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "report",
            "build-chapter",
            "--db",
            str(db.path),
            "--study-id",
            study.id,
            "--output-dir",
            str(out),
            "--author",
            "Tester",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Chapter:" in result.output
    assert (out / "chapter_impact_parametric_study.tex").is_file()
    assert (out / "references.bib").is_file()
