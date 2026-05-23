from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from railway_simulator.cli import app
from railway_simulator.domain.project import Project
from railway_simulator.persistence.database import initialize_project_database
from railway_simulator.persistence.repositories import ProjectRepository, StudyRepository
from railway_simulator.studies.parametric_grid_persistence import run_parametric_grid_persistent


MINI_SPEC_PATH = Path("configs/studies/impact_parametric_mini.yml")


def _fake_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time_s": [0.0, 0.1],
            "Impact_Force_MN": [0.0, 1.5],
            "Penetration_mm": [0.0, 4.0],
            "Acceleration_g": [0.0, 0.6],
        }
    )


def _count_rows(db_path: Path, table: str) -> int:
    db = initialize_project_database(db_path)
    with db.connect() as con:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_run_grid_dry_run_with_db_does_not_write_or_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = initialize_project_database(tmp_path / "project.sqlite")
    project = Project(name="impact_workbench", root_dir=tmp_path)
    ProjectRepository(db).create(project)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("solver should not run during dry-run")

    monkeypatch.setattr("railway_simulator.services.simulation_service.run_simulation", fail_if_called)

    result = CliRunner().invoke(
        app,
        ["study", "run-grid", "--spec", str(MINI_SPEC_PATH), "--dry-run", "--db", str(db.path)],
    )

    assert result.exit_code == 0, result.output
    assert "dry-run, not written" in result.output
    assert StudyRepository(db).list_studies(project.id) == []
    assert _count_rows(db.path, "runs") == 0


def test_run_grid_persistent_creates_project_study_scenario_and_run(tmp_path: Path) -> None:
    db_path = tmp_path / "project.sqlite"

    result = run_parametric_grid_persistent(
        MINI_SPEC_PATH,
        db_path=db_path,
        limit=1,
        study_name="custom_grid",
        run_case_fn=lambda _config, _scenario: _fake_frame(),
    )

    projects = ProjectRepository(result.db).list()
    studies = StudyRepository(result.db).list_studies(result.project.id)
    scenarios = StudyRepository(result.db).list_scenarios(result.study.id)
    runs = StudyRepository(result.db).list_runs_for_study(result.study.id)

    assert len(projects) == 1
    assert projects[0].name == "impact_workbench"
    assert len(studies) == 1
    assert studies[0].name == "custom_grid"
    assert studies[0].study_type == "parametric_grid"
    assert len(scenarios) == 1
    assert scenarios[0].name == "impact_velocity_mps=-5.55556__contact_law=hooke"
    assert scenarios[0].meta["scenario_index"] == 0
    assert scenarios[0].meta["study_type"] == "parametric_grid"
    assert len(runs) == 1
    assert runs[0].status == "ok"
    assert runs[0].result_csv_path is not None
    assert runs[0].result_csv_path.is_file()
    assert len(result.rows) == 1
    assert result.rows[0]["peak_Impact_Force_MN"] == 1.5
    assert result.rows[0]["peak_Penetration_mm"] == 4.0
    assert result.rows[0]["peak_Acceleration_g"] == 0.6
    assert result.rows[0]["t_end"] == 0.1
    assert result.rows[0]["n_steps"] == 1
    assert result.rows[0]["warnings"] == ""


def test_run_grid_cli_persistent_limit_and_out_work_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "railway_simulator.services.simulation_service.run_simulation",
        lambda *_args, **_kwargs: _fake_frame(),
    )
    db_path = tmp_path / "project.sqlite"
    out = tmp_path / "summary.csv"

    result = CliRunner().invoke(
        app,
        [
            "study",
            "run-grid",
            "--spec",
            str(MINI_SPEC_PATH),
            "--db",
            str(db_path),
            "--limit",
            "1",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Run: impact_parametric_mini" in result.output
    assert "Study id:" in result.output
    assert out.is_file()
    summary = pd.read_csv(out)
    assert len(summary) == 1
    assert summary.loc[0, "status"] == "ok"

    db = initialize_project_database(db_path)
    project = ProjectRepository(db).list()[0]
    study = StudyRepository(db).list_studies(project.id)[0]
    assert len(StudyRepository(db).list_scenarios(study.id)) == 1
    assert len(StudyRepository(db).list_runs_for_study(study.id)) == 1


def test_run_grid_persistent_repeat_creates_new_study_without_breaking_db(tmp_path: Path) -> None:
    db_path = tmp_path / "project.sqlite"

    first = run_parametric_grid_persistent(
        MINI_SPEC_PATH,
        db_path=db_path,
        limit=1,
        run_case_fn=lambda _config, _scenario: _fake_frame(),
    )
    second = run_parametric_grid_persistent(
        MINI_SPEC_PATH,
        db_path=db_path,
        limit=1,
        run_case_fn=lambda _config, _scenario: _fake_frame(),
    )

    studies = StudyRepository(second.db).list_studies(second.project.id)
    assert first.project.id == second.project.id
    assert len(studies) == 2
    assert _count_rows(db_path, "runs") == 2


def test_run_grid_persistent_non_strict_records_failed_scenario(tmp_path: Path) -> None:
    def fail_case(_config, _scenario):
        raise RuntimeError("boom")

    result = run_parametric_grid_persistent(
        MINI_SPEC_PATH,
        db_path=tmp_path / "project.sqlite",
        limit=1,
        strict=False,
        run_case_fn=fail_case,
    )

    runs = StudyRepository(result.db).list_runs_for_study(result.study.id)
    assert len(result.rows) == 1
    assert result.rows[0]["status"] == "failed"
    assert "boom" in result.rows[0]["error"]
    assert runs[0].status == "failed"
    assert "boom" in (runs[0].error_message or "")


def test_run_grid_persistent_strict_raises_on_failed_scenario(tmp_path: Path) -> None:
    def fail_case(_config, _scenario):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_parametric_grid_persistent(
            MINI_SPEC_PATH,
            db_path=tmp_path / "project.sqlite",
            limit=1,
            strict=True,
            run_case_fn=fail_case,
        )


def test_run_grid_cli_strict_fails_on_service_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_simulation(*_args, **_kwargs):
        raise RuntimeError("service boom")

    monkeypatch.setattr("railway_simulator.services.simulation_service.run_simulation", fail_simulation)

    result = CliRunner().invoke(
        app,
        [
            "study",
            "run-grid",
            "--spec",
            str(MINI_SPEC_PATH),
            "--db",
            str(tmp_path / "project.sqlite"),
            "--limit",
            "1",
            "--strict",
        ],
    )

    assert result.exit_code != 0
    assert "service boom" in result.output

