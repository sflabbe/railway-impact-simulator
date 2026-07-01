from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from railway_simulator.domain.project import Project
from railway_simulator.domain.result import SimulationRun
from railway_simulator.domain.scenario import Scenario
from railway_simulator.domain.spectrum import SRSCurve
from railway_simulator.domain.study import StudyDefinition
from railway_simulator.persistence.database import initialize_project_database
from railway_simulator.persistence.repositories import ConfigSnapshotRepository, ProjectRepository, SpectrumRepository, StudyRepository
from railway_simulator.ui.project_workbench import (
    DEFAULT_PARAMETRIC_DB_PATH,
    build_full_vs_lok_ratio_figure,
    build_parametric_peak_force_figure,
    load_saved_parametric_grid_summary,
    build_srs_envelope_figure,
    build_srs_overlay_figure,
    build_workbench_chapter_bundle,
    curve_label,
    default_chapter_output_dir,
    load_parametric_grid_ui_spec,
    make_log_period_grid_ms,
    parametric_grid_peak_force_chart_data,
    parametric_grid_peak_force_csv_bytes,
    parametric_grid_peak_force_html_bytes,
    parametric_grid_summary_csv_bytes,
    parametric_grid_warning_rows,
    parse_float_csv,
    preview_parametric_grid_ui_scenarios,
    run_parametric_grid_ui_study,
    sanitize_path_component,
    selected_curve_records,
    zip_report_bundle_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_PARAMETRIC_SPEC = REPO_ROOT / "configs" / "studies" / "impact_parametric_mini.yml"


def test_parse_float_csv_and_period_grid() -> None:
    assert parse_float_csv("10, 20; 30") == (10.0, 20.0, 30.0)
    grid = make_log_period_grid_ms(10.0, 1000.0, 5)
    assert len(grid) == 5
    assert grid[0] == 10.0
    assert round(grid[-1], 10) == 1000.0


def test_parametric_grid_ui_loads_yaml_and_previews_without_db_write(tmp_path: Path) -> None:
    db_path = tmp_path / "project.sqlite"

    definition = load_parametric_grid_ui_spec(MINI_PARAMETRIC_SPEC)
    preview = preview_parametric_grid_ui_scenarios(MINI_PARAMETRIC_SPEC)

    assert definition.grid.name == "impact_parametric_mini"
    assert len(preview) == 4
    assert preview["scenario_index"].tolist() == [0, 1, 2, 3]
    assert db_path.exists() is False


def test_parametric_grid_ui_run_limit_one_persists_summary_and_warnings(tmp_path: Path) -> None:
    db_path = tmp_path / "project.sqlite"

    def fake_case(_config, _scenario):
        frame = pd.DataFrame(
            {
                "Time_s": [0.0, 0.2],
                "Impact_Force_MN": [0.0, 2.5],
                "Penetration_mm": [0.0, 7.0],
                "Acceleration_g": [0.0, 0.9],
            }
        )
        return {"df": frame, "warnings": ["solver warning from fake runner"]}

    result = run_parametric_grid_ui_study(
        MINI_PARAMETRIC_SPEC,
        db_path=db_path,
        project_name="impact_workbench",
        study_name="",
        limit=1,
        persist_results=True,
        run_case_fn=fake_case,
    )

    assert result.persistent_result is not None
    assert len(result.summary) == 1
    assert "warnings" in result.summary.columns
    assert result.summary.loc[0, "peak_Impact_Force_MN"] == 2.5
    assert result.summary.loc[0, "peak_Penetration_mm"] == 7.0
    assert result.summary.loc[0, "peak_Acceleration_g"] == 0.9
    assert result.summary.loc[0, "t_end"] == 0.2
    assert result.summary.loc[0, "n_steps"] == 1
    assert "solver warning" in result.summary.loc[0, "warnings"]

    studies = StudyRepository(result.persistent_result.db).list_studies(result.persistent_result.project.id)
    scenarios = StudyRepository(result.persistent_result.db).list_scenarios(result.persistent_result.study.id)
    runs = StudyRepository(result.persistent_result.db).list_runs_for_study(result.persistent_result.study.id)
    assert len(studies) == 1
    assert studies[0].study_type == "parametric_grid"
    assert studies[0].name == "impact_parametric_mini"
    assert len(scenarios) == 1
    assert len(runs) == 1
    assert "solver warning" in scenarios[0].meta["solver_warnings"]

    saved_summary = load_saved_parametric_grid_summary(result.persistent_result.db, result.persistent_result.study.id)
    assert len(saved_summary) == 1
    assert saved_summary.loc[0, "peak_Impact_Force_MN"] == 2.5
    assert "solver warning" in saved_summary.loc[0, "warnings"]


def test_parametric_grid_ui_warning_chart_and_csv_helpers() -> None:
    summary = pd.DataFrame(
        [
            {
                "scenario_index": 0,
                "scenario_label": "case_a",
                "status": "ok",
                "impact_velocity_mps": -5.55,
                "peak_Impact_Force_MN": 3.0,
                "warnings": "watch this",
                "error": "",
            },
            {
                "scenario_index": 1,
                "scenario_label": "case_b",
                "status": "ok",
                "impact_velocity_mps": -11.11,
                "peak_Impact_Force_MN": None,
                "warnings": "",
                "error": "",
            },
        ]
    )

    warnings = parametric_grid_warning_rows(summary)
    chart_data = parametric_grid_peak_force_chart_data(summary)
    figure = build_parametric_peak_force_figure(summary)
    csv_bytes = parametric_grid_summary_csv_bytes(summary)
    chart_csv_bytes = parametric_grid_peak_force_csv_bytes(summary)
    chart_html_bytes = parametric_grid_peak_force_html_bytes(summary)
    roundtrip = pd.read_csv(io.BytesIO(csv_bytes))
    chart_roundtrip = pd.read_csv(io.BytesIO(chart_csv_bytes))

    assert warnings["scenario_label"].tolist() == ["case_a"]
    assert chart_data["scenario_label"].tolist() == ["case_a"]
    assert figure is not None
    assert len(figure.data) == 1
    assert roundtrip["scenario_label"].tolist() == ["case_a", "case_b"]
    assert chart_roundtrip["scenario_label"].tolist() == ["case_a"]
    assert b"Peak impact force by scenario" in chart_html_bytes


def test_parametric_grid_ui_helpers_do_not_modify_default_project_db(tmp_path: Path) -> None:
    default_db = DEFAULT_PARAMETRIC_DB_PATH
    before = (
        default_db.exists(),
        default_db.stat().st_size if default_db.exists() else None,
        default_db.stat().st_mtime_ns if default_db.exists() else None,
    )

    preview_parametric_grid_ui_scenarios(MINI_PARAMETRIC_SPEC)
    run_parametric_grid_ui_study(
        MINI_PARAMETRIC_SPEC,
        db_path=tmp_path / "project.sqlite",
        limit=1,
        persist_results=True,
        run_case_fn=lambda _config, _scenario: pd.DataFrame({"Time_s": [0.0], "Impact_Force_MN": [0.0]}),
    )

    after = (
        default_db.exists(),
        default_db.stat().st_size if default_db.exists() else None,
        default_db.stat().st_mtime_ns if default_db.exists() else None,
    )
    assert after == before


def test_curve_filter_and_label() -> None:
    records = [
        {"zeta": 0.05, "meta_mode": "lok_solo", "meta_contact_model": "flores", "meta_speed_kmh": 40.0, "meta_mu": 0.3},
        {"zeta": 0.10, "meta_mode": "zug_full", "meta_contact_model": "flores", "meta_speed_kmh": 40.0, "meta_mu": 0.3},
    ]
    assert len(selected_curve_records(records, zeta=0.05, modes={"lok_solo"})) == 1
    assert "40 km/h" in curve_label(records[0])
    assert "flores" in curve_label(records[0])


def test_srs_figures_from_persisted_curve_records(tmp_path) -> None:
    periods = np.array([10.0, 100.0, 1000.0])
    lok = pd.DataFrame({"zeta": 0.05, "force_column": "Impact_Force_MN", "Tn_ms": periods, "Feq_MN": [1.0, 2.0, 1.5]})
    zug = pd.DataFrame({"zeta": 0.05, "force_column": "Impact_Force_MN", "Tn_ms": periods, "Feq_MN": [2.0, 4.0, 3.0]})
    lok_path = tmp_path / "lok.csv"
    zug_path = tmp_path / "zug.csv"
    lok.to_csv(lok_path, index=False)
    zug.to_csv(zug_path, index=False)
    records = [
        {
            "curve_id": "srs_lok",
            "run_id": "run_lok",
            "curve_csv_path": str(lok_path),
            "zeta": 0.05,
            "force_column": "Impact_Force_MN",
            "scenario_name": "lok",
            "meta_mode": "lok_solo",
            "meta_speed_kmh": 40.0,
            "meta_contact_model": "anagnostopoulos",
            "meta_mu": 0.3,
        },
        {
            "curve_id": "srs_zug",
            "run_id": "run_zug",
            "curve_csv_path": str(zug_path),
            "zeta": 0.05,
            "force_column": "Impact_Force_MN",
            "scenario_name": "zug",
            "meta_mode": "zug_full",
            "meta_speed_kmh": 40.0,
            "meta_contact_model": "anagnostopoulos",
            "meta_mu": 0.3,
        },
    ]

    overlay = build_srs_overlay_figure(records)
    envelope = build_srs_envelope_figure(records)
    ratio = build_full_vs_lok_ratio_figure(records)

    assert len(overlay.data) == 2
    assert len(envelope.data) == 1
    assert len(ratio.data) == 1
    assert np.allclose(ratio.data[0].y, [2.0, 2.0, 2.0])



def _make_minimal_report_fixture(tmp_path: Path):
    db = initialize_project_database(tmp_path / "project.sqlite")
    project = Project(name="impact_workbench", root_dir=tmp_path)
    ProjectRepository(db).create(project)
    snapshot = ConfigSnapshotRepository(db).create(project.id, {"n_masses": 3, "contact_model": "anagnostopoulos"})
    study = StudyDefinition(project_id=project.id, name="Train Consist Comparison", study_type="full_train", base_config_id=snapshot.id)
    studies = StudyRepository(db)
    studies.create_study(study)

    spectra = SpectrumRepository(db)
    for mode, values in [("lok_solo", [1.0, 2.0, 1.5]), ("zug_full", [1.1, 2.4, 1.8])]:
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
                "Tn_ms": [10.0, 100.0, 1000.0],
                "Feq_MN": values,
            }
        ).to_csv(curve_csv, index=False)
        spectra.add_curve(SRSCurve(run_id=run.id, zeta=0.05, force_column="Impact_Force_MN", curve_csv_path=curve_csv))
    return db, project, study


def test_report_bundle_helpers_build_and_zip(tmp_path: Path) -> None:
    db, project, study = _make_minimal_report_fixture(tmp_path)

    assert sanitize_path_component("Train Consist Comparison / v0=80") == "train_consist_comparison_v0_80"
    default_dir = default_chapter_output_dir(project.normalized_root(), study_name=study.name, study_id=study.id)
    assert default_dir.parent.name == "reports"
    assert default_dir.name.startswith("chapter_train_consist_comparison_")

    result = build_workbench_chapter_bundle(
        db=db,
        study_id=study.id,
        output_dir=tmp_path / "chapter",
        title="Demo Chapter",
        author="Tester",
        zeta=0.05,
    )
    assert result.chapter_tex.is_file()
    assert result.main_tex.is_file()
    archive_bytes = zip_report_bundle_bytes(result.output_dir)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
    assert "main.tex" in names
    assert "chapter_impact_parametric_study.tex" in names
    assert "references.bib" in names
