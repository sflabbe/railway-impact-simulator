from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from railway_simulator.cli import app
from railway_simulator.studies import parametric_grid_cli
from railway_simulator.studies.parametric_grid_cli import run_grid_from_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_SPEC_PATH = REPO_ROOT / "configs" / "studies" / "impact_parametric_mini.yml"


def test_run_grid_dry_run_cli_finishes_ok() -> None:
    result = CliRunner().invoke(
        app,
        ["study", "run-grid", "--spec", str(MINI_SPEC_PATH), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run: impact_parametric_mini" in result.output


def test_run_grid_dry_run_cli_shows_four_scenarios() -> None:
    result = CliRunner().invoke(
        app,
        ["study", "run-grid", "--spec", str(MINI_SPEC_PATH), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "Scenarios: 4" in result.output
    assert result.output.count("impact_velocity_mps=") == 4


def test_run_grid_dry_run_limit_exports_one_scenario(tmp_path: Path) -> None:
    out = tmp_path / "preview.csv"
    result = CliRunner().invoke(
        app,
        [
            "study",
            "run-grid",
            "--spec",
            str(MINI_SPEC_PATH),
            "--dry-run",
            "--limit",
            "1",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Scenarios: 1" in result.output
    preview = pd.read_csv(out)
    assert len(preview) == 1
    assert list(preview.columns[:2]) == ["scenario_index", "scenario_label"]


def test_run_grid_dry_run_out_csv_has_one_row_per_scenario(tmp_path: Path) -> None:
    out = tmp_path / "preview.csv"
    result = CliRunner().invoke(
        app,
        [
            "study",
            "run-grid",
            "--spec",
            str(MINI_SPEC_PATH),
            "--dry-run",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert len(pd.read_csv(out)) == 4


def test_run_grid_dry_run_does_not_execute_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("solver should not run during dry-run")

    monkeypatch.setattr(parametric_grid_cli, "run_simulation", fail_if_called)

    result = CliRunner().invoke(
        app,
        ["study", "run-grid", "--spec", str(MINI_SPEC_PATH), "--dry-run"],
    )

    assert result.exit_code == 0, result.output


def test_run_grid_missing_spec_reports_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yml"
    result = CliRunner().invoke(
        app,
        ["study", "run-grid", "--spec", str(missing), "--dry-run"],
    )

    assert result.exit_code != 0
    assert "YAML file not found" in result.output


def test_run_grid_invalid_yaml_reports_clear_error(tmp_path: Path) -> None:
    spec = tmp_path / "invalid.yml"
    spec.write_text("study: [", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["study", "run-grid", "--spec", str(spec), "--dry-run"],
    )

    assert result.exit_code != 0
    assert "invalid YAML" in result.output


def test_run_grid_base_config_option_overrides_spec_base(tmp_path: Path) -> None:
    override = tmp_path / "base.yml"
    override.write_text("case_name: override\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "study",
            "run-grid",
            "--spec",
            str(MINI_SPEC_PATH),
            "--dry-run",
            "--base-config",
            str(override),
        ],
    )

    assert result.exit_code == 0, result.output
    assert str(override.resolve()) in result.output


def test_run_grid_real_execution_can_use_injected_runner_for_smoke() -> None:
    calls: list[int] = []

    def fake_run_case(_config, scenario):
        calls.append(scenario.index)
        return pd.DataFrame(
            {
                "Time_s": [0.0, 0.1],
                "Impact_Force_MN": [0.0, 1.2],
                "Penetration_mm": [0.0, 3.4],
                "Acceleration_g": [0.0, 0.5],
            }
        )

    _definition, _base, rows = run_grid_from_yaml(
        MINI_SPEC_PATH,
        limit=1,
        run_case_fn=fake_run_case,
    )

    assert calls == [0]
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["peak_Impact_Force_MN"] == 1.2
    assert rows[0]["peak_Penetration_mm"] == 3.4
    assert rows[0]["peak_Acceleration_g"] == 0.5
    assert rows[0]["t_end"] == 0.1
    assert rows[0]["n_steps"] == 1


def test_run_grid_non_strict_records_failed_scenario_and_continues() -> None:
    calls: list[int] = []

    def fake_run_case(_config, scenario):
        calls.append(scenario.index)
        if scenario.index == 0:
            raise RuntimeError("boom")
        return pd.DataFrame({"Time_s": [0.0], "Impact_Force_MN": [0.0]})

    _definition, _base, rows = run_grid_from_yaml(
        MINI_SPEC_PATH,
        limit=2,
        strict=False,
        run_case_fn=fake_run_case,
    )

    assert calls == [0, 1]
    assert [row["status"] for row in rows] == ["failed", "ok"]
    assert "boom" in rows[0]["error"]


def test_run_grid_strict_fails_immediately_on_failed_scenario() -> None:
    calls: list[int] = []

    def fake_run_case(_config, scenario):
        calls.append(scenario.index)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_grid_from_yaml(
            MINI_SPEC_PATH,
            limit=2,
            strict=True,
            run_case_fn=fake_run_case,
        )

    assert calls == [0]
