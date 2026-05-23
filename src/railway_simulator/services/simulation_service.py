"""Simulation execution service with project persistence."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from railway_simulator.core.engine import run_simulation
from railway_simulator.domain.common import utc_now_iso
from railway_simulator.domain.result import RunMetric, SimulationRun
from railway_simulator.domain.scenario import Scenario
from railway_simulator.persistence.repositories import StudyRepository


def extract_run_metrics(df: pd.DataFrame) -> list[tuple[str, float, str]]:
    """Extract stable scalar metrics from a time history DataFrame."""
    metrics: list[tuple[str, float, str]] = []
    if "Impact_Force_MN" in df.columns:
        force = df["Impact_Force_MN"].to_numpy(dtype=float)
        metrics.append(("Fpeak", float(np.nanmax(force)), "MN"))
        if "Time_s" in df.columns and len(force) >= 2:
            t = df["Time_s"].to_numpy(dtype=float)
            integrate_trapezoid = getattr(np, "trapezoid", None)
            if integrate_trapezoid is None:
                integrate_trapezoid = np.trapz
            impulse = float(integrate_trapezoid(np.nan_to_num(force), t))
            metrics.append(("Impulse", impulse, "MN*s"))
    if "Penetration_mm" in df.columns:
        metrics.append(("Penetration_max", float(np.nanmax(df["Penetration_mm"].to_numpy(dtype=float))), "mm"))
    if "Time_s" in df.columns and len(df):
        metrics.append(("T_end", float(df["Time_s"].iloc[-1]), "s"))
    return metrics


class SimulationService:
    """Run one scenario and persist its result CSV plus scalar metrics."""

    def __init__(self, *, repository: StudyRepository | None = None, runs_dir: str | Path | None = None):
        self.repository = repository
        self.runs_dir = Path(runs_dir) if runs_dir is not None else Path("runs")

    def run_scenario(self, scenario: Scenario) -> SimulationRun:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        started_at = utc_now_iso()
        start = time.perf_counter()
        try:
            df = run_simulation(dict(scenario.params), emit_peak_diagnostics=False)
            elapsed = time.perf_counter() - start
            run = SimulationRun(
                scenario_id=scenario.id,
                status="ok",
                config_hash=scenario.config_hash,
                result_csv_path=self.runs_dir / f"{scenario.id}.csv",
                elapsed_s=elapsed,
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
            df.to_csv(run.result_csv_path, index=False)
            if self.repository is not None:
                self.repository.add_run(run)
                for name, value, unit in extract_run_metrics(df):
                    self.repository.add_metric(RunMetric(run_id=run.id, name=name, value=value, unit=unit))
            return run
        except Exception as exc:
            elapsed = time.perf_counter() - start
            run = SimulationRun(
                scenario_id=scenario.id,
                status="failed",
                config_hash=scenario.config_hash,
                elapsed_s=elapsed,
                error_message=repr(exc),
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
            if self.repository is not None:
                self.repository.add_run(run)
            return run
