"""Simulation run and metric domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from railway_simulator.domain.common import new_id, utc_now_iso


@dataclass(frozen=True)
class SimulationRun:
    """Persisted execution of one scenario."""

    scenario_id: str
    status: str
    config_hash: str
    result_csv_path: Path | None = None
    elapsed_s: float | None = None
    error_message: str | None = None
    id: str = field(default_factory=lambda: new_id("run"))
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None


@dataclass(frozen=True)
class RunMetric:
    """Scalar metric extracted from a time history."""

    run_id: str
    name: str
    value: float
    unit: str = ""
    id: str = field(default_factory=lambda: new_id("met"))
