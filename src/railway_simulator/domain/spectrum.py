"""SRS domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from railway_simulator.domain.common import new_id, utc_now_iso


@dataclass(frozen=True)
class SRSSettings:
    """Settings for response spectrum equivalent force curves."""

    force_column: str = "Impact_Force_MN"
    time_column: str = "Time_s"
    zeta: float = 0.05
    oscillator_mass: float = 1.0
    Tn_grid_ms: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SRSCurve:
    """Persisted response spectrum curve."""

    run_id: str
    zeta: float
    force_column: str
    curve_csv_path: Path
    oscillator_mass: float = 1.0
    id: str = field(default_factory=lambda: new_id("srs"))
    created_at: str = field(default_factory=utc_now_iso)
