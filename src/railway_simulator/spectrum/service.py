"""Central response spectrum service.

This module intentionally wraps ``hazard.sdof.compute_response_spectrum`` so the
engine, UI and study runners all use the same SRS implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from railway_simulator.domain.result import SimulationRun
from railway_simulator.domain.spectrum import SRSCurve, SRSSettings
from railway_simulator.hazard.sdof import compute_response_spectrum
from railway_simulator.persistence.repositories import SpectrumRepository


class SpectrumService:
    """Compute, compare and persist response spectrum equivalent-force curves."""

    def __init__(self, *, repository: SpectrumRepository | None = None, spectra_dir: str | Path | None = None):
        self.repository = repository
        self.spectra_dir = Path(spectra_dir) if spectra_dir is not None else None

    def compute_srs(self, df: pd.DataFrame, settings: SRSSettings | None = None) -> pd.DataFrame:
        settings = settings or SRSSettings()
        if settings.time_column not in df.columns:
            raise KeyError(f"time column not found: {settings.time_column}")
        if settings.force_column not in df.columns:
            raise KeyError(f"force column not found: {settings.force_column}")
        periods = None
        if settings.Tn_grid_ms is not None:
            periods = np.asarray(settings.Tn_grid_ms, dtype=float)
        out = compute_response_spectrum(
            df[settings.time_column].to_numpy(dtype=float),
            df[settings.force_column].to_numpy(dtype=float),
            Tn_grid_ms=periods,
            zeta=float(settings.zeta),
            oscillator_mass=float(settings.oscillator_mass),
        )
        out = out.rename(columns={"Feq": "Feq_MN" if settings.force_column.endswith("_MN") else "Feq"})
        out.insert(0, "zeta", float(settings.zeta))
        out.insert(1, "force_column", settings.force_column)
        return out

    def compute_and_store(self, run: SimulationRun, settings: SRSSettings | None = None) -> SRSCurve:
        if run.result_csv_path is None:
            raise ValueError("run has no result_csv_path")
        settings = settings or SRSSettings()
        df = pd.read_csv(run.result_csv_path)
        curve_df = self.compute_srs(df, settings)
        spectra_dir = self.spectra_dir or Path(run.result_csv_path).parent / "spectra"
        spectra_dir.mkdir(parents=True, exist_ok=True)
        path = spectra_dir / f"{run.id}__{settings.force_column}__zeta{settings.zeta:g}.csv"
        curve_df.to_csv(path, index=False)
        curve = SRSCurve(
            run_id=run.id,
            zeta=float(settings.zeta),
            oscillator_mass=float(settings.oscillator_mass),
            force_column=settings.force_column,
            curve_csv_path=path,
        )
        if self.repository is not None:
            self.repository.add_curve(curve)
        return curve

    @staticmethod
    def load_curve(curve: SRSCurve) -> pd.DataFrame:
        df = pd.read_csv(curve.curve_csv_path)
        if "run_id" not in df.columns:
            df.insert(0, "run_id", curve.run_id)
        return df

    @staticmethod
    def envelope(curves: Iterable[pd.DataFrame], *, value_column: str = "Feq_MN") -> pd.DataFrame:
        frames = [c.copy() for c in curves]
        if not frames:
            raise ValueError("at least one curve is required")
        base_periods = frames[0]["Tn_ms"].to_numpy(dtype=float)
        values = []
        for frame in frames:
            if value_column not in frame.columns:
                raise KeyError(f"value column not found: {value_column}")
            values.append(np.interp(base_periods, frame["Tn_ms"].to_numpy(dtype=float), frame[value_column].to_numpy(dtype=float)))
        arr = np.vstack(values)
        return pd.DataFrame({"Tn_ms": base_periods, f"{value_column}_envelope": np.nanmax(arr, axis=0)})

    @staticmethod
    def ratio(numerator: pd.DataFrame, denominator: pd.DataFrame, *, value_column: str = "Feq_MN") -> pd.DataFrame:
        if value_column not in numerator.columns or value_column not in denominator.columns:
            raise KeyError(f"value column not found: {value_column}")
        periods = numerator["Tn_ms"].to_numpy(dtype=float)
        den = np.interp(periods, denominator["Tn_ms"].to_numpy(dtype=float), denominator[value_column].to_numpy(dtype=float))
        num = numerator[value_column].to_numpy(dtype=float)
        ratio = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=np.abs(den) > 0.0)
        return pd.DataFrame({"Tn_ms": periods, f"{value_column}_ratio": ratio})
