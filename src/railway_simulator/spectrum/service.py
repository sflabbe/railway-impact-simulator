"""Central response spectrum service.

This module intentionally wraps ``hazard.sdof.compute_response_spectrum`` so the
engine, UI and study runners all use the same SRS implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

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
            free_vibration_padding=bool(settings.free_vibration_padding),
            padding_periods=float(settings.padding_periods),
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
    def envelope(
        curves: Iterable[pd.DataFrame],
        *,
        value_column: str = "Feq_MN",
        allow_partial_overlap: bool = False,
    ) -> pd.DataFrame:
        frames = [c.copy() for c in curves]
        if not frames:
            raise ValueError("at least one curve is required")

        period_grids: list[np.ndarray] = []
        value_grids: list[np.ndarray] = []
        for frame in frames:
            if "Tn_ms" not in frame.columns:
                raise KeyError("period column not found: Tn_ms")
            if value_column not in frame.columns:
                raise KeyError(f"value column not found: {value_column}")
            source_periods = frame["Tn_ms"].to_numpy(dtype=float)
            source_values = frame[value_column].to_numpy(dtype=float)
            if source_periods.ndim != 1 or source_periods.size == 0 or not np.all(np.isfinite(source_periods)):
                raise ValueError("Tn_ms must be a finite, non-empty 1D period grid")
            if np.any(np.diff(source_periods) <= 0.0):
                raise ValueError("Tn_ms must be strictly increasing")
            if not np.all(np.isfinite(source_values)):
                raise ValueError(f"{value_column} must contain only finite values")
            period_grids.append(source_periods)
            value_grids.append(source_values)

        overlap_low = max(float(periods[0]) for periods in period_grids)
        overlap_high = min(float(periods[-1]) for periods in period_grids)
        if overlap_high < overlap_low:
            raise ValueError("period grids do not overlap")

        if allow_partial_overlap:
            target_periods = np.unique(np.concatenate(period_grids))
        else:
            target_periods = np.unique(
                np.concatenate(
                    [
                        np.array([overlap_low, overlap_high], dtype=float),
                        *[
                            periods[(periods >= overlap_low) & (periods <= overlap_high)]
                            for periods in period_grids
                        ],
                    ]
                )
            )

        values = [
            np.interp(
                target_periods,
                source_periods,
                source_values,
                left=np.nan,
                right=np.nan,
            )
            for source_periods, source_values in zip(period_grids, value_grids)
        ]
        arr = np.vstack(values)
        contributing = np.sum(np.isfinite(arr), axis=0)
        if not allow_partial_overlap and np.any(contributing != len(frames)):
            raise ValueError("internal envelope grid includes periods outside the common overlap")
        if allow_partial_overlap and np.any(contributing < len(frames)):
            warnings.warn(
                "Spectrum envelope evaluated with partial period overlap; "
                "see n_contributing_curves for per-period coverage.",
                RuntimeWarning,
                stacklevel=2,
            )

        data = {
            "Tn_ms": target_periods,
            f"{value_column}_envelope": np.nanmax(arr, axis=0),
        }
        if allow_partial_overlap:
            data["n_contributing_curves"] = contributing
        return pd.DataFrame(data)

    @staticmethod
    def ratio(numerator: pd.DataFrame, denominator: pd.DataFrame, *, value_column: str = "Feq_MN") -> pd.DataFrame:
        if value_column not in numerator.columns or value_column not in denominator.columns:
            raise KeyError(f"value column not found: {value_column}")
        periods = numerator["Tn_ms"].to_numpy(dtype=float)
        den = np.interp(periods, denominator["Tn_ms"].to_numpy(dtype=float), denominator[value_column].to_numpy(dtype=float))
        num = numerator[value_column].to_numpy(dtype=float)
        ratio = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=np.abs(den) > 0.0)
        return pd.DataFrame({"Tn_ms": periods, f"{value_column}_ratio": ratio})
