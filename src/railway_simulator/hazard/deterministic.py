"""Deterministic runout tables for structural-dynamics dissertation studies.

This module intentionally avoids probabilistic mechanism distributions.  It
computes conditional scenario rows for prescribed ``v0``, wall offset ``a``,
friction ``mu`` and derailment/velocity angle ``beta_d``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .runout import impact_velocity_ms, kmh_to_ms, max_lateral_reach_m, normal_velocity_ms


DEFAULT_GUIDANCE_THRESHOLD_MS = 1.7


@dataclass(frozen=True)
class ResponseMetrics:
    """Optional force-response metrics attached to a deterministic row."""

    f_peak_MN: float
    f_eq_MN: float
    response_model: str


ResponseProvider = Callable[[float], ResponseMetrics | None]


def beta_crit_deg(v_imp_ms: float, threshold_ms: float = DEFAULT_GUIDANCE_THRESHOLD_MS) -> float:
    """Return critical angle for ``v_imp*sin(beta)=threshold``.

    Returns NaN if the threshold cannot be reached because ``v_imp_ms`` is zero
    or below ``threshold_ms``.
    """
    v_imp = float(v_imp_ms)
    threshold = float(threshold_ms)
    if not math.isfinite(v_imp) or v_imp <= 0.0:
        return math.nan
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_ms must be finite and > 0")
    if v_imp < threshold:
        return math.nan
    return float(math.degrees(math.asin(min(1.0, threshold / v_imp))))


def lateral_regime_label(
    v_imp_ms: float,
    v_n_ms: float,
    threshold_ms: float = DEFAULT_GUIDANCE_THRESHOLD_MS,
) -> str:
    """Classify a deterministic scenario relative to the guidance threshold."""
    v_imp = float(v_imp_ms)
    vn = float(v_n_ms)
    if not math.isfinite(v_imp) or not math.isfinite(vn):
        raise ValueError("v_imp_ms and v_n_ms must be finite")
    if v_imp <= 0.0 or vn <= 0.0:
        return "no_impact"
    if vn < threshold_ms:
        return "guided_compatible"
    return "beyond_substitute_guidance"


def deterministic_runout_row(
    *,
    vehicle: str,
    v0_kmh: float,
    a_m: float,
    mu: float,
    beta_d_deg: float,
    threshold_ms: float = DEFAULT_GUIDANCE_THRESHOLD_MS,
    response_provider: ResponseProvider | None = None,
) -> dict:
    """Compute one conditional deterministic scenario row."""
    v0_ms = kmh_to_ms(v0_kmh)
    v_imp = impact_velocity_ms(v0_kmh, a_m, mu, beta_d_deg)
    v_n = normal_velocity_ms(v_imp, beta_d_deg)
    beta_crit = beta_crit_deg(v_imp, threshold_ms)
    a_max = max_lateral_reach_m(v0_kmh, mu, beta_d_deg)
    reaches = bool(v_imp > 0.0 or a_m == 0.0)

    row = {
        "vehicle": vehicle,
        "v0_kmh": float(v0_kmh),
        "v0_ms": float(v0_ms),
        "a_m": float(a_m),
        "mu": float(mu),
        "beta_d_deg": float(beta_d_deg),
        "a_max_m": float(a_max),
        "reaches_wall": reaches,
        "v_imp_ms": float(v_imp),
        "v_imp_kmh": float(v_imp * 3.6),
        "v_n_ms": float(v_n),
        "v_n_kmh": float(v_n * 3.6),
        "beta_crit_deg": beta_crit,
        "beta_over_beta_crit": bool(math.isfinite(beta_crit) and beta_d_deg >= beta_crit),
        "guidance_threshold_ms": float(threshold_ms),
        "regime": lateral_regime_label(v_imp, v_n, threshold_ms),
    }

    if response_provider is not None and v_n > 0.0:
        metrics = response_provider(v_n)
        if metrics is not None:
            row.update(
                {
                    "f_peak_MN": float(metrics.f_peak_MN),
                    "f_eq_MN": float(metrics.f_eq_MN),
                    "response_model": metrics.response_model,
                }
            )
    return row


def deterministic_runout_grid(
    *,
    vehicle: str,
    speeds_kmh: Sequence[float],
    distances_m: Sequence[float],
    mu_values: Sequence[float],
    beta_values_deg: Sequence[float],
    threshold_ms: float = DEFAULT_GUIDANCE_THRESHOLD_MS,
    response_provider: ResponseProvider | None = None,
) -> pd.DataFrame:
    """Build the conditional deterministic grid used in the dissertation table."""
    rows: list[dict] = []
    for v0 in speeds_kmh:
        for a in distances_m:
            for mu in mu_values:
                for beta in beta_values_deg:
                    rows.append(
                        deterministic_runout_row(
                            vehicle=vehicle,
                            v0_kmh=float(v0),
                            a_m=float(a),
                            mu=float(mu),
                            beta_d_deg=float(beta),
                            threshold_ms=threshold_ms,
                            response_provider=response_provider,
                        )
                    )
    return pd.DataFrame(rows)


def compact_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-page-friendly summary by speed/distance/beta.

    The summary collapses friction cases by reporting ranges for impact velocity
    and normal velocity.  This is useful for a dissertation text table while the
    full CSV preserves all scenario rows.
    """
    required = {"v0_kmh", "a_m", "beta_d_deg", "v_imp_kmh", "v_n_ms", "regime"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    grouped = []
    for keys, sub in df.groupby(["v0_kmh", "a_m", "beta_d_deg"], sort=True):
        v0, a, beta = keys
        regimes = "/".join(sorted(set(str(x) for x in sub["regime"])))
        grouped.append(
            {
                "v0_kmh": v0,
                "a_m": a,
                "beta_d_deg": beta,
                "v_imp_kmh_min": float(sub["v_imp_kmh"].min()),
                "v_imp_kmh_max": float(sub["v_imp_kmh"].max()),
                "v_n_ms_min": float(sub["v_n_ms"].min()),
                "v_n_ms_max": float(sub["v_n_ms"].max()),
                "regime_over_mu": regimes,
            }
        )
    return pd.DataFrame(grouped)


class LogPchipResponseProvider:
    """Small wrapper for a precomputed log-log response surrogate."""

    def __init__(self, vn_grid_ms: Sequence[float], f_peak_MN: Sequence[float], f_eq_MN: Sequence[float], name: str):
        from scipy.interpolate import PchipInterpolator

        vn = np.asarray(vn_grid_ms, dtype=float)
        peak = np.asarray(f_peak_MN, dtype=float)
        feq = np.asarray(f_eq_MN, dtype=float)
        if vn.ndim != 1 or vn.size < 2:
            raise ValueError("vn_grid_ms must be a 1D sequence with length >= 2")
        if peak.shape != vn.shape or feq.shape != vn.shape:
            raise ValueError("response arrays must have the same shape as vn_grid_ms")
        if np.any(vn <= 0.0) or np.any(peak <= 0.0) or np.any(feq <= 0.0):
            raise ValueError("surrogate inputs must be strictly positive")
        if np.any(np.diff(vn) <= 0.0):
            raise ValueError("vn_grid_ms must be strictly increasing")
        self.vn_min = float(vn[0])
        self.vn_max = float(vn[-1])
        self.name = str(name)
        self._peak = PchipInterpolator(np.log(vn), np.log(peak), extrapolate=False)
        self._feq = PchipInterpolator(np.log(vn), np.log(feq), extrapolate=False)

    def __call__(self, v_n_ms: float) -> ResponseMetrics | None:
        vn = float(v_n_ms)
        if not math.isfinite(vn) or vn <= 0.0:
            return None
        if vn < self.vn_min:
            # Below the surrogate domain, use the lower endpoint conservatively
            # for table continuity; the row still reports the true v_n.
            vn_eval = self.vn_min
        elif vn > self.vn_max:
            return None
        else:
            vn_eval = vn
        x = math.log(vn_eval)
        return ResponseMetrics(
            f_peak_MN=float(math.exp(float(self._peak(x)))),
            f_eq_MN=float(math.exp(float(self._feq(x)))),
            response_model=self.name,
        )
