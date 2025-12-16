"""
Numerical sensitivity study.

Purpose: quantify the sensitivity of key outputs to *numerical* parameters such as:
- time step (h_init)
- HHT-alpha parameter (alpha_hht)
- nonlinear solver tolerance (newton_tol)

This helps justify numerical choices in a thesis/report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from . import harmonize_time_grid, merge_with_engine_defaults, save_study_metadata

SimFunc = Callable[[Dict[str, Any]], pd.DataFrame]


def _interpolate_peak_parabolic(t: np.ndarray, f: np.ndarray) -> Tuple[float, float]:
    """
    3-point parabolic interpolation around the peak to get sub-sample resolution.

    Returns
    -------
    peak_interp : float
        Interpolated peak value.
    t_peak_interp : float
        Interpolated time of peak.

    Notes
    -----
    Fits a parabola through (t[i-1], f[i-1]), (t[i], f[i]), (t[i+1], f[i+1])
    where i is the index of max(f), and finds the vertex.
    If i is at the boundary, returns the sampled peak.
    """
    idx_peak = int(np.nanargmax(f))
    n = len(f)

    # Boundary check: need at least one point on each side
    if idx_peak == 0 or idx_peak == n - 1 or n < 3:
        return float(f[idx_peak]), float(t[idx_peak])

    # Three points around peak
    i = idx_peak
    t0, t1, t2 = t[i - 1], t[i], t[i + 1]
    f0, f1, f2 = f[i - 1], f[i], f[i + 1]

    # Parabola: f(t) = a*(t - t1)^2 + b*(t - t1) + c
    # We know f(t1) = f1, so c = f1
    # Solve for a and b using the other two points
    dt0 = t0 - t1
    dt2 = t2 - t1
    df0 = f0 - f1
    df2 = f2 - f1

    # System: a*dt0^2 + b*dt0 = df0
    #         a*dt2^2 + b*dt2 = df2
    denom = dt0 * dt2 * (dt2 - dt0)
    if abs(denom) < 1e-15:
        # Degenerate case (points too close or collinear)
        return float(f1), float(t1)

    a = (df0 * dt2 - df2 * dt0) / denom
    b = (df2 * dt0 * dt0 - df0 * dt2 * dt2) / denom

    # Vertex of parabola: t_vertex = t1 - b / (2*a)
    if abs(a) < 1e-15:
        # Nearly linear, no well-defined peak
        return float(f1), float(t1)

    t_vertex = t1 - b / (2.0 * a)
    f_vertex = a * (t_vertex - t1) ** 2 + b * (t_vertex - t1) + f1

    # Sanity check: vertex should be close to the peak region
    if t_vertex < t0 or t_vertex > t2:
        # Vertex outside interval, use sampled peak
        return float(f1), float(t1)

    return float(f_vertex), float(t_vertex)


def run_numerics_sensitivity(
    cfg_overrides: Dict[str, Any],
    *,
    dt_values: Optional[Sequence[float]] = None,
    alpha_values: Optional[Sequence[float]] = None,
    tol_values: Optional[Sequence[float]] = None,
    quantity: str = "Impact_Force_MN",
    out_dir: Optional[Path] = None,
    save_timeseries: bool = False,
    simulate_func: Optional[SimFunc] = None,
) -> pd.DataFrame:
    """
    Run a cartesian sweep over numeric parameters.

    If a list is None, the base value from config defaults/overrides is used.
    """
    if simulate_func is None:
        from railway_simulator.core.engine import run_simulation as simulate_func  # type: ignore

    base_full = merge_with_engine_defaults(cfg_overrides)
    base_full = harmonize_time_grid(base_full)

    dt_values = list(dt_values) if dt_values is not None else [float(base_full.get("h_init", 1e-4))]
    alpha_values = list(alpha_values) if alpha_values is not None else [float(base_full.get("alpha_hht", -0.15))]
    tol_values = list(tol_values) if tol_values is not None else [float(base_full.get("newton_tol", 1e-5))]

    # Identify the finest resolution case (most accurate) as baseline:
    # - Smallest dt
    # - Smallest (most negative) alpha for HHT stability
    # - Tightest (smallest) tolerance
    baseline_dt = min(dt_values)
    baseline_alpha = min(alpha_values)  # Most negative is most stable/accurate
    baseline_tol = min(tol_values)

    rows: List[Dict[str, Any]] = []
    baseline_peak: Optional[float] = None
    baseline_peak_interp: Optional[float] = None

    for dt in dt_values:
        for alpha in alpha_values:
            for tol in tol_values:
                cfg = dict(base_full)
                cfg["h_init"] = float(dt)
                cfg["alpha_hht"] = float(alpha)
                cfg["newton_tol"] = float(tol)
                cfg = harmonize_time_grid(cfg)

                df = simulate_func(cfg)

                t = df["Time_s"].to_numpy()
                f = df[quantity].to_numpy()

                peak = float(np.nanmax(f))
                peak_interp, t_peak_interp = _interpolate_peak_parabolic(t, f)
                impulse = float(np.trapz(f, t))
                max_pen = float(np.nanmax(df.get("Penetration_mm", pd.Series([np.nan])).to_numpy()))
                e_final = float(df.get("E_balance_error_J", pd.Series([np.nan])).iloc[-1]) if len(df) else float("nan")

                # Store baseline peak from finest resolution case
                is_baseline = (dt == baseline_dt and alpha == baseline_alpha and tol == baseline_tol)
                if is_baseline:
                    baseline_peak = peak
                    baseline_peak_interp = peak_interp

                attrs = getattr(df, "attrs", {})
                row = {
                    "dt_requested": float(dt),
                    "dt_eff": float(attrs.get("dt_eff", dt)),
                    "alpha_hht": float(alpha),
                    "newton_tol": float(tol),
                    "peak_force_MN": peak,
                    "peak_force_interp_MN": peak_interp,
                    "time_of_peak_s": float(df["Time_s"].iloc[int(np.nanargmax(f))]),
                    "time_of_peak_interp_s": t_peak_interp,
                    "impulse_MN_s": impulse,
                    "max_penetration_mm": max_pen,
                    "energy_balance_error_J_final": e_final,
                    "n_lu": int(attrs.get("n_lu", 0)),
                    "n_nonlinear_iters": int(attrs.get("n_nonlinear_iters", 0)),
                    "max_iters_per_step": int(attrs.get("max_iters_per_step", 0)),
                    "max_residual": float(attrs.get("max_residual", float("nan"))),
                    "step": int(cfg.get("step", np.nan)),
                    "is_baseline": is_baseline,
                }
                rows.append(row)

                if out_dir and save_timeseries:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out_dir / f"timeseries_dt_{dt:.3e}_a_{alpha:+.3f}_tol_{tol:.1e}.csv", index=False)

    summary = pd.DataFrame(rows)

    # Compute relative errors with respect to finest resolution case
    if baseline_peak is not None and baseline_peak != 0:
        summary["peak_force_rel_to_baseline_pct"] = (
            100.0 * (summary["peak_force_MN"] - baseline_peak) / baseline_peak
        )
    else:
        summary["peak_force_rel_to_baseline_pct"] = float("nan")

    if baseline_peak_interp is not None and baseline_peak_interp != 0:
        summary["peak_force_interp_rel_to_baseline_pct"] = (
            100.0 * (summary["peak_force_interp_MN"] - baseline_peak_interp) / baseline_peak_interp
        )
    else:
        summary["peak_force_interp_rel_to_baseline_pct"] = float("nan")

    # Drop internal flag column
    summary = summary.drop(columns=["is_baseline"], errors="ignore")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "numerics_sensitivity_summary.csv", index=False)
        (out_dir / "config_overrides.yml").write_text(
            yaml.safe_dump(cfg_overrides, sort_keys=False), encoding="utf-8"
        )
        save_study_metadata(
            out_dir,
            metadata={
                "study_type": "numerics_sensitivity",
                "dt_values": list(map(float, dt_values)),
                "alpha_values": list(map(float, alpha_values)),
                "tol_values": list(map(float, tol_values)),
                "quantity": quantity,
                "save_timeseries": bool(save_timeseries),
                "baseline_dt": float(baseline_dt),
                "baseline_alpha": float(baseline_alpha),
                "baseline_tol": float(baseline_tol),
                "baseline_peak": float(baseline_peak) if baseline_peak is not None else None,
                "baseline_peak_interp": float(baseline_peak_interp) if baseline_peak_interp is not None else None,
            },
        )
    return summary
