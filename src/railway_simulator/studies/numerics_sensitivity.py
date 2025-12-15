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

    rows: List[Dict[str, Any]] = []
    baseline_peak: Optional[float] = None

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
                impulse = float(np.trapz(f, t))
                max_pen = float(np.nanmax(df.get("Penetration_mm", pd.Series([np.nan])).to_numpy()))
                e_final = float(df.get("E_balance_error_J", pd.Series([np.nan])).iloc[-1]) if len(df) else float("nan")

                if baseline_peak is None:
                    baseline_peak = peak
                rel_peak = None if baseline_peak == 0 else 100.0 * (peak - baseline_peak) / baseline_peak

                row = {
                    "dt_s": float(dt),
                    "alpha_hht": float(alpha),
                    "newton_tol": float(tol),
                    "peak_force_MN": peak,
                    "peak_force_rel_to_baseline_pct": rel_peak,
                    "impulse_MN_s": impulse,
                    "max_penetration_mm": max_pen,
                    "energy_balance_error_J_final": e_final,
                    "n_lu": int(getattr(df, "attrs", {}).get("n_lu", 0)),
                    "n_nonlinear_iters": int(getattr(df, "attrs", {}).get("n_nonlinear_iters", 0)),
                    "step": int(cfg.get("step", np.nan)),
                }
                rows.append(row)

                if out_dir and save_timeseries:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out_dir / f"timeseries_dt_{dt:.3e}_a_{alpha:+.3f}_tol_{tol:.1e}.csv", index=False)

    summary = pd.DataFrame(rows)

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
            },
        )
    return summary
