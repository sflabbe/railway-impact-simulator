"""
Time-step convergence / numerical verification study.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

from . import harmonize_time_grid, merge_with_engine_defaults, save_study_metadata, set_by_path


SimFunc = Callable[[Dict[str, Any]], pd.DataFrame]


def _extract_metrics(df: pd.DataFrame, quantity: str = "Impact_Force_MN") -> Dict[str, float]:
    t = df["Time_s"].to_numpy()
    y = df[quantity].to_numpy()
    peak = float(np.nanmax(y))
    impulse = float(np.trapz(y, t))
    max_pen = float(np.nanmax(df.get("Penetration_mm", pd.Series([np.nan])).to_numpy()))
    max_acc = float(np.nanmax(df.get("Acceleration_g", pd.Series([np.nan])).to_numpy()))
    e_final = float(df.get("E_balance_error_J", pd.Series([np.nan])).iloc[-1]) if len(df) else float("nan")
    return {
        "peak_value": peak,
        "impulse_MN_s": impulse,
        "max_penetration_mm": max_pen,
        "max_acceleration_g": max_acc,
        "energy_balance_error_J_final": e_final,
    }


def run_convergence_study(
    cfg_overrides: Dict[str, Any],
    dt_values: Iterable[float],
    *,
    quantity: str = "Impact_Force_MN",
    out_dir: Optional[Path] = None,
    save_timeseries: bool = False,
    simulate_func: Optional[SimFunc] = None,
) -> pd.DataFrame:
    """
    Sweep the time step h_init and report convergence metrics.

    Parameters
    ----------
    cfg_overrides:
        Config overrides loaded from YAML (can be partial).
    dt_values:
        Iterable of time steps in seconds.
    quantity:
        Column name in the simulation output DataFrame to analyze.
    out_dir:
        If provided, write summary CSV + metadata.
    save_timeseries:
        If True, also save each run DataFrame as CSV.
    simulate_func:
        For testing; defaults to `railway_simulator.core.engine.run_simulation`.

    Returns
    -------
    pd.DataFrame with one row per dt.
    """
    import time

    if simulate_func is None:
        from railway_simulator.core.engine import run_simulation as simulate_func  # type: ignore

    base_full = merge_with_engine_defaults(cfg_overrides)

    rows: List[Dict[str, Any]] = []
    prev_peak: Optional[float] = None

    for dt in dt_values:
        dt = float(dt)
        cfg = dict(base_full)
        cfg["h_init"] = dt
        cfg = harmonize_time_grid(cfg)

        t0 = time.perf_counter()
        df = simulate_func(cfg)
        wall = time.perf_counter() - t0

        metrics = _extract_metrics(df, quantity=quantity)
        peak = metrics["peak_value"]
        rel = None if prev_peak is None or prev_peak == 0 else 100.0 * abs(peak - prev_peak) / abs(prev_peak)
        prev_peak = peak

        rows.append(
            {
                "dt_s": dt,
                "step": int(cfg.get("step", np.nan)),
                "wall_time_s": float(wall),
                "n_lu": int(getattr(df, "attrs", {}).get("n_lu", 0)),
                "n_nonlinear_iters": int(getattr(df, "attrs", {}).get("n_nonlinear_iters", 0)),
                "relative_change_peak_pct": rel,
                **metrics,
            }
        )

        if out_dir and save_timeseries:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / f"timeseries_dt_{dt:.3e}.csv", index=False)

    summary = pd.DataFrame(rows).sort_values("dt_s", ascending=False).reset_index(drop=True)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "convergence_summary.csv", index=False)
        (out_dir / "config_overrides.yml").write_text(
            yaml.safe_dump(cfg_overrides, sort_keys=False), encoding="utf-8"
        )
        save_study_metadata(
            out_dir,
            metadata={
                "study_type": "convergence",
                "dt_values": list(map(float, dt_values)),
                "quantity": quantity,
                "save_timeseries": bool(save_timeseries),
            },
        )
    return summary
