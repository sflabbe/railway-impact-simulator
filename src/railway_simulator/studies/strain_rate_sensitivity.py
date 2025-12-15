"""
Fixed-DIF sensitivity study.

This is a pragmatic first step to test the influence of a "dynamic increase factor" (DIF)
without introducing a full strain-rate material law. The DIF is applied as a multiplier
to a stiffness-like parameter (default: k_wall).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

from . import (
    get_by_path,
    harmonize_time_grid,
    merge_with_engine_defaults,
    save_study_metadata,
    set_by_path,
)

SimFunc = Callable[[Dict[str, Any]], pd.DataFrame]


def run_fixed_dif_sensitivity(
    cfg_overrides: Dict[str, Any],
    difs: Iterable[float],
    *,
    k_path: str = "k_wall",
    quantity: str = "Impact_Force_MN",
    out_dir: Optional[Path] = None,
    save_timeseries: bool = False,
    simulate_func: Optional[SimFunc] = None,
) -> pd.DataFrame:
    """
    Apply `k_scaled = k0 * dif` and run the simulation for each DIF.

    Parameters
    ----------
    cfg_overrides:
        Base YAML overrides.
    difs:
        Iterable of DIF multipliers.
    k_path:
        Parameter path to be scaled (supports e.g. "fy[0]" too).
    """
    if simulate_func is None:
        from railway_simulator.core.engine import run_simulation as simulate_func  # type: ignore

    base_full = merge_with_engine_defaults(cfg_overrides)
    base_full = harmonize_time_grid(base_full)

    k0 = float(get_by_path(base_full, k_path))

    rows: List[Dict[str, Any]] = []

    for dif in difs:
        dif = float(dif)
        k_scaled = k0 * dif
        cfg = set_by_path(base_full, k_path, k_scaled)
        cfg = harmonize_time_grid(cfg)

        df = simulate_func(cfg)

        t = df["Time_s"].to_numpy()
        f = df[quantity].to_numpy()
        peak = float(np.nanmax(f))
        impulse = float(np.trapz(f, t))
        max_pen = float(np.nanmax(df.get("Penetration_mm", pd.Series([np.nan])).to_numpy()))
        max_acc = float(np.nanmax(df.get("Acceleration_g", pd.Series([np.nan])).to_numpy()))
        e_final = float(df.get("E_balance_error_J", pd.Series([np.nan])).iloc[-1]) if len(df) else float("nan")

        rows.append(
            {
                "dif": dif,
                "k_path": k_path,
                "k0_N_m": k0,
                "k_scaled_N_m": k_scaled,
                "peak_force_MN": peak,
                "impulse_MN_s": impulse,
                "max_penetration_mm": max_pen,
                "max_acceleration_g": max_acc,
                "energy_balance_error_J_final": e_final,
            }
        )

        if out_dir and save_timeseries:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / f"timeseries_dif_{dif:.3f}.csv", index=False)

    summary = pd.DataFrame(rows)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "fixed_dif_summary.csv", index=False)
        (out_dir / "config_overrides.yml").write_text(
            yaml.safe_dump(cfg_overrides, sort_keys=False), encoding="utf-8"
        )
        save_study_metadata(
            out_dir,
            metadata={
                "study_type": "strain_rate_fixed_dif",
                "difs": list(map(float, difs)),
                "k_path": k_path,
                "quantity": quantity,
                "save_timeseries": bool(save_timeseries),
            },
        )

    return summary
