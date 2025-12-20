"""
Generic (single-parameter) sensitivity study.
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


def run_sensitivity_study(
    cfg_overrides: Dict[str, Any],
    *,
    param_path: str,
    values: Iterable[float],
    quantity: str = "Impact_Force_MN",
    out_dir: Optional[Path] = None,
    save_timeseries: bool = False,
    simulate_func: Optional[SimFunc] = None,
) -> pd.DataFrame:
    """
    Sweep one parameter (param_path) over `values` and summarize response.

    Notes
    -----
    `param_path` supports dot notation and optional indices: e.g. "fy[0]".
    """
    if simulate_func is None:
        from railway_simulator.core.engine import run_simulation as simulate_func  # type: ignore

    base_full = merge_with_engine_defaults(cfg_overrides)
    base_full = harmonize_time_grid(base_full)

    base_value = float(get_by_path(base_full, param_path))

    rows: List[Dict[str, Any]] = []

    for v in values:
        v = float(v)
        cfg = set_by_path(base_full, param_path, v)
        cfg = harmonize_time_grid(cfg)

        df = simulate_func(cfg)

        t = df["Time_s"].to_numpy()
        f = df[quantity].to_numpy()
        peak = float(np.nanmax(f))
        # Use trapezoid (NumPy 2.0+) or trapz (NumPy 1.x)
        trapz_func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        impulse = float(trapz_func(f, t))
        max_pen = float(np.nanmax(df.get("Penetration_mm", pd.Series([np.nan])).to_numpy()))
        max_acc = float(np.nanmax(df.get("Acceleration_g", pd.Series([np.nan])).to_numpy()))
        e_final = float(df.get("E_balance_error_J", pd.Series([np.nan])).iloc[-1]) if len(df) else float("nan")

        rows.append(
            {
                "param_path": param_path,
                "base_value": base_value,
                "param_value": v,
                "peak_force_MN": peak,
                "impulse_MN_s": impulse,
                "max_penetration_mm": max_pen,
                "max_acceleration_g": max_acc,
                "energy_balance_error_J_final": e_final,
                "n_lu": int(getattr(df, "attrs", {}).get("n_lu", 0)),
                "n_nonlinear_iters": int(getattr(df, "attrs", {}).get("n_nonlinear_iters", 0)),
            }
        )

        if out_dir and save_timeseries:
            out_dir.mkdir(parents=True, exist_ok=True)
            safe = re_sub_for_filename(str(v))
            df.to_csv(out_dir / f"timeseries_{param_path.replace('.','_')}_{safe}.csv", index=False)

    summary = pd.DataFrame(rows)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
        (out_dir / "config_overrides.yml").write_text(
            yaml.safe_dump(cfg_overrides, sort_keys=False), encoding="utf-8"
        )
        save_study_metadata(
            out_dir,
            metadata={
                "study_type": "sensitivity",
                "param_path": param_path,
                "values": list(map(float, values)),
                "quantity": quantity,
                "save_timeseries": bool(save_timeseries),
            },
        )

    return summary


def re_sub_for_filename(s: str) -> str:
    import re
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", s)
