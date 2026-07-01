from __future__ import annotations

import numpy as np
import pandas as pd


def wall_force_deltas(df: pd.DataFrame, *, feq_column_suffix: str = "") -> dict[str, float]:
    """Quantify total-wall force deltas relative to the legacy front column."""
    front = df["Impact_Force_front_MN"].to_numpy(dtype=float)
    total = df["Impact_Force_wall_total_MN"].to_numpy(dtype=float)
    front_peak = float(np.nanmax(front))
    total_peak = float(np.nanmax(total))
    delta_fpeak = float(total_peak / front_peak - 1.0) if front_peak > 0.0 else np.nan

    front_feq_col = f"Feq_front{feq_column_suffix}"
    total_feq_col = f"Feq_wall_total{feq_column_suffix}"
    if front_feq_col in df.columns and total_feq_col in df.columns:
        front_feq = float(np.nanmax(df[front_feq_col].to_numpy(dtype=float)))
        total_feq = float(np.nanmax(df[total_feq_col].to_numpy(dtype=float)))
        delta_feq = float(total_feq / front_feq - 1.0) if front_feq > 0.0 else np.nan
    else:
        delta_feq = np.nan

    return {"delta_Fpeak": delta_fpeak, "delta_Feq": delta_feq}
