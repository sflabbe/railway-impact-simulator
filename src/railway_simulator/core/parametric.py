from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .engine import run_simulation


@dataclass
class ScenarioDefinition:
    """
    Definition of a parametric scenario.

    Attributes
    ----------
    name : str
        Scenario identifier (e.g. 'v320').
    params : dict
        Simulation parameter dict passed to `run_simulation`.
    weight : float, optional
        Statistical weight in [0, 1] for weighted mean history.
    meta : dict, optional
        Additional metadata (e.g. speed_kmh) copied into the summary table.
    """
    name: str
    params: Dict[str, Any]
    weight: float = 1.0
    meta: Dict[str, Any] | None = None


def run_parametric_envelope(
    scenarios: List[ScenarioDefinition],
    quantity: str = "Impact_Force_MN",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run a family of simulations and compute an envelope ('Umhüllende').

    Parameters
    ----------
    scenarios :
        List of ScenarioDefinition objects.
    quantity :
        Column name in the results DataFrame to envelope
        (e.g. 'Impact_Force_MN').

    Returns
    -------
    envelope_df :
        Time history DataFrame with columns:

        - 'Time_s'
        - 'Time_ms'
        - f'{quantity}_envelope'
        - f'{quantity}_weighted_mean'

    summary_df :
        One row per scenario with columns:

        - 'scenario', 'weight', 'peak', 'time_of_peak_s'
        - 'n_lu', 'n_dof'
        - plus any keys from `meta`.

    meta :
        Currently an empty dict, reserved for future extensions.
    """
    if not scenarios:
        raise ValueError("run_parametric_envelope: no scenarios provided.")

    results: List[Tuple[ScenarioDefinition, pd.DataFrame]] = []
    time_s_ref: np.ndarray | None = None

    for scen in scenarios:
        df = run_simulation(scen.params)

        if "Time_s" not in df.columns:
            raise KeyError("Result DataFrame is missing 'Time_s' column.")

        t = df["Time_s"].to_numpy()
        if time_s_ref is None:
            time_s_ref = t
        else:
            if t.shape != time_s_ref.shape or not np.allclose(t, time_s_ref):
                raise ValueError(
                    "All scenarios must share the same time grid for envelope computation."
                )

        results.append((scen, df))

    assert time_s_ref is not None
    time_s = time_s_ref
    time_ms = time_s * 1000.0

    # Stack quantity histories into a 2D array
    values = []
    weights = []
    for scen, df in results:
        if quantity not in df.columns:
            raise KeyError(
                f"Column '{quantity}' not found in results for scenario '{scen.name}'."
            )
        values.append(df[quantity].to_numpy())
        weights.append(scen.weight)

    values_arr = np.vstack(values)
    weights_arr = np.asarray(weights, dtype=float)
    if np.any(weights_arr < 0.0):
        raise ValueError("Scenario weights must be non-negative.")

    if weights_arr.sum() > 0.0:
        weights_arr = weights_arr / weights_arr.sum()
    else:
        # all weights == 0 -> treat as equal
        weights_arr[:] = 1.0 / len(weights_arr)

    env_vals = np.nanmax(values_arr, axis=0)
    mean_vals = np.average(values_arr, axis=0, weights=weights_arr)

    envelope_df = pd.DataFrame(
        {
            "Time_s": time_s,
            "Time_ms": time_ms,
            f"{quantity}_envelope": env_vals,
            f"{quantity}_weighted_mean": mean_vals,
        }
    )

    # Per-scenario summary table
    rows: List[Dict[str, Any]] = []
    for (scen, df), w in zip(results, weights_arr):
        q = df[quantity].to_numpy()
        peak = float(np.nanmax(q))
        idx_peak = int(np.nanargmax(q))
        t_peak = float(df["Time_s"].iloc[idx_peak])

        row: Dict[str, Any] = {
            "scenario": scen.name,
            "weight": float(w),
            "peak": peak,
            "time_of_peak_s": t_peak,
            "n_lu": df.attrs.get("n_lu", 0),
            "n_dof": df.attrs.get("n_dof", 0),
        }

        if scen.meta:
            for k, v in scen.meta.items():
                if k not in row:
                    row[k] = v

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    meta: Dict[str, Any] = {}

    return envelope_df, summary_df, meta

