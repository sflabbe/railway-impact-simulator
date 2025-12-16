from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
            # Strain-rate metrics
            "strain_rate_peak_1_s": float(df.attrs.get("strain_rate_peak_1_s", 0.0)),
            "strain_rate_rms_1_s": float(df.attrs.get("strain_rate_rms_1_s", 0.0)),
            "strain_rate_p95_1_s": float(df.attrs.get("strain_rate_p95_1_s", 0.0)),
        }

        if scen.meta:
            for k, v in scen.meta.items():
                if k not in row:
                    row[k] = v

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    meta: Dict[str, Any] = {}

    return envelope_df, summary_df, meta


def build_speed_scenarios(
    base_params: Dict[str, Any],
    speeds_kmh: List[float],
    weights: List[float] | None = None,
    prefix: str = "v",
) -> List[ScenarioDefinition]:
    """
    Build scenario definitions from speeds and optional weights.

    Parameters
    ----------
    base_params : dict
        Base simulation parameters (will be copied for each scenario).
    speeds_kmh : list of float
        Impact speeds in km/h.
    weights : list of float, optional
        Statistical weights for each speed. If None, equal weights are used.
    prefix : str, default "v"
        Prefix for scenario names (e.g., "v320", "v200").

    Returns
    -------
    List[ScenarioDefinition]
        List of scenario definitions ready for parametric envelope computation.
    """
    if weights is None:
        weights = [1.0 / len(speeds_kmh)] * len(speeds_kmh)
    elif len(weights) != len(speeds_kmh):
        raise ValueError("Length of weights must match length of speeds_kmh.")

    # Normalize weights
    total_w = sum(weights)
    if total_w <= 0.0:
        raise ValueError("Sum of weights must be positive.")
    weights = [w / total_w for w in weights]

    scenarios = []
    for v_kmh, w in zip(speeds_kmh, weights):
        name = f"{prefix}{int(round(v_kmh))}"
        params_i = dict(base_params)
        params_i["v0_init"] = -v_kmh / 3.6  # Convert to m/s, negative for barrier approach

        scenarios.append(
            ScenarioDefinition(
                name=name,
                params=params_i,
                weight=w,
                meta={"speed_kmh": v_kmh},
            )
        )

    return scenarios


def make_envelope_figure(
    envelope_df: pd.DataFrame,
    quantity: str,
    title: str = "Envelope",
):
    """
    Create a Plotly figure for the parametric envelope.

    Parameters
    ----------
    envelope_df : pd.DataFrame
        Envelope DataFrame with Time_ms and quantity columns.
    quantity : str
        Name of the quantity to plot (e.g., "Impact_Force_MN").
    title : str, default "Envelope"
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object.

    Raises
    ------
    ImportError
        If plotly is not available.
    RuntimeError
        If the quantity column is not found in envelope_df.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for make_envelope_figure. "
            "Install with: pip install plotly"
        )

    # Resolve column name
    if quantity in envelope_df.columns:
        y_col = quantity
    else:
        env_col = f"{quantity}_envelope"
        if env_col in envelope_df.columns:
            y_col = env_col
        else:
            non_time_cols = [
                c for c in envelope_df.columns if c not in ("Time_s", "Time_ms")
            ]
            if not non_time_cols:
                raise RuntimeError(
                    f"Could not find column for quantity '{quantity}' in envelope_df."
                )
            y_col = non_time_cols[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=envelope_df["Time_ms"],
            y=envelope_df[y_col],
            mode="lines",
            line=dict(width=2),
            name=f"Envelope {y_col}",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time [ms]",
        yaxis_title=quantity,
        height=500,
        xaxis=dict(rangemode="tozero"),
        yaxis=dict(rangemode="tozero"),
    )

    return fig

