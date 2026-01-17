"""
Plotting utilities for the Railway Impact Simulator UI.

Provides comprehensive visualization functions for simulation results.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_results_plots(df: pd.DataFrame) -> go.Figure:
    """
    Create comprehensive results visualization.

    Args:
        df: DataFrame with simulation results including columns:
            - Time_ms: Time in milliseconds
            - Impact_Force_MN: Impact force in MN
            - Penetration_mm: Penetration in mm
            - Acceleration_g: Acceleration in g
            - E_kin_J, E_mech_J, etc.: Energy columns (optional)
            - Backbone_Force_MN: Backbone force (optional)
            - E_num_ratio: Numerical residual ratio (optional)

    Returns:
        Plotly figure with 6 subplots
    """
    fig = make_subplots(
        rows=6,
        cols=1,
        subplot_titles=("Force", "Penetration", "Acceleration", "Hysteresis", "Energy", "Energy Balance Quality"),
        vertical_spacing=0.05,
    )

    # Force vs Time
    fig.add_trace(
        go.Scatter(
            x=df["Time_ms"],
            y=df["Impact_Force_MN"],
            line=dict(width=2, color="#1f77b4"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Force (MN)", row=1, col=1)

    # Penetration vs Time
    fig.add_trace(
        go.Scatter(
            x=df["Time_ms"],
            y=df["Penetration_mm"],
            line=dict(width=2, color="#ff7f0e"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Penetration (mm)", row=2, col=1)

    # Acceleration vs Time
    fig.add_trace(
        go.Scatter(
            x=df["Time_ms"],
            y=df["Acceleration_g"],
            line=dict(width=2, color="#2ca02c"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Acceleration (g)", row=3, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=3, col=1)

    # Hysteresis: line + backbone + colorbar
    fig.add_trace(
        go.Scatter(
            x=df["Penetration_mm"],
            y=df["Impact_Force_MN"],
            mode="lines",
            line=dict(width=2, color="#1f77b4"),
            name="Hysteresis",
            showlegend=False,
        ),
        row=4,
        col=1,
    )

    if "Backbone_Force_MN" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Penetration_mm"],
                y=df["Backbone_Force_MN"],
                mode="lines",
                line=dict(width=1.5, dash="dash", color="rgba(120,120,120,0.9)"),
                name="Contact backbone",
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    # Colorbar localized to hysteresis row
    y0, y1 = fig.layout.yaxis4.domain
    cb_y = 0.5 * (y0 + y1)
    cb_len = 0.9 * (y1 - y0)

    fig.add_trace(
        go.Scatter(
            x=df["Penetration_mm"],
            y=df["Impact_Force_MN"],
            mode="markers",
            marker=dict(
                size=0,
                color=df["Time_ms"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Time (ms)",
                    x=1.02,
                    y=cb_y,
                    len=cb_len,
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=4,
        col=1,
    )

    fig.update_xaxes(title_text="Penetration (mm)", row=4, col=1)
    fig.update_yaxes(title_text="Force (MN)", row=4, col=1)

    # Energy vs Time (Euler-Lagrange formulation)
    if "E_kin_J" in df.columns:
        # Kinetic energy T
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["E_kin_J"] / 1e6,
                line=dict(width=1.5, color="orange"),
                name="Kinetic (T)",
                showlegend=True,
            ),
            row=5,
            col=1,
        )
        # Potential energy V
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["E_pot_J"] / 1e6,
                line=dict(width=1.5, color="green"),
                name="Potential (V)",
                showlegend=True,
            ),
            row=5,
            col=1,
        )
        # Mechanical energy E = T + V
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["E_mech_J"] / 1e6,
                line=dict(width=2, color="blue", dash="dash"),
                name="Mechanical (T+V)",
                showlegend=True,
            ),
            row=5,
            col=1,
        )
        # External work
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["W_ext_J"] / 1e6,
                line=dict(width=1.5, color="cyan"),
                name="External work",
                showlegend=True,
            ),
            row=5,
            col=1,
        )
        # Total dissipation
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["E_diss_total_J"] / 1e6,
                line=dict(width=1.5, color="red"),
                name="Dissipated (total)",
                showlegend=True,
            ),
            row=5,
            col=1,
        )
        # Numerical residual (KEY!)
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["E_num_J"] / 1e6,
                line=dict(width=2, color="magenta", dash="dot"),
                name="Numerical residual",
                showlegend=True,
            ),
            row=5,
            col=1,
        )
        # Initial energy reference line
        if "E0_J" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Time_ms"],
                    y=df["E0_J"] / 1e6,
                    line=dict(width=1, color="gray", dash="dash"),
                    name="E₀ (initial)",
                    showlegend=True,
                ),
                row=5,
                col=1,
            )

        fig.update_yaxes(title_text="Energy (MJ)", row=5, col=1)
        fig.update_xaxes(title_text="Time (ms)", row=5, col=1)

    # Energy Balance Quality: Numerical Residual Ratio
    # NOTE: E_num_ratio is stored as a *ratio* (dimensionless). We plot it in percent.
    if "E_num_ratio" in df.columns:
        residual_ratio = df["E_num_ratio"].astype(float)
        residual_pct = 100.0 * residual_ratio

        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=residual_pct,
                line=dict(width=2, color="red"),
                name="|E_num| / E₀ (%)",
                showlegend=True,
                hovertemplate="t=%{x:.3f} ms<br>|E_num|/E₀=%{y:.6f}%<extra></extra>",
            ),
            row=6,
            col=1,
        )

        # Add 1% target reference line
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=[1.0] * len(df),
                line=dict(width=1, color="green", dash="dash"),
                name="1% target",
                showlegend=True,
                hovertemplate="t=%{x:.3f} ms<br>target=1.000000%<extra></extra>",
            ),
            row=6,
            col=1,
        )

        y_max = max(1.0, float(residual_pct.max()) * 1.10)
        fig.update_yaxes(
            title_text="Residual (% of E₀)",
            range=[0.0, y_max],
            tickformat=".3f",
            row=6,
            col=1,
        )
        fig.update_xaxes(title_text="Time (ms)", row=6, col=1)

    fig.update_layout(
        height=2000,  # Increased for 6th subplot
        showlegend=True,
        legend=dict(
            orientation="h",   # horizontal legend
            x=0.5,             # centered horizontally
            xanchor="center",
            y=-0.05,           # slightly below the last subplot
            yanchor="top",
        ),
        margin=dict(t=60, b=100, l=60, r=80),
    )
    return fig


def create_mass_kinematics_plots(df: pd.DataFrame, mass_index: int) -> go.Figure | None:
    """
    Create time history plots for mass acceleration, velocity, and position.

    Args:
        df: DataFrame with simulation results.
        mass_index: Zero-based mass index.

    Returns:
        Plotly figure with 3 subplots or None if data is missing.
    """
    time_col = "Time_ms" if "Time_ms" in df.columns else "Time_s" if "Time_s" in df.columns else None
    if time_col is None:
        return None

    idx = mass_index + 1
    position_cols = (f"Mass{idx}_Position_x_m", f"Mass{idx}_Position_y_m")
    velocity_cols = (f"Mass{idx}_Velocity_x_m_s", f"Mass{idx}_Velocity_y_m_s")
    acceleration_cols = (f"Mass{idx}_Acceleration_x_m_s2", f"Mass{idx}_Acceleration_y_m_s2")

    required_cols = (*position_cols, *velocity_cols, *acceleration_cols)
    if not all(col in df.columns for col in required_cols):
        return None

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Acceleration", "Velocity", "Position"),
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[acceleration_cols[0]],
            line=dict(width=2, color="#1f77b4"),
            name="Ax",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[acceleration_cols[1]],
            line=dict(width=2, color="#ff7f0e"),
            name="Ay",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[velocity_cols[0]],
            line=dict(width=2, color="#2ca02c"),
            name="Vx",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[velocity_cols[1]],
            line=dict(width=2, color="#d62728"),
            name="Vy",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[position_cols[0]],
            line=dict(width=2, color="#9467bd"),
            name="X",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[position_cols[1]],
            line=dict(width=2, color="#8c564b"),
            name="Y",
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Position (m)", row=3, col=1)
    fig.update_xaxes(title_text="Time (ms)" if time_col == "Time_ms" else "Time (s)", row=3, col=1)

    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1, yanchor="top"),
        margin=dict(t=60, b=80, l=60, r=60),
    )
    return fig


def create_spring_plots(df: pd.DataFrame, spring_index: int) -> go.Figure | None:
    """
    Create spring plots: hysteresis, displacement vs time, force vs time.

    Args:
        df: DataFrame with simulation results.
        spring_index: Zero-based spring index.

    Returns:
        Plotly figure with 3 subplots or None if data is missing.
    """
    time_col = "Time_ms" if "Time_ms" in df.columns else "Time_s" if "Time_s" in df.columns else None
    if time_col is None:
        return None

    idx = spring_index + 1
    disp_col = f"Spring{idx}_Disp_m"
    force_col = f"Spring{idx}_Force_N"
    if disp_col not in df.columns or force_col not in df.columns:
        return None

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Spring Hysteresis", "Displacement vs Time", "Force vs Time"),
        vertical_spacing=0.08,
    )

    force_mn = df[force_col] / 1e6

    fig.add_trace(
        go.Scatter(
            x=df[disp_col],
            y=force_mn,
            mode="lines",
            line=dict(width=2, color="#1f77b4"),
            name="Hysteresis",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    y0, y1 = fig.layout.yaxis.domain
    cb_y = 0.5 * (y0 + y1)
    cb_len = 0.9 * (y1 - y0)

    fig.add_trace(
        go.Scatter(
            x=df[disp_col],
            y=force_mn,
            mode="markers",
            marker=dict(
                size=0,
                color=df[time_col],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Time (ms)" if time_col == "Time_ms" else "Time (s)",
                    x=1.02,
                    y=cb_y,
                    len=cb_len,
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Displacement (m)", row=1, col=1)
    fig.update_yaxes(title_text="Force (MN)", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[disp_col],
            line=dict(width=2, color="#ff7f0e"),
            name="Displacement",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Displacement (m)", row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=force_mn,
            line=dict(width=2, color="#2ca02c"),
            name="Force",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Force (MN)", row=3, col=1)
    fig.update_xaxes(title_text="Time (ms)" if time_col == "Time_ms" else "Time (s)", row=3, col=1)

    fig.update_layout(
        height=900,
        showlegend=False,
        margin=dict(t=60, b=80, l=60, r=80),
    )
    return fig


def create_mass_force_displacement_plots(
    df: pd.DataFrame,
    mass_index: int,
    mode: str = "auto",  # "auto" | "total" | "internal" | "net" | "left" | "right"
) -> go.Figure | None:
    """Create force–displacement plots for a selected mass.

    Preferred source (if available):
      - Core-exported nodal forces: ``Mass{i}_Force_total_x_N`` (includes wall, friction, mass-contact)

    Fallback source:
      - Reconstruct from neighboring spring forces (chain model):
          F_net = F_left - F_right

    Args:
        df: Simulation results DataFrame.
        mass_index: Zero-based index of the mass.
        mode:
            - "auto": use core total force if available, else "net"
            - "total": force = core total x-force
            - "internal": force = core internal x-force (springs only)
            - "net": force = left spring - right spring
            - "left": force = left spring only
            - "right": force = - right spring only

    Returns:
        Plotly figure with 3 subplots (F–u, u(t), F(t)) or None if data is missing.
    """
    time_col = "Time_ms" if "Time_ms" in df.columns else ("Time_s" if "Time_s" in df.columns else None)
    if time_col is None:
        return None

    idx = mass_index + 1  # DataFrame uses 1-based naming
    pos_col = f"Mass{idx}_Position_x_m"
    if pos_col not in df.columns:
        return None

    import re

    # Infer number of masses from available position columns
    mass_ids = []
    for c in df.columns:
        mm = re.match(r"Mass(\d+)_Position_x_m$", c)
        if mm:
            mass_ids.append(int(mm.group(1)))
    n_masses = max(mass_ids) if mass_ids else int(df.attrs.get("n_masses", 0) or 0)
    if n_masses <= 0:
        return None

    u = df[pos_col] - float(df[pos_col].iloc[0])

    # -------------------------------
    # Select force source
    # -------------------------------
    col_total = f"Mass{idx}_Force_total_x_N"
    col_internal = f"Mass{idx}_Force_internal_x_N"

    F = None
    label = ""

    if mode in ("auto", "total") and col_total in df.columns:
        F = df[col_total]
        label = "Total nodal force in x (core)"
    elif mode == "internal" and col_internal in df.columns:
        F = df[col_internal]
        label = "Internal nodal force in x (core)"
    else:
        # Fallback: neighbor springs (1-based spring naming: Spring1 between Mass1-Mass2)
        left_force = None
        right_force = None

        if mass_index > 0:
            col_left = f"Spring{mass_index}_Force_N"
            if col_left in df.columns:
                left_force = df[col_left]

        if mass_index < n_masses - 1:
            col_right = f"Spring{mass_index + 1}_Force_N"
            if col_right in df.columns:
                right_force = df[col_right]

        if mode == "left":
            if left_force is None:
                return None
            F = left_force
            label = "Left spring force on mass"
        elif mode == "right":
            if right_force is None:
                return None
            F = -right_force
            label = "Right spring force on mass"
        else:
            # default fallback
            if mode not in ("auto", "net"):
                # unknown mode
                return None
            if left_force is None and right_force is None:
                return None
            F = (left_force if left_force is not None else 0.0) - (right_force if right_force is not None else 0.0)
            label = "Net internal force (left - right)"

    if F is None:
        return None

    F_mn = F / 1e6

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Mass Force–Displacement", "Displacement vs Time", "Force vs Time"),
        vertical_spacing=0.08,
    )

    # Hysteresis
    fig.add_trace(
        go.Scatter(
            x=u,
            y=F_mn,
            mode="lines",
            name=label,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Time colorbar (as invisible markers)
    y0, y1 = fig.layout.yaxis.domain
    cb_y = 0.5 * (y0 + y1)
    cb_len = 0.9 * (y1 - y0)

    fig.add_trace(
        go.Scatter(
            x=u,
            y=F_mn,
            mode="markers",
            marker=dict(
                size=0,
                color=df[time_col],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Time (ms)" if time_col == "Time_ms" else "Time (s)",
                    x=1.02,
                    y=cb_y,
                    len=cb_len,
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Displacement of mass (m)", row=1, col=1)
    fig.update_yaxes(title_text="Force on mass (MN)", row=1, col=1)

    # u(t)
    fig.add_trace(go.Scatter(x=df[time_col], y=u, showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="Displacement (m)", row=2, col=1)

    # F(t)
    fig.add_trace(go.Scatter(x=df[time_col], y=F_mn, showlegend=False), row=3, col=1)
    fig.update_yaxes(title_text="Force (MN)", row=3, col=1)
    fig.update_xaxes(title_text="Time (ms)" if time_col == "Time_ms" else "Time (s)", row=3, col=1)

    fig.update_layout(height=900, showlegend=False, margin=dict(t=60, b=80, l=60, r=80))
    return fig




def _infer_n_masses_from_df(df: pd.DataFrame) -> int:
    """Infer number of masses from exported column names."""
    import re

    mass_ids = []
    for c in df.columns:
        mm = re.match(r"Mass(\d+)_Position_x_m$", str(c))
        if mm:
            mass_ids.append(int(mm.group(1)))
    return int(max(mass_ids) if mass_ids else int(df.attrs.get("n_masses", 0) or 0))


def estimate_nodal_field_color_range(
    df: pd.DataFrame,
    *,
    quantity: str = "acceleration",
    component: str = "magnitude",
    log_color: bool = True,
    to_g: bool = True,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> tuple[float, float] | None:
    """Estimate robust (cmin, cmax) for nodal-field plots.

    The returned range is in the *same space as the color values*:
    - if log_color=True: log10(|field|)
    - else: field

    Uses quantiles to avoid a single spike dominating the scale.
    """
    import numpy as np

    n_masses = _infer_n_masses_from_df(df)
    if n_masses <= 0:
        return None

    # Select base column name
    if quantity == "acceleration":
        cx = "Acceleration_x_m_s2"
        cy = "Acceleration_y_m_s2"
    elif quantity == "velocity":
        cx = "Velocity_x_m_s"
        cy = "Velocity_y_m_s"
    elif quantity == "position":
        cx = "Position_x_m"
        cy = "Position_y_m"
    else:
        return None

    cols_x = [f"Mass{i}_{cx}" for i in range(1, n_masses + 1)]
    cols_y = [f"Mass{i}_{cy}" for i in range(1, n_masses + 1)]
    if not all(c in df.columns for c in cols_x):
        return None

    X = df[cols_x].to_numpy().T
    Y = df[cols_y].to_numpy().T if all(c in df.columns for c in cols_y) else None

    if component == "x":
        field = X
    elif component == "y":
        if Y is None:
            return None
        field = Y
    else:
        field = (X if Y is None else (X**2 + Y**2) ** 0.5)

    if quantity == "acceleration" and to_g:
        field = field / 9.80665

    eps = 1e-12
    if log_color:
        vals = np.log10(np.maximum(np.abs(field).ravel(), eps))
    else:
        vals = field.ravel()

    # Guard against NaNs
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    lo = float(np.quantile(vals, q_low))
    hi = float(np.quantile(vals, q_high))
    if lo == hi:
        # fallback: small symmetric pad
        hi = lo + 1e-9
    return (lo, hi)


def _downsample_time_indices(
    field: "np.ndarray",
    t_ms: "np.ndarray",
    *,
    max_time_points: int,
    mode: str = "uniform",  # "uniform" | "impact"
    impact_window_ms: float = 30.0,
) -> "np.ndarray":
    """Pick time indices with either uniform sampling or impact-focused sampling."""
    import numpy as np

    n_t = field.shape[1]
    if n_t <= max_time_points:
        return np.arange(n_t, dtype=int)

    if mode not in {"uniform", "impact"}:
        mode = "uniform"

    if mode == "uniform":
        return np.linspace(0, n_t - 1, max_time_points).astype(int)

    # Impact-focused: detect peak of envelope over nodes
    env = np.max(np.abs(field), axis=0)
    peak_i = int(np.argmax(env))
    t_peak = float(t_ms[peak_i])

    half = 0.5 * float(impact_window_ms)
    lo_t = t_peak - half
    hi_t = t_peak + half

    in_win = np.where((t_ms >= lo_t) & (t_ms <= hi_t))[0]
    out_win = np.where((t_ms < lo_t) | (t_ms > hi_t))[0]

    # Allocate more points to the impact window
    n_dense = int(max_time_points * 0.7)
    n_sparse = max_time_points - n_dense

    def pick(arr: "np.ndarray", k: int) -> "np.ndarray":
        if arr.size == 0 or k <= 0:
            return np.array([], dtype=int)
        if arr.size <= k:
            return arr.astype(int)
        return arr[np.linspace(0, arr.size - 1, k).astype(int)].astype(int)

    sel = np.unique(np.concatenate([pick(in_win, n_dense), pick(out_win, n_sparse)]))
    sel.sort()

    # Ensure we always include endpoints (nice for axes)
    sel = np.unique(np.concatenate([[0, n_t - 1], sel]))
    sel.sort()

    # If we went slightly over budget, thin uniformly
    if sel.size > max_time_points:
        sel = sel[np.linspace(0, sel.size - 1, max_time_points).astype(int)]
    return sel.astype(int)


def export_nodal_field_heatmap_bytes(
    df: pd.DataFrame,
    *,
    quantity: str = "acceleration",
    component: str = "magnitude",
    log_color: bool = True,
    to_g: bool = True,
    time_unit: str = "ms",  # "ms"|"s"
    add_contours: bool = True,
    cmin: float | None = None,
    cmax: float | None = None,
    max_time_points: int = 2000,
    max_nodes: int = 200,
    downsample_mode: str = "impact",
    impact_window_ms: float = 30.0,
    fmt: str = "png",  # "png"|"svg"
    dpi: int = 300,
) -> bytes | None:
    """Export the nodal field heatmap (paper-style) as PNG/SVG bytes.

    Uses Matplotlib (no kaleido dependency).
    """
    import io
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = _compute_nodal_field_matrix(
        df,
        quantity=quantity,
        component=component,
        log_color=log_color,
        to_g=to_g,
        time_unit=time_unit,
        max_time_points=max_time_points,
        max_nodes=max_nodes,
        downsample_mode=downsample_mode,
        impact_window_ms=impact_window_ms,
        cmin=cmin,
        cmax=cmax,
        for_export=True,
    )
    if data is None:
        return None

    t, nodes, z, meta = data

    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)

    extent = [float(t[0]), float(t[-1]), float(nodes[0]), float(nodes[-1])]
    im = ax.imshow(
        z,
        origin='lower',
        aspect='auto',
        extent=extent,
        vmin=meta.get('cmin', None),
        vmax=meta.get('cmax', None),
    )

    if add_contours:
        # Reasonable number of levels; fall back if range is tiny
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        if np.isfinite(zmin) and np.isfinite(zmax) and (zmax - zmin) > 1e-12:
            levels = np.linspace(zmin, zmax, 18)
            # Build grid for contour
            tt = np.linspace(float(t[0]), float(t[-1]), z.shape[1])
            nn = np.linspace(float(nodes[0]), float(nodes[-1]), z.shape[0])
            TT, NN = np.meshgrid(tt, nn)
            ax.contour(TT, NN, z, levels=levels, linewidths=0.35, alpha=0.55)

    ax.set_xlabel(meta['x_label'])
    ax.set_ylabel('Node number')

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(meta['cbar_title'])

    buf = io.BytesIO()
    fmt = fmt.lower().strip()
    if fmt == 'svg':
        fig.savefig(buf, format='svg')
    else:
        fig.savefig(buf, format='png', dpi=int(dpi))
    plt.close(fig)

    return buf.getvalue()


def _compute_nodal_field_matrix(
    df: pd.DataFrame,
    *,
    quantity: str,
    component: str,
    plot_type: str | None = None,
    log_color: bool,
    to_g: bool,
    time_unit: str,
    max_time_points: int,
    max_nodes: int,
    downsample_mode: str,
    impact_window_ms: float,
    cmin: float | None,
    cmax: float | None,
    for_export: bool = False,
):
    """Shared computation: returns (t_axis, nodes, color_matrix, meta)."""
    import numpy as np

    time_col = "Time_s" if "Time_s" in df.columns else ("Time_ms" if "Time_ms" in df.columns else None)
    if time_col is None:
        return None

    n_masses = _infer_n_masses_from_df(df)
    if n_masses <= 0:
        return None

    # Select base column name
    if quantity == "acceleration":
        cx = "Acceleration_x_m_s2"
        cy = "Acceleration_y_m_s2"
        unit = "g" if to_g else "m/s²"
    elif quantity == "velocity":
        cx = "Velocity_x_m_s"
        cy = "Velocity_y_m_s"
        unit = "m/s"
    elif quantity == "position":
        cx = "Position_x_m"
        cy = "Position_y_m"
        unit = "m"
    else:
        return None

    cols_x = [f"Mass{i}_{cx}" for i in range(1, n_masses + 1)]
    cols_y = [f"Mass{i}_{cy}" for i in range(1, n_masses + 1)]
    if not all(c in df.columns for c in cols_x):
        return None

    X = df[cols_x].to_numpy().T
    Y = df[cols_y].to_numpy().T if all(c in df.columns for c in cols_y) else None

    if component == "x":
        field = X
    elif component == "y":
        if Y is None:
            return None
        field = Y
    else:
        field = (X if Y is None else (X**2 + Y**2) ** 0.5)

    if quantity == "acceleration" and to_g:
        field = field / 9.80665

    # Time arrays
    if time_col == "Time_ms":
        t_ms = df[time_col].to_numpy().astype(float)
        t_s = t_ms / 1000.0
    else:
        t_s = df[time_col].to_numpy().astype(float)
        t_ms = t_s * 1000.0

    # Node downsample (uniform)
    n_nodes, n_t = field.shape

    def pick_nodes(n: int, nmax: int):
        if n <= nmax:
            return np.arange(n, dtype=int)
        return np.linspace(0, n - 1, nmax).astype(int)

    ni = pick_nodes(n_nodes, int(max_nodes))

    # Time downsample
    ti = _downsample_time_indices(
        field,
        t_ms,
        max_time_points=int(max_time_points),
        mode=downsample_mode,
        impact_window_ms=float(impact_window_ms),
    )

    field_ds = field[np.ix_(ni, ti)]

    # Axis selection
    time_unit = (time_unit or "ms").lower().strip()
    if time_unit not in {"ms", "s"}:
        time_unit = "ms"

    if time_unit == "ms":
        t_axis = t_ms[ti]
        x_label = "Time (ms)"
    else:
        t_axis = t_s[ti]
        x_label = "Time (s)"

    nodes = (ni + 1).astype(int)

    eps = 1e-12
    if log_color:
        z = np.log10(np.maximum(np.abs(field_ds), eps))
        cbar_title = f"log10(|a| [{unit}])" if quantity == "acceleration" else f"log10(|{quantity}| [{unit}])"
    else:
        z = field_ds
        cbar_title = f"a [{unit}]" if quantity == "acceleration" else f"{quantity} [{unit}]"

    meta = {
        "unit": unit,
        "x_label": x_label,
        "cbar_title": cbar_title,
        "cmin": cmin,
        "cmax": cmax,
    }

    return t_axis, nodes, z, meta


def create_nodal_field_surface(
    df: pd.DataFrame,
    quantity: str = "acceleration",  # "acceleration"|"velocity"|"position"
    component: str = "magnitude",    # "magnitude"|"x"|"y"
    plot_type: str = "surface",      # "surface"|"heatmap"
    log_color: bool = True,
    to_g: bool = True,
    time_unit: str = "ms",           # "ms"|"s"
    add_contours: bool = True,
    cmin: float | None = None,
    cmax: float | None = None,
    downsample_mode: str = "uniform",   # "uniform"|"impact"
    impact_window_ms: float = 30.0,
    max_time_points: int = 800,
    max_nodes: int = 40,
) -> go.Figure | None:
    """Create a node-vs-time field plot.

    - Paper default (recommended): plot_type="heatmap", add_contours=True, log_color=True, to_g=True.
    - Keeps the 3D surface option for "wow".

    Notes
    -----
    * Color scaling can be fixed across runs using (cmin, cmax). If log_color=True,
      these should be provided in log10 space.
    * Time axis can be shown in ms or s via time_unit.
    * downsample_mode="impact" concentrates time samples around the global impact peak.
    """
    data = _compute_nodal_field_matrix(
        df,
        quantity=quantity,
        component=component,
        plot_type=plot_type,
        log_color=log_color,
        to_g=to_g,
        time_unit=time_unit,
        max_time_points=int(max_time_points),
        max_nodes=int(max_nodes),
        downsample_mode=downsample_mode,
        impact_window_ms=float(impact_window_ms),
        cmin=cmin,
        cmax=cmax,
        for_export=False,
    )
    if data is None:
        return None

    import numpy as np

    t_axis, nodes, z, meta = data

    plot_type = (plot_type or "heatmap").lower().strip()
    if plot_type not in {"surface", "heatmap"}:
        plot_type = "heatmap"

    # If cmin/cmax not provided, keep Plotly autoscale
    zmin = meta.get("cmin", None)
    zmax = meta.get("cmax", None)

    if plot_type == "heatmap":
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=t_axis,
                y=nodes,
                z=z,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title=meta["cbar_title"]),
            )
        )

        if add_contours:
            # Overlay fine contour lines for readability
            z_flat = z[np.isfinite(z)]
            if z_flat.size > 0:
                zz_min = float(np.min(z_flat))
                zz_max = float(np.max(z_flat))
                if (zz_max - zz_min) > 1e-12:
                    n_levels = 18
                    step = (zz_max - zz_min) / n_levels
                    fig.add_trace(
                        go.Contour(
                            x=t_axis,
                            y=nodes,
                            z=z,
                            showscale=False,
                            contours=dict(
                                coloring="none",
                                showlines=True,
                                start=zz_min,
                                end=zz_max,
                                size=step,
                            ),
                            line=dict(width=0.6),
                            hoverinfo="skip",
                        )
                    )

        fig.update_layout(
            height=560,
            margin=dict(l=50, r=30, t=40, b=45),
            xaxis_title=meta["x_label"],
            yaxis_title="Node number",
        )
        return fig


    # NOTE: For surface, use |field| as height, and color as z (already log or linear).
    # We rebuild a consistent surface trace here.
    # To keep memory low, reuse z for coloring and derive height from original scale:
    # If log_color=True: use 10**z for height; else |z|.
    if log_color:
        height = 10 ** z
    else:
        height = np.abs(z)

    fig = go.Figure(
        data=go.Surface(
            x=t_axis,
            y=nodes,
            z=height,
            surfacecolor=z,
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(title=meta["cbar_title"]),
        )
    )

    ztitle = f"|{quantity}| ({meta['unit']})"
    fig.update_layout(
        height=640,
        margin=dict(l=40, r=20, t=40, b=40),
        scene=dict(
            xaxis_title=meta["x_label"],
            yaxis_title="Node number",
            zaxis_title=ztitle,
        ),
    )
    return fig
