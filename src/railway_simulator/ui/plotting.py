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
    if "E_num_ratio" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["E_num_ratio"] * 100,  # Convert to percentage
                line=dict(width=2, color="red"),
                name="|E_num| / E₀",
                showlegend=True,
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
            ),
            row=6,
            col=1,
        )
        fig.update_yaxes(title_text="Residual (%)", row=6, col=1)
        fig.update_xaxes(title_text="Time (ms)", row=6, col=1)

    fig.update_layout(
        height=2000,  # Increased for 6th subplot
        showlegend=True,
        legend=dict(
            orientation="h",   # leyenda horizontal
            x=0.5,             # centrada horizontalmente
            xanchor="center",
            y=-0.05,           # un poco debajo del último subplot
            yanchor="top",
        ),
        margin=dict(t=60, b=100, l=60, r=80),
    )
    return fig
