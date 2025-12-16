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

    Returns:
        Plotly figure with 5 subplots
    """
    fig = make_subplots(
        rows=5,
        cols=1,
        subplot_titles=("Force", "Penetration", "Acceleration", "Hysteresis", "Energy"),
        vertical_spacing=0.06,
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

    # Energy vs Time
    if "E_kin_J" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["E_kin_J"] / 1e6,
                line=dict(width=1.5),
                name="Kinetic",
                showlegend=True,
            ),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["E_mech_J"] / 1e6,
                line=dict(width=1.5, dash="dash"),
                name="Mech (T+V)",
                showlegend=True,
            ),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["E_total_tracked_J"] / 1e6,
                line=dict(width=1.5),
                name="Total tracked",
                showlegend=True,
            ),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Time_ms"],
                y=df["E_diss_tracked_J"] / 1e6,
                line=dict(width=1.5),
                name="Dissipated (all mechanisms)",
                showlegend=True,
            ),
            row=5,
            col=1,
        )

        fig.update_yaxes(title_text="Energy (MJ)", row=5, col=1)
        fig.update_xaxes(title_text="Time (ms)", row=5, col=1)

    fig.update_layout(
        height=1700,
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
