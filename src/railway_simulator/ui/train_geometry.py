"""
Train geometry visualization for the Railway Impact Simulator UI.

Provides functions to plot lumped mass distributions and cumulative mass.
"""

from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_train_geometry_plot(params: Dict[str, Any]) -> go.Figure:
    """
    Plot lumped masses and cumulative (Riera-type) mass distribution.

    Args:
        params: Dictionary containing:
            - masses: Array of mass values
            - x_init: Array of initial x positions

    Returns:
        Plotly figure with 2 subplots showing mass distribution and cumulative mass
    """
    masses = np.asarray(params.get("masses", []), dtype=float)
    x = np.asarray(params.get("x_init", []), dtype=float)

    if masses.size == 0 or x.size == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No train data available",
            xaxis_title="x (m)",
            yaxis_title="Mass (t)",
        )
        return fig

    order = np.argsort(x)
    x_sorted = x[order]
    m_sorted_t = masses[order] / 1000.0  # t
    cum_mass_t = np.cumsum(m_sorted_t)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Lumped mass distribution", "Cumulative mass M(x)"),
    )

    fig.add_trace(
        go.Scatter(
            x=x_sorted,
            y=m_sorted_t,
            mode="lines+markers",
            line=dict(width=2, shape="hv"),
            marker=dict(size=8),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Mass per node (t)", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=x_sorted,
            y=cum_mass_t,
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=6),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Cumulative mass (t)", row=2, col=1)
    fig.update_xaxes(title_text="Longitudinal position x (m)", row=2, col=1)

    fig.update_layout(height=800, showlegend=False)
    return fig
