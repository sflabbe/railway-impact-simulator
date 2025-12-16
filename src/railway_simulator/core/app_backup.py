"""
Streamlit UI for the Railway Impact Simulator

This file provides the full interactive user interface and plotting,
and delegates all heavy numerical work to core.engine.run_simulation().
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.constants import g as GRAVITY

from railway_simulator.core.engine import TrainBuilder, TrainConfig, run_simulation
from railway_simulator.core.parametric import (
    build_speed_scenarios,
    make_envelope_figure,
    run_parametric_envelope,
)

from railway_simulator.studies import parse_floats_csv
from railway_simulator.studies.numerics_sensitivity import run_numerics_sensitivity
from railway_simulator.studies.strain_rate_sensitivity import run_fixed_dif_sensitivity

# ====================================================================
# HEADER / ABOUT
# ====================================================================

def display_header():
    """Display header with institutional logos and research information."""

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### KIT")
        st.markdown("**Karlsruher Institut für Technologie**")

    with col2:
        st.markdown("### EBA")
        st.markdown("**Eisenbahn-Bundesamt**")

    with col3:
        st.markdown("### DZSF")
        st.markdown("**Deutsches Zentrum für Schienenverkehrsforschung**")

    st.markdown("---")

    # Research information
    st.markdown(
        """
    ### Research Background

    **Report Title (German):**  
    *Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr*

    **Report Title (English):**  
    *Review and Adjustment of Impact Loads from Railway Traffic*

    **Authors:**
    - Lothar Stempniewski (KIT)
    - Sebastián Labbé (KIT)
    - Steffen Siegel (Siegel und Wünschel PartG mbB)
    - Robin Bosch (Siegel und Wünschel PartG mbB)

    **Research Institutions:**
    - Karlsruher Institut für Technologie (KIT)  
      Institut für Massivbau und Baustofftechnologie

    **Publication:**  
    DZSF Bericht 53 (2024)  
    Project Number: 2018-08-U-1217  
    Study Completion: June 2021  
    Publication Date: June 2024

    **DOI:** [10.48755/dzsf.240006.01](https://doi.org/10.48755/dzsf.240006.01)  
    **ISSN:** 2629-7973  
    **License:** CC BY 4.0

    **Download Report:**  
    [DZSF Forschungsbericht 53/2024 (PDF)](https://www.dzsf.bund.de/SharedDocs/Downloads/DZSF/Veroeffentlichungen/Forschungsberichte/2024/ForBe_53_2024_Anpralllasten.pdf?__blob=publicationFile&v=2)

    **Commissioned by:**  
    Eisenbahn-Bundesamt (EBA)

    **Published by:**  
    Deutsches Zentrum für Schienenverkehrsforschung (DZSF)
    """
    )

    st.markdown("---")

def display_citation():
    """Show how to cite the underlying research report."""
    st.markdown("---")
    st.markdown(
        """
    ### Citation

    If you use this simulator in your research, please cite the original research report:

    **Plain Text:**
    ```
    Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024).
    Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr.
    Berichte des Deutschen Zentrums für Schienenverkehrsforschung, Bericht 53.
    Deutsches Zentrum für Schienenverkehrsforschung beim Eisenbahn-Bundesamt.
    https://doi.org/10.48755/dzsf.240006.01
    ```

    **BibTeX:**
    ```bibtex
    @techreport{Stempniewski2024Anpralllasten,
      author       = {Stempniewski, Lothar and
                      Labbé, Sebastián and
                      Siegel, Steffen and
                      Bosch, Robin},
      title        = {Überprüfung und Anpassung der Anpralllasten
                      aus dem Eisenbahnverkehr},
      institution  = {Deutsches Zentrum für Schienenverkehrsforschung
                      beim Eisenbahn-Bundesamt},
      year         = {2024},
      type         = {Bericht},
      number       = {53},
      address      = {Dresden, Germany},
      note         = {Projektnummer 2018-08-U-1217,
                      Commissioned by Eisenbahn-Bundesamt},
      doi          = {10.48755/dzsf.240006.01},
      issn         = {2629-7973},
      url          = {https://www.dzsf.bund.de/SharedDocs/Downloads/DZSF/Veroeffentlichungen/Forschungsberichte/2024/ForBe_53_2024_Anpralllasten.pdf}
    }
    ```

    **APA 7th Edition:**
    ```
    Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024).
    Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr
    (DZSF Bericht No. 53). Deutsches Zentrum für Schienenverkehrsforschung
    beim Eisenbahn-Bundesamt. https://doi.org/10.48755/dzsf.240006.01
    ```

    ---
    **License:** This work is licensed under CC BY 4.0
    """
    )

# ====================================================================
# UTILITIES (EXPORT + PLOTS)
# ====================================================================

def to_excel(df: pd.DataFrame) -> bytes:
    """Generate Excel file for download."""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Dynamic Load History")
    except ImportError:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Dynamic Load History")
    return output.getvalue()


def create_results_plots(df: pd.DataFrame) -> go.Figure:
    """Create comprehensive results visualization."""
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
                name="Dissipated (Rayleigh+fric)",
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

# ====================================================================
# BUILDING SDOF + RESPONSE SPECTRA
# ====================================================================

def _solve_sdof_newmark_force(
    t: np.ndarray,
    F: np.ndarray,
    m: float,
    k: float,
    zeta: float,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve m*u¨ + c*u˙ + k*u = F(t) with Newmark-β (average acceleration).
    """
    t = np.asarray(t, dtype=float)
    F = np.asarray(F, dtype=float)
    n = len(t)

    if n < 2 or F.size != n or m <= 0.0 or k <= 0.0:
        return np.zeros(n), np.zeros(n), np.zeros(n)

    dt_array = np.diff(t)
    if not np.all(dt_array > 0.0):
        return np.zeros(n), np.zeros(n), np.zeros(n)
    dt = float(dt_array.mean())

    omega_n = np.sqrt(k / m)
    c = 2.0 * zeta * m * omega_n

    # Newmark coefficients
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    k_eff = k + a0 * m + a1 * c

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # Initial acceleration from equilibrium
    a[0] = (F[0] - c * v[0] - k * u[0]) / m

    for i in range(n - 1):
        P_eff = (
            F[i + 1]
            + m * (a0 * u[i] + a2 * v[i] + a3 * a[i])
            + c * (a1 * u[i] + a4 * v[i] + a5 * a[i])
        )

        u_next = P_eff / k_eff
        a_next = a0 * (u_next - u[i]) - a2 * v[i] - a3 * a[i]
        v_next = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a_next)

        u[i + 1] = u_next
        v[i + 1] = v_next
        a[i + 1] = a_next

    return u, v, a


def _run_sdof_linear_newmark(
    t: np.ndarray,
    F: np.ndarray,
    m: float,
    k: float,
    zeta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear SDOF: m*u¨ + c*u˙ + k*u = F(t)."""
    return _solve_sdof_newmark_force(t, F, m, k, zeta)


class _TakedaState:
    """
    Very simple Takeda-type bilinear hysteresis with pinching.
    Used only in post-processing (no feedback into dynamics).
    """

    __slots__ = (
        "k0",
        "uy",
        "alpha",
        "gamma",
        "k_post",
        "u_prev",
        "F_prev",
        "dir_prev",
        "u_max_pos",
        "u_max_neg",
        "u_rev_pos",
        "F_rev_pos",
        "u_pinched_pos",
        "u_rev_neg",
        "F_rev_neg",
        "u_pinched_neg",
    )

    def __init__(self, k0: float, uy: float, alpha: float, gamma: float):
        self.k0 = float(k0)
        self.uy = max(abs(uy), 1e-9)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.k_post = self.alpha * self.k0

        self.u_prev = 0.0
        self.F_prev = 0.0
        self.dir_prev = 0  # -1, 0, +1

        # symmetric yield limits
        self.u_max_pos = self.uy
        self.u_max_neg = -self.uy

        self.u_rev_pos = 0.0
        self.F_rev_pos = 0.0
        self.u_pinched_pos = self.gamma * self.u_max_pos

        self.u_rev_neg = 0.0
        self.F_rev_neg = 0.0
        self.u_pinched_neg = self.gamma * self.u_max_neg

    def _envelope_abs(self, u_abs: float) -> float:
        """Bilinear envelope in terms of |u|."""
        if u_abs <= self.uy:
            return self.k0 * u_abs
        else:
            return self.k0 * self.uy + self.k_post * (u_abs - self.uy)

    def update(self, u: float) -> float:
        eps = 1e-12
        du = u - self.u_prev
        if abs(du) < eps:
            return self.F_prev

        dir_now = 1 if du > 0.0 else -1
        if self.dir_prev == 0:
            self.dir_prev = dir_now

        sign_u = 1 if u >= 0.0 else -1

        # Positive side
        if sign_u >= 0:
            if u > self.u_max_pos:
                self.u_max_pos = u

            if dir_now >= 0:
                # Loading towards + : envelope
                F_env = self._envelope_abs(abs(u))
                F_new = F_env
            else:
                # Unloading towards 0/negative
                if self.dir_prev > 0:
                    self.u_rev_pos = self.u_prev
                    self.F_rev_pos = self.F_prev
                    u_max_eff = max(self.u_max_pos, self.uy)
                    self.u_pinched_pos = self.gamma * u_max_eff

                u1 = self.u_rev_pos
                F1 = self.F_rev_pos
                u0 = self.u_pinched_pos

                if abs(u1 - u0) < eps:
                    k_unload = -self.k0
                else:
                    k_unload = (0.0 - F1) / (u0 - u1)

                F_new = F1 + k_unload * (u - u1)

        # Negative side
        else:
            if u < self.u_max_neg:
                self.u_max_neg = u

            if dir_now <= 0:
                F_env_abs = self._envelope_abs(abs(u))
                F_new = -F_env_abs
            else:
                if self.dir_prev < 0:
                    self.u_rev_neg = self.u_prev
                    self.F_rev_neg = self.F_prev
                    u_min_eff = min(self.u_max_neg, -self.uy)
                    self.u_pinched_neg = self.gamma * u_min_eff

                u1 = self.u_rev_neg
                F1 = self.F_rev_neg
                u0 = self.u_pinched_neg

                if abs(u1 - u0) < eps:
                    k_unload = self.k0
                else:
                    k_unload = (0.0 - F1) / (u0 - u1)

                F_new = F1 + k_unload * (u - u1)

        if abs(u) < 1e-6 and abs(F_new) < 1e-3 * self.k0 * self.uy:
            F_new = 0.0

        self.u_prev = u
        self.F_prev = F_new
        self.dir_prev = dir_now
        return F_new


def _compute_takeda_force_history(
    u: np.ndarray,
    k0: float,
    uy: float,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    """Generate a Takeda-type degrading hysteresis F(u) from displacement history u(t)."""
    state = _TakedaState(k0, uy, alpha, gamma)
    F = np.zeros_like(u, dtype=float)
    for i in range(len(u)):
        F[i] = state.update(float(u[i]))
    return F


def compute_building_sdof_response(
    df: pd.DataFrame,
    k_wall: float,
    m_build: float,
    zeta: float,
    model: str = "Linear elastic SDOF",
    uy_mm: float = 10.0,
    alpha: float = 0.05,
    gamma: float = 0.4,
) -> pd.DataFrame:
    """
    Compute SDOF building response at the impact location.

    - Linear dynamics (Newmark-β) with F_contact(t)
    - Restoring force:
      * Linear: F_h = k_wall * u
      * Takeda: F_h = Takeda(u, α, γ) (post-processing)
    """
    t = df["Time_s"].to_numpy()
    F = df["Impact_Force_MN"].to_numpy() * 1e6  # [N]

    n = len(t)
    if n < 2 or np.allclose(F, 0.0) or k_wall <= 0.0 or m_build <= 0.0:
        return pd.DataFrame()

    dt = t[1] - t[0]
    if dt <= 0.0:
        return pd.DataFrame()

    m = float(m_build)
    k = float(k_wall)
    omega_n = np.sqrt(k / m)
    f_n = omega_n / (2.0 * np.pi)
    T_n = 2.0 * np.pi / omega_n
    c_crit = 2.0 * m * omega_n
    c = 2.0 * zeta * omega_n * m

    T_total = t[-1] - t[0]
    n_cycles_total = T_total / T_n if T_n > 0.0 else np.inf
    if n_cycles_total < 5.0:
        try:
            st.info(
                f"Building SDOF: current simulation covers only "
                f"{n_cycles_total:.1f} natural periods (Tₙ = {T_n:.2f} s). "
                f"Consider increasing 'Max Simulation Time' to at least "
                f"5–10 × Tₙ for a full decay of the response."
            )
        except Exception:
            pass

    u, v, a = _run_sdof_linear_newmark(t, F, m, k, zeta)
    if len(u) == 0:
        return pd.DataFrame()

    uy = uy_mm / 1000.0
    if model == "Takeda degrading hysteresis":
        Fh = _compute_takeda_force_history(u, k0=k, uy=uy, alpha=alpha, gamma=gamma)
    else:
        Fh = k * u

    n_out = len(u)
    out = pd.DataFrame(
        {
            "Building_u_mm": u * 1000.0,
            "Building_v_m_s": v,
            "Building_a_g": a / GRAVITY,
            "Building_restoring_force_MN": Fh / 1e6,
            "Building_omega_n_rad_s": np.full(n_out, omega_n, dtype=float),
            "Building_f_n_Hz": np.full(n_out, f_n, dtype=float),
            "Building_T_n_s": np.full(n_out, T_n, dtype=float),
            "Building_zeta": np.full(n_out, zeta, dtype=float),
            "Building_c_crit_kNs_m": np.full(n_out, c_crit / 1000.0, dtype=float),
            "Building_c_kNs_m": np.full(n_out, c / 1000.0, dtype=float),
        }
    )
    return out


def create_building_response_plots(df: pd.DataFrame) -> go.Figure:
    """Create plots for SDOF building displacement, velocity and acceleration."""
    if "Building_a_g" not in df.columns:
        raise ValueError("Building response columns not found in DataFrame.")

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Building displacement",
            "Building velocity",
            "Building acceleration",
        ),
        vertical_spacing=0.08,
    )

    # Displacement
    fig.add_trace(
        go.Scatter(
            x=df["Time_ms"],
            y=df["Building_u_mm"],
            line=dict(width=2, color="#1f77b4"),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="u (mm)", row=1, col=1)

    # Velocity
    fig.add_trace(
        go.Scatter(
            x=df["Time_ms"],
            y=df["Building_v_m_s"],
            line=dict(width=2, color="#ff7f0e"),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="v (m/s)", row=2, col=1)

    # Acceleration
    fig.add_trace(
        go.Scatter(
            x=df["Time_ms"],
            y=df["Building_a_g"],
            line=dict(width=2, color="#2ca02c"),
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="a (g)", row=3, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=3, col=1)

    fig.update_layout(height=900, showlegend=False)
    return fig


def create_building_hysteresis_plot(df: pd.DataFrame) -> go.Figure:
    """
    Plot building restoring force vs. top displacement (hysteresis),
    with time shown as a color scale.
    """
    required_cols = [
        "Building_u_mm",
        "Building_restoring_force_MN",
        "Time_ms",
    ]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            "Building hysteresis plot requires columns "
            "'Building_u_mm', 'Building_restoring_force_MN' and 'Time_ms'."
        )

    u = df["Building_u_mm"]
    F = df["Building_restoring_force_MN"]
    t_ms = df["Time_ms"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=u,
            y=F,
            mode="lines",
            line=dict(width=2, color="#1f77b4"),
            showlegend=False,
            name="Building hysteresis",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=u,
            y=F,
            mode="markers",
            marker=dict(
                size=0,
                color=t_ms,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Time (ms)",
                    x=1.02,
                    y=0.5,
                    len=0.9,
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_xaxes(title_text="Top displacement u (mm)")
    fig.update_yaxes(title_text="Restoring force (MN)")
    fig.update_layout(
        title="Building hysteresis loop",
        height=500,
        margin=dict(l=60, r=80, b=60, t=40),
    )
    return fig

def _cantilever_shape_with_point_load(y: np.ndarray, L: float, a: float) -> np.ndarray:
    """
    Normalized deflection shape φ(y) for a cantilever of length L
    with a point load at height a (measured from the fixed base).

    φ(L) = 1, φ(0) = 0.

    For 0 <= y <= a:
        φ(y) = y^2 (3a - y) / (a^2 (3L - a))

    For a <= y <= L:
        φ(y) = (-a + 3y) / (3L - a)
    """
    y = np.asarray(y, dtype=float)

    # Evitar a = 0 o a = L exacto
    eps = 1e-6
    a = float(np.clip(a, eps, L - eps))

    phi = np.zeros_like(y, dtype=float)

    mask_lower = y <= a
    yl = y[mask_lower]
    phi[mask_lower] = yl**2 * (3.0 * a - yl) / (a**2 * (3.0 * L - a))

    mask_upper = y > a
    yu = y[mask_upper]
    phi[mask_upper] = (-a + 3.0 * yu) / (3.0 * L - a)

    return phi


def create_building_animation(
    df: pd.DataFrame,
    height_m: float,
    scale_factor: float = 1.0,
    impact_height_m: float | None = None,
    speed_factor: float = 1.0,
    n_shape_points: int = 50,
    use_beam_element: bool = False,
):
    """
    Plotly animation of a cantilever-like building deformation over time.

    Parameters
    ----------
    df : DataFrame
        Must contain columns 'Building_u_mm' and 'Time_ms'.
        'Building_u_mm' is the top displacement in mm.
    height_m : float
        Total height of the building/cantilever [m].
    scale_factor : float, optional
        Visual scale for horizontal displacement.
    impact_height_m : float, optional
        Height of impact / point load [m] from the base.
        If None, a standard train impact height is used (~1/3–1/2 of h_train=3 m).
    speed_factor : float, optional
        Playback speed factor; >1.0 = faster, <1.0 = slower.
    n_shape_points : int, optional
        Number of points along the height for plotting the deformed shape.
    use_beam_element : bool, optional
        Reserved for a future “true” beam element time integration.
        If False (default), a static cantilever shape is used (fast).
    """
    if (
        "Building_u_mm" not in df.columns
        or "Time_ms" not in df.columns
        or height_m <= 0.0
    ):
        return None

    # Top displacement in meters
    u = df["Building_u_mm"].to_numpy() / 1000.0
    u_scaled = scale_factor * u
    t_ms = df["Time_ms"].to_numpy()

    n = len(u_scaled)
    if n == 0:
        return None

    max_frames = 200
    stride = max(1, n // max_frames)

    # --- Impact height default (1/3–1/2 of train height, h_train≈3 m) ---
    if impact_height_m is None:
        h_train = 3.0  # m
        h_imp_std = h_train * 5.0 / 12.0  # promedio 1/3–1/2 ≈ 1.25 m
        impact_height_m = min(h_imp_std, height_m - 0.1)

    # Clamp por seguridad
    impact_height_m = float(np.clip(impact_height_m, 0.1, height_m - 1e-3))

    # Vertical coordinates
    y_coords = np.linspace(0.0, height_m, n_shape_points)

    # ---- MODO RÁPIDO: forma estática de voladizo escalada con u_top(t) ----
    # (Aquí es donde luego puedes enganchar el “beam element” dinámico si quieres.)
    phi = _cantilever_shape_with_point_load(
        y_coords, L=height_m, a=impact_height_m
    )

    # Para la escala del eje x
    max_disp = float(np.max(np.abs(u_scaled))) if np.any(u_scaled) else 0.05
    x_lim = max(0.05, 1.2 * max_disp)

    frames = []
    for idx in range(0, n, stride):
        if not use_beam_element:
            # Rápido: forma estática φ(y) * u_top(t)
            x_frame = phi * u_scaled[idx]
        else:
            # FUTURO: aquí podrías meter la solución de un beam element dinámico
            # w(y, t_idx) calculado con M, C, K y F(t). Por ahora, usamos lo mismo.
            x_frame = phi * u_scaled[idx]

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x_frame,
                        y=y_coords,
                        mode="lines+markers",
                        line=dict(width=3),
                        marker=dict(size=6),
                    )
                ],
                name=f"{t_ms[idx]:.1f} ms",
            )
        )

    # Initial frame
    x0 = phi * u_scaled[0]

    # Playback speed
    base_duration_ms = 40
    sf = max(speed_factor, 0.1)
    frame_duration = int(base_duration_ms / sf)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x0,
                y=y_coords,
                mode="lines+markers",
                line=dict(width=3),
                marker=dict(size=6),
            )
        ],
        layout=go.Layout(
            xaxis=dict(
                title="Horizontal displacement (m)",
                range=[-x_lim, x_lim],
                zeroline=True,
            ),
            yaxis=dict(
                title="Height (m)",
                range=[0.0, height_m * 1.1],
                scaleanchor="x",
                scaleratio=1.0,
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": frame_duration,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
            margin=dict(l=60, r=40, b=60, t=40),
        ),
        frames=frames,
    )
    return fig

def compute_force_response_spectrum(
    df: pd.DataFrame,
    zeta: float,
    m_eff: float,
    f_min: float = 0.1,
    f_max: float = 100.0,
    n_freq: int = 80,
    use_logspace: bool = True,
) -> pd.DataFrame:
    """
    Force-based response spectrum: pseudo-acceleration Sa(f)
    for a family of linear SDOF oscillators excited by F_contact(t).
    """
    if (
        "Time_s" not in df.columns
        or "Impact_Force_MN" not in df.columns
        or m_eff <= 0.0
    ):
        return pd.DataFrame()

    t = df["Time_s"].to_numpy(dtype=float)
    F = df["Impact_Force_MN"].to_numpy(dtype=float) * 1e6  # [N]

    n = len(t)
    if n < 2 or np.allclose(F, 0.0):
        return pd.DataFrame()

    if use_logspace:
        f_min_eff = max(f_min, 1e-3)
        freqs = np.logspace(np.log10(f_min_eff), np.log10(f_max), n_freq)
    else:
        freqs = np.linspace(f_min, f_max, n_freq)

    Sd = np.zeros_like(freqs)
    Sv = np.zeros_like(freqs)
    Sa_g = np.zeros_like(freqs)

    for i, f in enumerate(freqs):
        omega_n = 2.0 * np.pi * f
        if omega_n <= 0.0:
            continue

        m = float(m_eff)
        k = m * omega_n * omega_n

        u, v, a = _solve_sdof_newmark_force(
            t=t,
            F=F,
            m=m,
            k=k,
            zeta=zeta,
        )

        u_max = float(np.max(np.abs(u)))
        Sd[i] = u_max
        Sv[i] = omega_n * u_max
        Sa_g[i] = (omega_n * omega_n * u_max) / GRAVITY

    return pd.DataFrame(
        {
            "freq_Hz": freqs,
            "Sd_m": Sd,
            "Sv_m_s": Sv,
            "Sa_g": Sa_g,
        }
    )


def compute_multi_damping_force_response_spectrum(
    df: pd.DataFrame,
    zeta_values,
    m_eff: float,
    f_min: float = 0.1,
    f_max: float = 100.0,
    n_freq: int = 80,
) -> pd.DataFrame:
    """
    Compute force-based response spectra for several damping ratios.
    """
    if m_eff <= 0.0:
        return pd.DataFrame()

    frames = []
    for zeta in np.asarray(list(zeta_values), dtype=float):
        if zeta <= 0.0:
            continue

        spec_df = compute_force_response_spectrum(
            df=df,
            zeta=zeta,
            m_eff=m_eff,
            f_min=f_min,
            f_max=f_max,
            n_freq=n_freq,
        )
        if spec_df is None or spec_df.empty:
            continue

        tmp = spec_df.copy()
        tmp["zeta"] = zeta
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def create_multi_damping_response_spectrum_plot(
    spec_multi_df: pd.DataFrame,
    zeta_ref: float | None = None,
) -> go.Figure:
    """
    Plot Sa(f) for several damping ratios ζ on a log frequency axis.
    """
    if spec_multi_df is None or spec_multi_df.empty:
        return go.Figure()

    fig = go.Figure()

    for zeta, grp in spec_multi_df.groupby("zeta"):
        label = f"ζ = {zeta*100:.0f} %"
        highlight = zeta_ref is not None and abs(zeta - zeta_ref) < 1e-4

        fig.add_trace(
            go.Scatter(
                x=grp["freq_Hz"],
                y=grp["Sa_g"],
                mode="lines",
                line=dict(
                    width=3.0 if highlight else 1.5,
                    dash="solid" if highlight else "dot",
                ),
                name=label,
            )
        )

    fig.update_xaxes(
        title_text="Frequency f (Hz)",
        type="log",
        dtick=1,            # one tick per decade: 0.1, 1, 10, 100, ...
    )

    fig.update_yaxes(title_text="Sa (g)")

    fig.update_layout(
        title="Force-based response spectrum for multiple damping ratios",
        height=500,
        legend_title="Damping ζ",
    )
    return fig


def create_response_spectrum_plot(
    spec_df: pd.DataFrame,
    m_build: float = 0.0,
) -> go.Figure:
    """
    Pseudo-acceleration Sa [g] vs f [Hz] + optional equivalent static force.
    """
    if spec_df is None or spec_df.empty:
        return go.Figure()

    use_force_axis = m_build is not None and m_build > 0.0

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": use_force_axis}]],
        subplot_titles=("Force-based response spectrum",),
    )

    fig.add_trace(
        go.Scatter(
            x=spec_df["freq_Hz"],
            y=spec_df["Sa_g"],
            mode="lines",
            line=dict(width=2),
            name="Pseudo-acceleration Sa",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    if use_force_axis:
        F_eq_MN = spec_df["Sa_g"] * GRAVITY * m_build / 1e6
        fig.add_trace(
            go.Scatter(
                x=spec_df["freq_Hz"],
                y=F_eq_MN,
                mode="lines",
                line=dict(width=2, dash="dash"),
                name=f"Equivalent force F_eq (m={m_build/1000:.0f} t)",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    fig.update_xaxes(
        title_text="Frequency f (Hz)",
        type="log",
        dtick=1,            # same: one major tick per decade
        row=1,
        col=1,
    )

    fig.update_yaxes(title_text="Sa (g)", row=1, col=1, secondary_y=False)
    if use_force_axis:
        fig.update_yaxes(title_text="F_eq (MN)", row=1, col=1, secondary_y=True)

    fig.update_layout(height=500, showlegend=True)
    return fig


# ====================================================================
# TRAIN GEOMETRY PLOT
# ====================================================================

def create_train_geometry_plot(params: Dict[str, Any]) -> go.Figure:
    """Plot lumped masses and cumulative (Riera-type) mass distribution."""
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


# ====================================================================
# STREAMLIT SIDEBAR: PARAMETER INPUT
# ====================================================================

def build_parameter_ui() -> Dict[str, Any]:
    """Build parameter input UI in sidebar."""
    with st.sidebar:
        st.header("⚙️ Parameters")

        params: Dict[str, Any] = {}

        # Time & Integration
        with st.expander("🕐 Time & Integration", expanded=True):
            v0_kmh = st.slider("Impact Velocity (km/h)", 10, 200, 56, 1)
            params["v0_init"] = -v0_kmh / 3.6

            h_ms = st.number_input(
                "Time Step Δt (ms)",
                0.01,
                1.0,
                0.1,
                0.01,
                help="Time step size in milliseconds",
            )
            T_max = st.number_input(
                "Max Simulation Time (s)",
                0.1,
                10.0,
                0.3,
                0.1,
                help=(
                    "Maximum simulation duration. "
                    "For building response, aim for at least 5–10 natural periods "
                    "of the equivalent SDOF."
                ),
            )

            params["h_init"] = h_ms / 1000.0
            params["T_max"] = T_max
            params["step"] = int(T_max / params["h_init"])
            params["T_int"] = (0.0, T_max)

            d0_cm = st.number_input(
                "Initial Distance to Wall (cm)",
                0.0,
                100.0,
                1.0,
                0.1,
                help="Additional initial gap between front mass and wall",
            )
            params["d0"] = d0_cm / 100.0

            angle_deg = st.number_input("Impact Angle (°)", 0.0, 45.0, 0.0, 0.1)
            params["angle_rad"] = angle_deg * np.pi / 180

            params["alpha_hht"] = st.slider(
                "HHT-α parameter",
                -0.3,
                0.0,
                -0.1,
                0.01,
                help="Negative values add numerical damping",
            )

            params["newton_tol"] = st.number_input(
                "Convergence tolerance", 1e-8, 1e-2, 1e-4, format="%.1e"
            )

            params["max_iter"] = st.number_input("Max iterations", 5, 100, 50, 1)

        # Train Geometry
        train_params = build_train_geometry_ui()
        params.update(train_params)

        # Material Properties
        material_params = build_material_ui(params["n_masses"])
        params.update(material_params)

        # Contact & Friction (+ Building SDOF)
        contact_params = build_contact_friction_ui()
        params.update(contact_params)

        # Apply YAML example config overrides (if selected in Train Geometry)
        yaml_cfg = st.session_state.get("yaml_example_cfg")
        if isinstance(yaml_cfg, dict) and yaml_cfg:
            apply_full = bool(st.session_state.get("yaml_apply_full", False))
            use_time = bool(st.session_state.get("yaml_use_time", True))
            use_material = bool(st.session_state.get("yaml_use_material", True))
            use_contact = bool(st.session_state.get("yaml_use_contact", True))

            time_keys = {
                "v0_init",
                "T_max",
                "h_init",
                "alpha_hht",
                "newton_tol",
                "max_iter",
                "d0",
                "angle_rad",
                "step",
                "T_int",
            }
            material_keys = {"fy", "uy", "bw_a", "bw_A", "bw_beta", "bw_gamma", "bw_n"}
            contact_keys = {
                "contact_model",
                "k_wall",
                "cr_wall",
                "mu_s",
                "mu_k",
                "sigma_0",
                "sigma_1",
                "sigma_2",
                # building keys (optional)
                "enable_building_sdof",
                "building_mass",
                "building_zeta",
                "building_height",
                "building_k",
                "building_c",
                "building_uy",
                "building_uy_mm",
                "building_alpha",
                "building_gamma",
            }

            if apply_full:
                params.update(yaml_cfg)
            else:
                if use_time:
                    for k in time_keys:
                        if k in yaml_cfg and yaml_cfg.get(k) is not None:
                            params[k] = yaml_cfg[k]
                if use_material:
                    for k in material_keys:
                        if k in yaml_cfg and yaml_cfg.get(k) is not None:
                            params[k] = np.asarray(yaml_cfg[k], dtype=float) if k in {"fy", "uy"} else yaml_cfg[k]
                if use_contact:
                    for k in contact_keys:
                        if k in yaml_cfg and yaml_cfg.get(k) is not None:
                            params[k] = yaml_cfg[k]

            # Ensure step/T_int are consistent if YAML changed T_max/h_init but did not set step
            if ("T_max" in yaml_cfg or "h_init" in yaml_cfg or "T_int" in yaml_cfg) and (
                "step" not in yaml_cfg or yaml_cfg.get("step") is None
            ):
                try:
                    T_max = float(params.get("T_max", 0.4))
                    h = float(params.get("h_init", 1e-4))
                    if h > 0:
                        params["step"] = int(np.ceil(T_max / h))
                        params["T_int"] = (0.0, T_max)
                except Exception:
                    pass

    return params



def build_train_geometry_ui() -> Dict[str, Any]:
    """Build train geometry UI.

    Supports:
      - Research locomotive model (default)
      - Built-in parametric presets
      - YAML example configs from ./configs/*.yml
    """
    with st.expander("🚃 Train Geometry", expanded=True):
        config_mode = st.radio(
            "Configuration mode",
            ("Research locomotive model", "Example trains", "YAML example configs"),
            index=0,
        )

        # Clear YAML selection when leaving YAML mode (avoid stale overrides).
        if config_mode != "YAML example configs":
            for k in (
                "yaml_example_cfg",
                "yaml_apply_full",
                "yaml_use_time",
                "yaml_use_material",
                "yaml_use_contact",
                "yaml_config_path",
            ):
                st.session_state.pop(k, None)

        if config_mode == "Research locomotive model":
            n_masses = st.slider("Number of Masses", 2, 20, 7)

            default_masses = np.array([4, 10, 4, 4, 4, 10, 4]) * 1000.0  # kg
            default_x = np.array([0.02, 3.02, 6.52, 10.02, 13.52, 17.02, 20.02])  # m
            default_y = np.zeros(7)

            if n_masses == 7:
                masses = default_masses
                x_init = default_x
                y_init = default_y
            else:
                M_total = st.number_input(
                    "Total Mass (kg)", 100.0, 1e6, 40000.0, 100.0
                )
                masses = np.ones(n_masses) * M_total / n_masses

                L_total = st.number_input(
                    "Total Length (m)", 1.0, 200.0, 20.0, 0.1
                )
                x_init = np.linspace(0.02, 0.02 + L_total, n_masses)
                y_init = np.zeros(n_masses)

            return {
                "n_masses": int(n_masses),
                "masses": masses,
                "x_init": x_init,
                "y_init": y_init,
            }

        if config_mode == "Example trains":
            train_config = build_example_train_ui()
            n_masses, masses, x_init, y_init = TrainBuilder.build_train(train_config)

            return {
                "n_masses": int(n_masses),
                "masses": masses,
                "x_init": x_init,
                "y_init": y_init,
            }

        # ------------------------------
        # YAML example configs
        # ------------------------------
        cfg_dir = Path.cwd() / "configs"
        if not cfg_dir.is_dir():
            # Fallback: repo layout when running from source tree
            try:
                cfg_dir = Path(__file__).resolve().parents[3] / "configs"
            except Exception:
                cfg_dir = Path.cwd() / "configs"

        yaml_files = sorted(list(cfg_dir.glob("*.yml")) + list(cfg_dir.glob("*.yaml")))
        if not yaml_files:
            st.warning(
                "No YAML configs found. Expected ./configs/*.yml in the repository. "
                "Falling back to 'Example trains'."
            )
            train_config = build_example_train_ui()
            n_masses, masses, x_init, y_init = TrainBuilder.build_train(train_config)
            return {
                "n_masses": int(n_masses),
                "masses": masses,
                "x_init": x_init,
                "y_init": y_init,
            }

        labels: list[str] = []
        parsed: list[dict] = []
        for p in yaml_files:
            try:
                d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                if not isinstance(d, dict):
                    d = {}
            except Exception:
                d = {}
            case = d.get("case_name", p.stem)
            labels.append(f"{p.name} — {case}")
            parsed.append(d)

        idx = st.selectbox(
            "YAML config",
            options=list(range(len(yaml_files))),
            format_func=lambda i: labels[i],
            index=0,
        )

        yaml_path = yaml_files[idx]
        yaml_cfg = parsed[idx] if isinstance(parsed[idx], dict) else {}

        st.session_state["yaml_example_cfg"] = yaml_cfg
        st.session_state["yaml_config_path"] = str(yaml_path)

        case_name = yaml_cfg.get("case_name", yaml_path.stem)

        st.caption(f"Selected YAML: **{yaml_path.name}**  ·  case_name: **{case_name}**")

        apply_full = st.checkbox(
            "Apply full YAML config (overrides all sidebar params)",
            value=False,
            help="If enabled, the simulation will run exactly with the YAML values (time/material/contact/etc).",
        )

        use_time = st.checkbox(
            "Use time/integration values from YAML (v0_init, T_max, h_init, α, tol, ...)",
            value=True,
            disabled=apply_full,
        )
        use_material = st.checkbox(
            "Use train material values from YAML (fy/uy/Bouc–Wen)",
            value=True,
            disabled=apply_full,
        )
        use_contact = st.checkbox(
            "Use contact/friction values from YAML (k_wall/contact model/μ)",
            value=True,
            disabled=apply_full,
        )

        st.session_state["yaml_apply_full"] = bool(apply_full)
        st.session_state["yaml_use_time"] = bool(use_time)
        st.session_state["yaml_use_material"] = bool(use_material)
        st.session_state["yaml_use_contact"] = bool(use_contact)

        # Geometry: required keys
        n_masses = int(yaml_cfg.get("n_masses", 0) or 0)
        masses_raw = yaml_cfg.get("masses", None)
        x_raw = yaml_cfg.get("x_init", None)
        y_raw = yaml_cfg.get("y_init", None)

        if n_masses <= 0 and isinstance(masses_raw, (list, tuple)):
            n_masses = len(masses_raw)

        if n_masses <= 0:
            st.error("YAML config does not define 'n_masses' or 'masses'.")
            n_masses = 2
            masses = np.ones(n_masses) * 20000.0
            x_init = np.linspace(0.0, 10.0, n_masses)
            y_init = np.zeros(n_masses)
        else:
            masses = np.asarray(masses_raw if masses_raw is not None else [0.0] * n_masses, dtype=float)
            x_init = np.asarray(x_raw if x_raw is not None else np.linspace(0.0, 10.0, n_masses), dtype=float)
            y_init = np.asarray(y_raw if y_raw is not None else np.zeros(n_masses), dtype=float)

            # Basic consistency warnings
            if masses.size != n_masses:
                st.warning(f"'masses' length ({masses.size}) != n_masses ({n_masses}). Engine will attempt to coerce.")
            if x_init.size != n_masses:
                st.warning(f"'x_init' length ({x_init.size}) != n_masses ({n_masses}). Engine will attempt to coerce.")
            if y_init.size != n_masses:
                st.warning(f"'y_init' length ({y_init.size}) != n_masses ({n_masses}). Engine will attempt to coerce.")

        with st.expander("Preview YAML (top-level keys)", expanded=False):
            st.json({k: yaml_cfg.get(k) for k in sorted(yaml_cfg.keys())})

        return {
            "case_name": case_name,
            "n_masses": int(n_masses),
            "masses": masses,
            "x_init": x_init,
            "y_init": y_init,
        }


def build_example_train_ui() -> TrainConfig:
    """Build example train configuration UI."""
    preset = st.selectbox(
        "Train preset",
        ["Generic European", "ICE3-like", "TGV-like", "TRAXX freight"],
    )

    presets = {
        "ICE3-like": (7, 70.0, 48.0, 25.0, 25.0),
        "TGV-like": (7, 68.0, 31.0, 22.0, 20.0),
        "TRAXX freight": (6, 84.0, 80.0, 19.0, 15.0),
        "Generic European": (7, 80.0, 50.0, 20.0, 20.0),
    }

    n_wag, m_lok, m_wag, L_lok, L_wag = presets[preset]

    n_wagons = int(st.number_input("Number of wagons", 0, 20, n_wag, 1))
    mass_lok_t = st.number_input(
        "Locomotive mass (t)", 10.0, 200.0, m_lok, 0.5
    )
    mass_wagon_t = st.number_input(
        "Wagon mass (t)", 10.0, 200.0, m_wag, 0.5
    )
    L_lok_val = st.number_input(
        "Locomotive length (m)", 5.0, 40.0, L_lok, 0.1
    )
    L_wagon_val = st.number_input(
        "Wagon length (m)", 5.0, 40.0, L_wag, 0.1
    )
    gap = st.number_input("Gap between cars (m)", 0.0, 5.0, 1.0, 0.1)

    mass_points_lok = st.radio("Mass points (loco)", [2, 3], index=1, horizontal=True)
    mass_points_wagon = st.radio(
        "Mass points (wagon)", [2, 3], index=0, horizontal=True
    )

    return TrainConfig(
        n_wagons=n_wagons,
        mass_lok_t=mass_lok_t,
        mass_wagon_t=mass_wagon_t,
        L_lok=L_lok_val,
        L_wagon=L_wagon_val,
        mass_points_lok=mass_points_lok,
        mass_points_wagon=mass_points_wagon,
        gap=gap,
    )


def build_material_ui(n_masses: int) -> Dict[str, Any]:
    """Build Bouc-Wen material parameters UI."""
    with st.expander("🔧 Bouc-Wen Material", expanded=True):

        st.markdown("---")
        st.markdown("### 📋 Train Material Presets (Chapter 7.5)")

        show_presets_info = st.checkbox("Show material comparison info", value=False)

        if show_presets_info:
            st.markdown(
                """
            **Influence of train materials on impact behavior:**

            Older generation trains (steel) are stiffer than modern trains (aluminum).
            This significantly affects peak forces and impact duration.

            **Figure 7.8: ICE 1 Material Comparison at 80 km/h**

            | Property | Aluminum ICE 1 | Steel S355 ICE 1 | Ratio |
            |----------|----------------|------------------|-------|
            | Peak Force | 11.81 MN | 18.73 MN | 1.6× |
            | Plateau Force | 8.5 MN | 18 MN | 2.1× |
            | Impact Duration | 1160 ms | 1700 ms | 1.5× |
            | Spring Fy | ~8 MN | ~18 MN | 2.25× |
            | Spring uy | ~100 mm | ~40 mm | 0.4× |
            | Stiffness k | ~80 MN/m | ~450 MN/m | 5.6× |

            **Key Observation:** Stiffer materials (steel) produce:
            - Higher peak forces (1.6× increase)
            - Higher plateau forces (2.1× increase)
            - Longer impact duration (1.5× increase)
            """
            )

        st.markdown("---")

        use_material_preset = st.checkbox("Use material preset", value=False)

        if use_material_preset:
            material_type = st.selectbox(
                "Train material",
                [
                    "Aluminum (Modern trains - ICE 1, ICE 3, TGV)",
                    "Steel S355 (Older generation trains)",
                    "Custom",
                ],
                help="Select train body material. Affects stiffness and energy dissipation.",
            )

            if "Aluminum" in material_type:
                st.info("📘 **Aluminum Train Properties** (Modern, lightweight construction)")
                fy_default = 8.0
                uy_default = 100.0
            elif "Steel" in material_type:
                st.info("🔩 **Steel S355 Train Properties** (Older generation, stiffer)")
                fy_default = 18.0
                uy_default = 40.0
            else:
                st.info("🔧 **Custom Material Properties**")
                fy_default = 15.0
                uy_default = 200.0

            col1, col2 = st.columns(2)
            with col1:
                fy_MN = st.number_input("Yield Force Fy (MN)", 0.1, 100.0, fy_default, 0.1)
            with col2:
                uy_mm = st.number_input("Yield Deformation uy (mm)", 1.0, 500.0, uy_default, 1.0)

            fy = np.ones(n_masses - 1) * fy_MN * 1e6
            uy = np.ones(n_masses - 1) * uy_mm / 1000

            k_spring = fy[0] / uy[0] / 1e6
            st.success(f"**Spring Stiffness: k = {k_spring:.1f} MN/m**")

        else:
            fy_MN = st.number_input("Yield Force Fy (MN)", 0.1, 100.0, 15.0, 0.1)
            fy = np.ones(n_masses - 1) * fy_MN * 1e6

            uy_mm = st.number_input("Yield Deformation uy (mm)", 1.0, 500.0, 200.0, 1.0)
            uy = np.ones(n_masses - 1) * uy_mm / 1000

            st.write(f"Stiffness: {(fy[0]/uy[0])/1e6:.1f} MN/m")

        return {
            "fy": fy,
            "uy": uy,
            "bw_a": st.slider("Elastic ratio (a)", 0.0, 1.0, 0.0, 0.05),
            "bw_A": st.number_input("A", 0.1, 10.0, 1.0, 0.1),
            "bw_beta": st.number_input("β", 0.0, 5.0, 0.1, 0.05),
            "bw_gamma": st.number_input("γ", 0.0, 5.0, 0.9, 0.05),
            "bw_n": int(st.number_input("n", 1, 20, 8, 1)),
        }


def build_contact_friction_ui() -> Dict[str, Any]:
    """Build contact and friction parameters UI (incl. building SDOF)."""
    params: Dict[str, Any] = {}

    with st.expander("💥 Contact", expanded=True):

        st.markdown("---")
        st.markdown("### 🧮 Wall Stiffness Calculator (Cantilever Method - Eq. 5.10)")

        show_calculator_info = st.checkbox("Show calculator formula", value=False)

        if show_calculator_info:
            st.markdown(
                r"""
            **Cantilever beam approximation for wall stiffness:**

            $$k_{eff} = \frac{6EI}{x^2(3a-x)}$$

            Where:
            - E = Young's modulus of wall material [Pa]
            - I = Second moment of area [m^4]
            - a = Distance from support to impact point [m]
            - x = Distance from impact point to top [m]
            - l = Total cantilever length, l = a + x [m]
            """
            )

        use_calculator = st.checkbox("Use calculator to estimate k_wall")

        if use_calculator:
            col1, col2 = st.columns(2)
            with col1:
                E_GPa = st.number_input(
                    "E - Young's Modulus (GPa)",
                    1.0,
                    500.0,
                    30.0,
                    1.0,
                    help="Concrete: ~30 GPa, Steel: ~200 GPa",
                )
                a_m = st.number_input("a - Distance from support (m)", 0.1, 20.0, 2.0, 0.1)
                x_m = st.number_input("x - Distance to top (m)", 0.1, 20.0, 1.0, 0.1)

            with col2:
                width_m = st.number_input(
                    "Width (m)", 0.1, 10.0, 1.0, 0.1, help="Rectangular section width"
                )
                height_m = st.number_input(
                    "Height (m)", 0.1, 5.0, 0.5, 0.1, help="Rectangular section height"
                )

                I = (width_m * height_m ** 3) / 12.0
                st.write(f"I = bh³/12 = {I:.6e} m⁴")

            E_Pa = E_GPa * 1e9
            k_eff = (6 * E_Pa * I) / (x_m ** 2 * (3 * a_m - x_m))
            k_eff_MN_m = k_eff / 1e6

            st.success(f"**Calculated k_eff = {k_eff_MN_m:.2f} MN/m**")

            if st.button("✓ Use this value"):
                params["k_wall"] = k_eff
            else:
                params["k_wall"] = (
                    st.number_input(
                        "Wall Stiffness (MN/m)", 1.0, 1000.0, k_eff_MN_m, 1.0
                    )
                    * 1e6
                )
        else:
            params["k_wall"] = (
                st.number_input("Wall Stiffness (MN/m)", 1.0, 100.0, 45.0, 1.0) * 1e6
            )

        st.markdown("---")

        st.markdown("### ℹ️ Coefficient of Restitution Reference (Table 5.4)")

        show_cr_reference = st.checkbox(
            "Show coefficient of restitution table", value=False
        )

        if show_cr_reference:
            st.markdown(
                """
            **Typical values for different contact situations:**

            | Contact Situation | cr [-] |
            |------------------|--------|
            | Collision of two train wagons | 0.90 - 0.95 |
            | Concrete and steel | 0.86 |
            | Dynamic behavior of reinforced concrete structure | 0.80 |
            | Concrete and aluminum | 0.76 |
            | Elastomeric bearing of reinforced concrete structure | 0.50 |
            | Rubber-block | 0.37 - 0.44 |
            """
            )

        params["cr_wall"] = st.slider(
            "Coeff. of Restitution",
            0.1,
            0.99,
            0.8,
            0.01,
            help=(
                "cr=0.8 for concrete (dynamic), 0.86 for steel, "
                "0.90–0.95 for train–train collision"
            ),
        )

        st.markdown("### 📖 Contact model recommendations")
        st.info(
            "For **hard impacts of trains against stiff walls/abutments** "
            "(crushing dominated by the vehicle), a **Hertz-type model with "
            "energy-consistent damping** reproduces the front-mass acceleration "
            "and contact duration better than a purely linear Kelvin–Voigt law.\n\n"
            "- **Recommended default:** `lankarani-nikravesh` – Hertz contact with "
            "energy-consistent damping. In parametric studies around **50 km/h**, "
            "it matches the measured acceleration history particularly well.\n"
            "- **Alternative (linear pounding):** `anagnostopoulos` – classic "
            "linear spring–dashpot (Kelvin–Voigt), robust and suitable for "
            "building–building pounding or when a simple linear model is preferred.\n"
            "- Other Hertz-type models (`hunt-crossley`, `gonthier`, `flores`) "
            "can be used to explore sensitivity of the results to dissipation "
            "formulation."
        )

        show_model_desc = st.checkbox(
            "Show short description of each contact model",
            value=False,
        )
        if show_model_desc:
            st.markdown(
                """
            - **anagnostopoulos** – Linear spring + dashpot (Kelvin–Voigt).  
              Good for building–building pounding, simple and robust.
            - **ye / pant-wijeyewickrema** – Linear spring with refined damping;
              still linear in penetration but with energy-based damping terms.
            - **hooke** – Purely elastic linear spring (no rate dependence).
            - **hertz** – Elastic Hertz contact (δ¹⋅⁵), no damping.
            - **hunt-crossley** – Hertz contact with velocity-dependent damping term.
            - **lankarani-nikravesh** – Hertz contact with energy-consistent damping;
              widely used for impacts and recommended here for train–wall collisions.
            - **flores / gonthier** – Alternative energy-based Hertz-type laws,
              useful to check model sensitivity.
            """
            )

        contact_model_options = [
            "anagnostopoulos",
            "ye",
            "hooke",
            "hertz",
            "hunt-crossley",
            "lankarani-nikravesh",
            "flores",
            "gonthier",
            "pant-wijeyewickrema",
        ]

        params["contact_model"] = st.selectbox(
            "Contact model",
            contact_model_options,
            index=5,  # lankarani-nikravesh
            help=(
                "For train–wall impacts, 'lankarani-nikravesh' is recommended. "
                "Use 'anagnostopoulos' for linear pounding or when a simple "
                "Kelvin–Voigt contact law is desired."
            ),
        )

        # Building SDOF configuration
        st.markdown("---")
        st.markdown("### 🏢 Building SDOF (pier/abutment response)")

        params["building_enable"] = st.checkbox(
            "Compute equivalent building (SDOF) response",
            value=True,
            help=(
                "Represents a cantilever pier/abutment excited by the contact force. "
                "Stiffness is taken from k_wall; you choose effective mass, "
                "damping and hysteresis model."
            ),
        )

        if params["building_enable"]:
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                m_build_t = st.number_input(
                    "Effective mass at impact level (t)",
                    10.0,
                    5000.0,
                    500.0,
                    10.0,
                    help="Lumped mass of pier + superstructure in the first mode.",
                )
            with bc2:
                zeta_build = st.slider(
                    "Modal damping ζ [-]",
                    0.0,
                    0.2,
                    0.05,
                    0.005,
                    help="Typical RC: 0.02–0.05; heavily damped systems up to ~0.10.",
                )
            with bc3:
                h_build = st.number_input(
                    "Representative height (m)",
                    2.0,
                    40.0,
                    8.0,
                    0.5,
                    help="Used only for the SDOF cantilever animation.",
                )

            params["building_mass"] = m_build_t * 1000.0  # kg
            params["building_zeta"] = zeta_build
            params["building_height"] = h_build

            building_model = st.selectbox(
                "Building model type",
                ["Linear elastic SDOF", "Takeda degrading hysteresis"],
                index=0,
            )

            if building_model == "Takeda degrading hysteresis":
                st.markdown(
                    "Takeda-type bilinear with pinching (post-processing). "
                    "Use u<sub>y</sub>, α and γ to control yield, post-yield slope "
                    "and pinching intensity.",
                    unsafe_allow_html=True,
                )
                colt1, colt2, colt3 = st.columns(3)
                with colt1:
                    building_uy_mm = st.number_input(
                        "Yield displacement u_y at top (mm)",
                        min_value=0.1,
                        max_value=500.0,
                        value=10.0,
                        step=0.5,
                    )
                with colt2:
                    building_alpha = st.number_input(
                        "Post-yield stiffness ratio α",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.05,
                        step=0.01,
                    )
                with colt3:
                    building_gamma = st.number_input(
                        "Takeda pinching factor γ",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.4,
                        step=0.05,
                        help="0 = strong pinching, 1 = no pinching",
                    )
            else:
                building_uy_mm = st.number_input(
                    "Reference yield displacement u_y (mm, for info only)",
                    min_value=0.1,
                    max_value=500.0,
                    value=10.0,
                    step=0.5,
                )
                building_alpha = 0.05
                building_gamma = 0.4

            params["building_model"] = building_model
            params["building_uy_mm"] = building_uy_mm
            params["building_uy"] = building_uy_mm / 1000.0
            params["building_alpha"] = building_alpha
            params["building_gamma"] = building_gamma

            # Dynamic properties
            try:
                omega_n = (params["k_wall"] / params["building_mass"]) ** 0.5
                f_n = omega_n / (2.0 * np.pi)
                Tn = 2.0 * np.pi / omega_n
                c_crit = 2.0 * params["building_mass"] * omega_n
                c = 2.0 * params["building_zeta"] * params["building_mass"] * omega_n

                st.markdown("#### Building dynamic properties")
                colp1, colp2, colp3 = st.columns(3)
                with colp1:
                    st.metric("fₙ [Hz]", f"{f_n:.2f}")
                    st.metric("Tₙ [s]", f"{Tn:.2f}")
                with colp2:
                    st.metric("ωₙ [rad/s]", f"{omega_n:.2f}")
                    st.metric("ζ [-]", f"{params['building_zeta']:.3f}")
                with colp3:
                    st.metric("c₍crit₎ [kNs/m]", f"{c_crit/1000.0:.1f}")
                    st.metric("c [kNs/m]", f"{c/1000.0:.1f}")

                st.caption(
                    "For a clear ring-down of the building, choose 'Max Simulation Time' "
                    "in the time settings to be ≥ 5–10 × Tₙ."
                )
            except Exception:
                pass
        else:
            params["building_mass"] = 0.0
            params["building_zeta"] = 0.0
            params["building_height"] = 0.0
            params["building_model"] = "Linear elastic SDOF"
            params["building_uy_mm"] = 10.0
            params["building_uy"] = 0.01
            params["building_alpha"] = 0.05
            params["building_gamma"] = 0.4

    with st.expander("🛞 Friction", expanded=True):
        params["friction_model"] = st.selectbox(
            "Friction model",
            ["lugre", "dahl", "coulomb", "brown-mcphee"],
            index=0,
        )
        params["mu_s"] = st.slider("μs (static)", 0.0, 1.0, 0.4, 0.01)
        params["mu_k"] = st.slider("μk (kinetic)", 0.0, 1.0, 0.3, 0.01)
        params["sigma_0"] = st.number_input("σ₀", 1e3, 1e7, 1e5, format="%.0e")
        params["sigma_1"] = st.number_input("σ₁", 1.0, 1e5, 316.0, 1.0)
        params["sigma_2"] = st.number_input("σ₂ (viscous)", 0.0, 2.0, 0.4, 0.1)

    return params


# ====================================================================
# EXECUTE SIMULATION + TABS
# ====================================================================

def execute_simulation(params: Dict[str, Any], run_new: bool = False):
    """
    Ejecuta (o reutiliza) la simulación vía core.engine y muestra los resultados.

    - Si run_new=True o no hay resultados cacheados, se llama a run_simulation(params)
      y se guarda el DataFrame base (impacto tren-muro) en st.session_state["sim_results"].
    - En todos los casos se recalcula el SDOF del edificio con los parámetros
      actuales (barato) y se dibujan las tabs.
    """
    # ------------------------------------------------------
    # 1) Simulación base de impacto (cara → solo con botón)
    # ------------------------------------------------------
    df_core = st.session_state.get("sim_results", None)

    if run_new or df_core is None:
        with st.spinner("Running HHT-α simulation..."):
            try:
                df_core = run_simulation(params)
            except Exception as e:
                st.error(f"Simulation error: {e}")
                return

        st.session_state["sim_results"] = df_core
        st.session_state["sim_params_core"] = params
        st.success("✅ Complete!")
    else:
        st.info(
            "Using cached impact history from last run. "
            "Press **Run Simulation** again after changing time/train/contact parameters."
        )

    # ------------------------------------------------------
    # 2) SDOF del edificio (barato → recalcular siempre)
    # ------------------------------------------------------
    building_df = None
    df = df_core.copy()

    if (
        params.get("building_enable", False)
        and params.get("k_wall", 0.0) > 0.0
        and params.get("building_mass", 0.0) > 0.0
    ):
        try:
            building_df = compute_building_sdof_response(
                df_core,
                k_wall=params["k_wall"],
                m_build=params["building_mass"],
                zeta=params["building_zeta"],
                model=params.get("building_model", "Linear elastic SDOF"),
                uy_mm=params.get("building_uy_mm", 10.0),
                alpha=params.get("building_alpha", 0.05),
                gamma=params.get("building_gamma", 0.4),
            )
            if building_df is not None and not building_df.empty:
                df = pd.concat([df_core, building_df], axis=1)
        except Exception as e:
            st.warning(f"Building SDOF response could not be computed: {e}")
            building_df = None

    # ------------------------------------------------------
    # 3) Tabs de resultados (igual que antes, usando df + building_df)
    # ------------------------------------------------------
    tab_global, tab_building, tab_train = st.tabs(
        ["📈 Global Results", "🏢 Building Response (SDOF)", "🚃 Train Configuration"]
    )

    # ----- Global results -----
    with tab_global:
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Force", f"{df['Impact_Force_MN'].max():.2f} MN")
        c2.metric("Max Penetration", f"{df['Penetration_mm'].max():.2f} mm")
        c3.metric(
            "Max Acceleration (front mass)",
            f"{df['Acceleration_g'].max():.1f} g",
        )

        if "E_total_initial_J" in df.columns:
            E0 = float(df["E_total_initial_J"].iloc[0])
            Eb_err = float(np.abs(df["E_balance_error_J"]).max())
            st.caption(
                f"Initial mechanical energy ≈ {E0/1e6:.2f} MJ, "
                f"max. energy balance deviation ≈ {Eb_err/1e6:.3f} MJ."
            )

        fig = create_results_plots(df)
        st.plotly_chart(fig, width="stretch")

        st.subheader("📥 Export")
        e1, e2, e3 = st.columns(3)

        e1.download_button(
            "📄 CSV",
            df.to_csv(index=False).encode(),
            "results.csv",
            "text/csv",
            width="stretch",
        )

        e2.download_button(
            "📝 TXT",
            df.to_string(index=False).encode(),
            "results.txt",
            "text/plain",
            width="stretch",
        )

        e3.download_button(
            "📊 XLSX",
            to_excel(df),
            "results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    # ----- Building SDOF response -----
    with tab_building:
        if (
            params.get("building_enable", False)
            and building_df is not None
            and not building_df.empty
        ):
            st.markdown(
                "This view shows the response of an equivalent **SDOF building/pier** "
                "excited by the simulated contact force. Stiffness is taken from "
                "`k_wall`; effective mass, damping and – optionally – a **Takeda "
                "degrading hysteresis** are defined in the 💥 Contact section."
            )

            fig_b = create_building_response_plots(df)
            st.plotly_chart(fig_b, width="stretch")

            st.markdown("#### Building hysteresis (restoring force vs displacement)")
            try:
                fig_h = create_building_hysteresis_plot(df)
                st.plotly_chart(fig_h, width="stretch")
            except Exception as e:
                st.warning(f"Building hysteresis could not be plotted: {e}")

            building_height_m = params.get("building_height", 0.0)
            if building_height_m > 0.0:
                st.markdown("#### Cantilever animation (top displacement over time)")

                # Visual scale factor
                scale_factor = st.slider(
                    "Visual scale factor for horizontal displacement",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    help="Purely visual scaling. Does not affect the analysis results.",
                )

                # Impact height slider
                h_train = 3.0  # m
                h_imp_max = float(min(h_train, building_height_m))
                h_imp_default = float(min(h_train * 5.0 / 12.0, h_imp_max))

                impact_height_m = st.slider(
                    "Impact height above track (m)",
                    min_value=0.1,
                    max_value=h_imp_max,
                    value=h_imp_default,
                    step=0.05,
                    help=(
                        "Standard impact height is roughly between 1/3 and 1/2 of the "
                        "train height (h_train ≈ 3 m)."
                    ),
                )

                # Playback speed slider
                speed_factor = st.slider(
                    "Playback speed factor",
                    min_value=0.25,
                    max_value=4.0,
                    value=1.0,
                    step=0.25,
                    help="0.5 = slower, 2.0 = faster, etc.",
                )

                # Optional beam-element mode
                use_beam_element = st.checkbox(
                    "Use beam-element mode (slower, experimental)",
                    value=False,
                    help=(
                        "If enabled, this is the hook for a full beam-element time "
                        "integration with fixed base (u=0, θ=0) after impact. "
                        "May take longer to compute."
                    ),
                )

                try:
                    anim_fig = create_building_animation(
                        df=df,
                        height_m=building_height_m,
                        scale_factor=scale_factor,
                        impact_height_m=impact_height_m,
                        speed_factor=speed_factor,
                        use_beam_element=use_beam_element,
                    )
                    if anim_fig is not None:
                        st.plotly_chart(anim_fig, width="stretch")
                except Exception as e:
                    st.warning(f"Cantilever animation could not be created: {e}")

            st.markdown("#### Force-based response spectrum (pseudo-acceleration)")
            try:
                k_wall = params.get("k_wall", 0.0)
                m_build = params.get("building_mass", 0.0)
                zeta_build = params.get("building_zeta", 0.05)

                if k_wall > 0.0 and m_build > 0.0:
                    omega_build = np.sqrt(k_wall / m_build)
                    f_build = omega_build / (2.0 * np.pi)
                else:
                    f_build = None

                f_min = 0.1
                f_max = 100.0
                if f_build is not None:
                    f_max = max(f_max, 5.0 * f_build)

                spec_df = compute_force_response_spectrum(
                    df,
                    zeta=zeta_build,
                    m_eff=m_build,
                    f_min=f_min,
                    f_max=f_max,
                    n_freq=80,
                    use_logspace=True,
                )

                if spec_df is not None and not spec_df.empty:
                    fig_spec = create_response_spectrum_plot(
                        spec_df,
                        m_build=m_build,
                    )
                    st.plotly_chart(fig_spec, width="stretch")

                    if f_build is not None:
                        freq_array = spec_df["freq_Hz"].to_numpy()
                        idx = int(np.argmin(np.abs(freq_array - f_build)))
                        Sa_build = float(spec_df["Sa_g"].iloc[idx])
                        F_eq_build_MN = Sa_build * GRAVITY * m_build / 1e6

                        st.caption(
                            f"At the building fundamental frequency "
                            f"f₁ ≈ {f_build:.2f} Hz: "
                            f"Sa ≈ {Sa_build:.2f} g, "
                            f"F_eq ≈ {F_eq_build_MN:.2f} MN."
                        )

                    # Multi-damping spectra
                    zeta_values = [0.01, 0.02, 0.05, 0.10]
                    if zeta_build > 0.0 and zeta_build not in zeta_values:
                        zeta_values.append(zeta_build)
                    zeta_values = sorted(set(zeta_values))

                    spec_multi_df = compute_multi_damping_force_response_spectrum(
                        df=df,
                        zeta_values=zeta_values,
                        m_eff=m_build,
                        f_min=f_min,
                        f_max=f_max,
                        n_freq=80,
                    )

                    if spec_multi_df is not None and not spec_multi_df.empty:
                        fig_multi = create_multi_damping_response_spectrum_plot(
                            spec_multi_df,
                            zeta_ref=zeta_build,
                        )
                        st.plotly_chart(fig_multi, width="stretch")

            except Exception as e:
                st.warning(f"Response spectrum could not be computed: {e}")
        else:
            st.info(
                "Enable **Building SDOF response** under 💥 Contact to compute "
                "and visualise building accelerations and hysteresis."
            )

    # ----- Train geometry / Riera mass distribution -----
    with tab_train:
        st.markdown("### Train configuration and Riera-type mass distribution")
        fig_train = create_train_geometry_plot(params)
        st.plotly_chart(fig_train, width="stretch")

        masses = np.asarray(params["masses"], dtype=float)
        x = np.asarray(params["x_init"], dtype=float)
        total_mass_t = masses.sum() / 1000.0 if masses.size > 0 else 0.0
        if masses.size > 0:
            x_cm = float(np.sum(masses * x) / masses.sum())
            st.caption(
                f"Total train mass ≈ {total_mass_t:.1f} t, "
                f"center of mass at x ≈ {x_cm:.2f} m (measured from the front node). "
                "The lower curve M(x) = ∑ mᵢ up to x corresponds to the discrete "
                "Riera mass distribution."
            )
        else:
            st.caption("No mass data available for the current configuration.")



# ====================================================================
# MAIN STREAMLIT ENTRYPOINT
# ====================================================================


def main():
    """Main Streamlit application."""
    st.set_page_config(
        layout="wide",
        page_title="Railway Impact Simulator - DZSF Research",
        page_icon="🚂",
    )

    # Sidebar parameter UI is shared across all tabs
    params = build_parameter_ui()

    tab_sim, tab_param, tab_about = st.tabs(
        ["🚂 Simulator", "🧪 Parametric Studies", "📖 About / Documentation"]
    )

    # --------------------------------------------------------------
    # SIMULATOR TAB
    # --------------------------------------------------------------
    with tab_sim:
        st.title("Railway Impact Simulator")
        st.markdown("**HHT-α implicit integration with Bouc–Wen hysteresis**")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📊 Configuration")
            try:
                st.metric("Velocity", f"{-float(params['v0_init']) * 3.6:.1f} km/h")
            except Exception:
                st.metric("Velocity", "—")
            st.metric("Masses", int(params.get("n_masses", 0)))
            st.metric("Time Step", f"{float(params.get('h_init', 0.0))*1000:.2f} ms")
            st.metric("Initial Gap", f"{float(params.get('d0', 0.0))*100:.1f} cm")
            if params.get("case_name"):
                st.caption(f"case_name: **{params['case_name']}**")
            st.markdown("---")
            run_btn = st.button(
                "▶️ **Run Simulation**",
                type="primary",
                width="stretch",
            )

        with col2:
            has_results = st.session_state.get("sim_results", None) is not None

            if run_btn:
                execute_simulation(params, run_new=True)
            elif has_results:
                execute_simulation(params, run_new=False)
            else:
                st.info(
                    "👈 Configure parameters in the sidebar and press **Run Simulation**"
                )

    # --------------------------------------------------------------
    # PARAMETRIC STUDIES TAB
    # --------------------------------------------------------------
    with tab_param:
        st.title("Parametric Studies")
        st.markdown(
            "Run reproducible sweeps using the same configuration as the Simulator tab."
        )

        sub_env, sub_sens, sub_dif = st.tabs(
            ["🚄 Speed envelope", "🧮 Numerics sensitivity", "⚡ Strain-rate (DIF)"]
        )

        # --------------------------
        # Speed envelope (existing)
        # --------------------------
        with sub_env:
            st.markdown("### Envelope over multiple speeds")

            st.write(
                "This parametric study reuses the current train/material/contact "
                "settings and only varies the impact speed. Define a set of "
                "speeds and statistical weights; the tool computes the quantity "
                "envelope ('Umhüllende') and a weighted mean history."
            )

            default_data = {
                "speed_kmh": [80.0, 56.0, 40.0],
                "weight": [0.25, 0.50, 0.25],
            }

            df_scenarios = st.data_editor(
                pd.DataFrame(default_data),
                num_rows="dynamic",
                key="parametric_table",
            )

            quantity_options = {
                "Impact force at barrier [MN]": "Impact_Force_MN",
                "Vehicle acceleration [g]": "Acceleration_g",
                "Penetration [mm]": "Penetration_mm",
            }

            quantity_label = st.selectbox(
                "Quantity for envelope",
                list(quantity_options.keys()),
                index=0,
                key="env_quantity",
            )
            quantity = quantity_options[quantity_label]

            if st.button("Run parametric envelope", type="primary", key="run_env"):
                try:
                    speeds = df_scenarios["speed_kmh"].astype(float).tolist()
                    weights = df_scenarios["weight"].astype(float).tolist()

                    if not speeds:
                        st.warning("Please define at least one scenario.")
                    else:
                        base_params = dict(params)

                        scenarios = build_speed_scenarios(
                            base_params,
                            speeds_kmh=speeds,
                            weights=weights,
                            prefix="v",
                        )

                        envelope_df, summary_df, _ = run_parametric_envelope(
                            scenarios, quantity=quantity
                        )

                        fig_env = make_envelope_figure(
                            envelope_df,
                            quantity=quantity,
                            title=f"{quantity_label} – envelope over defined speeds",
                        )
                        st.plotly_chart(fig_env, width="stretch")

                        st.markdown("#### Scenario summary")
                        st.dataframe(summary_df)
                except Exception as exc:
                    st.error(f"Parametric study failed: {exc}")

        # --------------------------
        # Numerics sensitivity
        # --------------------------
        with sub_sens:
            st.markdown("### Sensitivity to numerical parameters (Δt, HHT-α, tolerance)")

            left, right = st.columns([1, 2])

            with left:
                st.caption("Comma-separated lists are supported (e.g. `1e-4,2e-4,5e-5`).")

                q_label = st.selectbox(
                    "Quantity",
                    ["Impact_Force_MN", "Acceleration_g", "Penetration_mm"],
                    index=0,
                    key="sens_quantity",
                )

                dt_str = st.text_input("Δt values (s)", value="1e-4,2e-4", key="sens_dt")
                alpha_str = st.text_input("HHT-α values", value="-0.15", key="sens_alpha")
                tol_str = st.text_input("Tolerance values", value="1e-4", key="sens_tol")

                max_runs = st.number_input("Max runs to plot (overlay)", 1, 20, 10, 1)

                run_sens = st.button("Run numerics sensitivity", type="primary", key="run_sens")

            with right:
                if run_sens:
                    try:
                        dt_vals = parse_floats_csv(dt_str) if dt_str.strip() else None
                        alpha_vals = parse_floats_csv(alpha_str) if alpha_str.strip() else None
                        tol_vals = parse_floats_csv(tol_str) if tol_str.strip() else None

                        captured: list[tuple[dict, pd.DataFrame]] = []

                        def _cap_sim(cfg: Dict[str, Any]) -> pd.DataFrame:
                            df = run_simulation(cfg)
                            captured.append((cfg, df))
                            return df

                        summary_df = run_numerics_sensitivity(
                            dict(params),
                            dt_values=dt_vals,
                            alpha_values=alpha_vals,
                            tol_values=tol_vals,
                            quantity=q_label,
                            simulate_func=_cap_sim,
                        )

                        st.session_state["sens_summary_df"] = summary_df
                        st.session_state["sens_captured"] = captured

                    except Exception as exc:
                        st.error(f"Sensitivity study failed: {exc}")

                summary_df = st.session_state.get("sens_summary_df", None)
                captured = st.session_state.get("sens_captured", None)

                if summary_df is None:
                    st.info("Run the study to see results.")
                else:
                    st.markdown("#### Summary")
                    st.dataframe(summary_df)

                    # Simple peak plot
                    try:
                        fig_peak = go.Figure()
                        fig_peak.add_trace(
                            go.Scatter(
                                x=summary_df["dt_s"],
                                y=summary_df["peak_force_MN"],
                                mode="markers+lines",
                                name="Peak force",
                            )
                        )
                        fig_peak.update_layout(
                            title="Peak force vs Δt (aggregated over α/tol)",
                            xaxis_title="Δt (s)",
                            yaxis_title="Peak force (MN)",
                            height=350,
                        )
                        st.plotly_chart(fig_peak, width="stretch")
                    except Exception:
                        pass

                    # Overlay time histories for the selected quantity
                    if captured:
                        st.markdown("#### Time histories (overlay)")
                        fig = go.Figure()
                        shown = 0
                        for cfg, df in captured:
                            if shown >= int(max_runs):
                                break
                            if q_label not in df.columns:
                                continue
                            if "Time_ms" in df.columns:
                                t = df["Time_ms"]
                                xlab = "Time (ms)"
                            else:
                                t = df["Time_s"] * 1000.0
                                xlab = "Time (ms)"
                            dt = float(cfg.get("h_init", np.nan))
                            a = float(cfg.get("alpha_hht", np.nan))
                            tol = float(cfg.get("newton_tol", np.nan))
                            label = f"dt={dt:.1e}s, α={a:+.2f}, tol={tol:.1e}"
                            fig.add_trace(go.Scatter(x=t, y=df[q_label], mode="lines", name=label))
                            shown += 1
                        fig.update_layout(
                            xaxis_title=xlab,
                            yaxis_title=q_label,
                            height=450,
                        )
                        st.plotly_chart(fig, width="stretch")

        # --------------------------
        # Strain-rate proxy (Fixed DIF)
        # --------------------------
        with sub_dif:
            st.markdown("### Strain-rate influence (proxy): fixed DIF multiplier")

            left, right = st.columns([1, 2])

            with left:
                st.caption(
                    "Applies k_scaled = k0 * DIF to a scalar parameter path (default: k_wall). "
                    "Advanced: you can set paths like `fy[0]`."
                )

                dif_str = st.text_input("DIF values", value="0.8,1.0,1.2", key="dif_vals")
                k_path = st.text_input("Parameter path to scale", value="k_wall", key="dif_kpath")
                q_label = st.selectbox(
                    "Quantity",
                    ["Impact_Force_MN", "Acceleration_g", "Penetration_mm"],
                    index=0,
                    key="dif_quantity",
                )
                max_runs = st.number_input("Max runs to plot (overlay)", 1, 20, 10, 1, key="dif_max_runs")

                run_dif = st.button("Run DIF study", type="primary", key="run_dif")

            with right:
                if run_dif:
                    try:
                        dif_vals = parse_floats_csv(dif_str) if dif_str.strip() else [1.0]

                        captured: list[tuple[dict, pd.DataFrame]] = []

                        def _cap_sim(cfg: Dict[str, Any]) -> pd.DataFrame:
                            df = run_simulation(cfg)
                            captured.append((cfg, df))
                            return df

                        summary_df = run_fixed_dif_sensitivity(
                            dict(params),
                            dif_vals,
                            k_path=k_path.strip(),
                            quantity=q_label,
                            simulate_func=_cap_sim,
                        )

                        st.session_state["dif_summary_df"] = summary_df
                        st.session_state["dif_captured"] = captured

                    except Exception as exc:
                        st.error(f"DIF study failed: {exc}")

                summary_df = st.session_state.get("dif_summary_df", None)
                captured = st.session_state.get("dif_captured", None)

                if summary_df is None:
                    st.info("Run the study to see results.")
                else:
                    st.markdown("#### Summary")
                    st.dataframe(summary_df)

                    # Peak vs DIF
                    try:
                        fig_peak = go.Figure()
                        fig_peak.add_trace(
                            go.Scatter(
                                x=summary_df["dif"],
                                y=summary_df["peak_force_MN"],
                                mode="markers+lines",
                                name="Peak force",
                            )
                        )
                        fig_peak.update_layout(
                            title=f"Peak force vs DIF (scaling {k_path})",
                            xaxis_title="DIF",
                            yaxis_title="Peak force (MN)",
                            height=350,
                        )
                        st.plotly_chart(fig_peak, width="stretch")
                    except Exception:
                        pass

                    # Overlay time histories
                    if captured:
                        st.markdown("#### Time histories (overlay)")
                        fig = go.Figure()
                        shown = 0
                        for cfg, df in captured:
                            if shown >= int(max_runs):
                                break
                            if q_label not in df.columns:
                                continue
                            if "Time_ms" in df.columns:
                                t = df["Time_ms"]
                                xlab = "Time (ms)"
                            else:
                                t = df["Time_s"] * 1000.0
                                xlab = "Time (ms)"
                            dif = float(cfg.get("dif", np.nan)) if "dif" in cfg else np.nan
                            # If dif isn't present in cfg (it won't be), infer from scaled parameter
                            dif_val = float(summary_df["dif"].iloc[shown]) if (shown < len(summary_df)) else float("nan")
                            label = f"DIF={dif_val:.3f}"
                            fig.add_trace(go.Scatter(x=t, y=df[q_label], mode="lines", name=label))
                            shown += 1
                        fig.update_layout(
                            xaxis_title=xlab,
                            yaxis_title=q_label,
                            height=450,
                        )
                        st.plotly_chart(fig, width="stretch")

    # --------------------------------------------------------------
    # ABOUT / DOCUMENTATION TAB
    # --------------------------------------------------------------
    with tab_about:
        display_header()
        display_citation()


if __name__ == "__main__":
    main()