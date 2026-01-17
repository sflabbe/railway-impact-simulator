"""
SDOF (Single Degree of Freedom) building response calculations and response spectra.

This module provides functions for:
- Newmark-β time integration for SDOF systems
- Takeda-type hysteresis modeling
- Building response calculations under impact loads
- Response spectrum computations
- Visualization of building response and spectra
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.constants import g as GRAVITY



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
        h_imp_std = h_train * 5.0 / 12.0  # average 1/3–1/2 ≈ 1.25 m
        impact_height_m = min(h_imp_std, height_m - 0.1)

    # Safety clamp
    impact_height_m = float(np.clip(impact_height_m, 0.1, height_m - 1e-3))

    # Vertical coordinates
    y_coords = np.linspace(0.0, height_m, n_shape_points)

    # ---- FAST MODE: static cantilever shape scaled with u_top(t) ----
    # (You can hook in the dynamic “beam element” here later if needed.)
    phi = _cantilever_shape_with_point_load(
        y_coords, L=height_m, a=impact_height_m
    )

    # X-axis scale
    max_disp = float(np.max(np.abs(u_scaled))) if np.any(u_scaled) else 0.05
    x_lim = max(0.05, 1.2 * max_disp)

    frames = []
    for idx in range(0, n, stride):
        if not use_beam_element:
            # Fast: static shape φ(y) * u_top(t)
            x_frame = phi * u_scaled[idx]
        else:
            # FUTURE: you could insert a dynamic beam element solution here
            # w(y, t_idx) computed with M, C, K, and F(t). For now, use the same.
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

