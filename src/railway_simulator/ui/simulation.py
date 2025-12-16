"""
Simulation execution and results display for the Railway Impact Simulator UI.

Provides the main simulation execution logic and results visualization including:
- Simulation execution and caching
- Results tabbed interface (Results, Animation, Building SDOF, Response Spectrum)
- Export functionality (CSV, Excel)
- Interactive plots and animations
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from railway_simulator.core.engine import run_simulation
from railway_simulator.core.parametric import (
    build_speed_scenarios,
    make_envelope_figure,
    run_parametric_envelope,
)
from railway_simulator.studies import parse_floats_csv
from railway_simulator.studies.numerics_sensitivity import run_numerics_sensitivity
from railway_simulator.studies.strain_rate_sensitivity import run_fixed_dif_sensitivity

# Import from other UI modules
from .export import to_excel
from .plotting import create_results_plots
from .sdof import (
    compute_building_sdof_response,
    compute_force_response_spectrum,
    compute_multi_damping_force_response_spectrum,
    create_building_animation,
    create_building_hysteresis_plot,
    create_building_response_plots,
    create_multi_damping_response_spectrum_plot,
    create_response_spectrum_plot,
)
from .train_geometry import create_train_geometry_plot


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



