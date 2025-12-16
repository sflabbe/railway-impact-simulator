"""
Streamlit UI for the Railway Impact Simulator

This file provides the main application entry point and delegates functionality
to specialized UI modules in the ui package.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
from railway_simulator.ui import (
    build_parameter_ui,
    display_citation,
    display_header,
    execute_simulation,
)


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
                use_container_width=True,
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
                        st.plotly_chart(fig_env, use_container_width=True)

                        st.markdown("#### Scenario summary")
                        st.dataframe(summary_df)
                except Exception as exc:
                    st.error(f"Parametric study failed: {exc}")

        # --------------------------
        # Numerics sensitivity
        # --------------------------
        # NOTE: No Streamlit caching is applied to run_simulation or run_numerics_sensitivity.
        # If caching is added in the future, ensure the cache key includes:
        # (dt, alpha_hht, newton_tol, max_iter, model params) to avoid stale results.
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
                    st.info(
                        "📊 **Baseline:** Relative errors are computed with respect to the finest "
                        "resolution case (smallest Δt, most negative α, tightest tolerance). "
                        "This shows numerical error vs. the most accurate result."
                    )
                    st.dataframe(summary_df)

                    # Simple peak plot
                    try:
                        # Use dt_requested if available (new), fallback to dt_s (old)
                        dt_col = "dt_requested" if "dt_requested" in summary_df.columns else "dt_s"
                        fig_peak = go.Figure()
                        fig_peak.add_trace(
                            go.Scatter(
                                x=summary_df[dt_col],
                                y=summary_df["peak_force_MN"],
                                mode="markers+lines",
                                name="Peak force (sampled)",
                                line=dict(dash="dot"),
                            )
                        )
                        # Add interpolated peak if available
                        if "peak_force_interp_MN" in summary_df.columns:
                            fig_peak.add_trace(
                                go.Scatter(
                                    x=summary_df[dt_col],
                                    y=summary_df["peak_force_interp_MN"],
                                    mode="markers+lines",
                                    name="Peak force (interpolated)",
                                )
                            )
                        fig_peak.update_layout(
                            title="Peak force vs Δt (aggregated over α/tol)",
                            xaxis_title="Δt (s)",
                            yaxis_title="Peak force (MN)",
                            height=350,
                        )
                        st.plotly_chart(fig_peak, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate peak plot: {e}")

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
                        st.plotly_chart(fig, use_container_width=True)

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
                        st.plotly_chart(fig_peak, use_container_width=True)
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
                        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------
    # ABOUT / DOCUMENTATION TAB
    # --------------------------------------------------------------
    with tab_about:
        display_header()
        display_citation()


if __name__ == "__main__":
    main()
