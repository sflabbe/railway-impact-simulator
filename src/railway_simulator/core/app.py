"""
Streamlit UI for the Railway Impact Simulator

This file provides the main application entry point and delegates functionality
to specialized UI modules in the ui package.
"""

from __future__ import annotations

from typing import Any, Dict

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from railway_simulator.ui.st_compat import safe_button, safe_download_button, safe_plotly_chart
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

from railway_simulator.ui.export import (
    make_bundle_zip,
    sanitize_filename,
    replot_matplotlib_script,
    to_excel,
    utc_timestamp,
)


def _cfg_speed_kmh(cfg: Dict[str, Any]) -> float | None:
    """Try to recover impact speed in km/h from a config dict."""
    try:
        v0 = float(cfg.get("v0_init"))
        return abs(v0) * 3.6
    except Exception:
        return None


def _compute_envelope_from_captured(
    captured_items: list[dict[str, Any]],
    *,
    quantity: str,
) -> pd.DataFrame | None:
    """Compute envelope + weighted mean directly from already captured runs."""
    if not captured_items:
        return None

    # Time axis (ms)
    first_df = captured_items[0].get("df")
    if not isinstance(first_df, pd.DataFrame):
        return None

    if "Time_ms" in first_df.columns:
        t_ms = first_df["Time_ms"].to_numpy()
    elif "Time_s" in first_df.columns:
        t_ms = first_df["Time_s"].to_numpy() * 1000.0
    else:
        t_ms = np.arange(len(first_df), dtype=float)

    values: list[np.ndarray] = []
    weights: list[float] = []
    for item in captured_items:
        df = item.get("df")
        if not isinstance(df, pd.DataFrame) or quantity not in df.columns:
            continue
        values.append(df[quantity].to_numpy())
        try:
            weights.append(float(item.get("weight", 1.0)))
        except Exception:
            weights.append(1.0)

    if not values:
        return None

    vals = np.vstack(values)
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        w = np.maximum(w, 0.0)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w[:] = 1.0 / len(w)

    env = np.nanmax(vals, axis=0)
    mean = np.average(vals, axis=0, weights=w)

    return pd.DataFrame(
        {
            "Time_ms": t_ms,
            f"{quantity}_envelope": env,
            f"{quantity}_weighted_mean": mean,
        }
    )


def _get_time_axis(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Extract time axis from dataframe in milliseconds.

    Args:
        df: DataFrame with either Time_ms or Time_s column

    Returns:
        Tuple of (time_values_ms, axis_label)
    """
    if "Time_ms" in df.columns:
        return df["Time_ms"].to_numpy(), "Time (ms)"
    elif "Time_s" in df.columns:
        return df["Time_s"].to_numpy() * 1000.0, "Time (ms)"
    else:
        # Fallback: assume time step from config
        return np.arange(len(df)) * 0.1, "Time (ms)"


def _plot_time_history_overlay(
    captured: list[tuple[dict, pd.DataFrame]],
    q_label: str,
    max_runs: int,
    label_fn,
    title: str = "Time histories (overlay)"
) -> go.Figure:
    """Create overlay plot of time histories from captured simulations.

    Args:
        captured: List of (config, dataframe) tuples from simulation runs
        q_label: Column name of quantity to plot
        max_runs: Maximum number of runs to include in overlay
        label_fn: Callable that takes (cfg, df, index) and returns label string
        title: Plot title

    Returns:
        Plotly figure with overlaid time histories
    """
    fig = go.Figure()
    shown = 0
    xlab = "Time (ms)"

    for idx, (cfg, df) in enumerate(captured):
        if shown >= int(max_runs):
            break
        if q_label not in df.columns:
            continue

        t, xlab = _get_time_axis(df)
        label = label_fn(cfg, df, idx)
        fig.add_trace(go.Scatter(x=t, y=df[q_label], mode="lines", name=label))
        shown += 1

    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=q_label,
        height=450,
    )
    return fig


def _stack_parametric_histories(
    captured: list[tuple[dict, pd.DataFrame]],
    *,
    scenario_name_fn,
    columns: list[str],
) -> pd.DataFrame:
    """Stack time histories from captured runs into a long-format table.

    Output columns:
      - scenario
      - Time_s, Time_ms (when available)
      - requested data columns
    """
    rows: list[pd.DataFrame] = []
    for idx, (cfg, df) in enumerate(captured):
        scen = str(scenario_name_fn(cfg, df, idx))
        keep_cols = [c for c in ("Time_s", "Time_ms") if c in df.columns]
        keep_cols += [c for c in columns if c in df.columns]
        if not keep_cols:
            continue
        out = df[keep_cols].copy()
        out.insert(0, "scenario", scen)
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _zip_parametric_runs(
    captured: list[tuple[dict, pd.DataFrame]],
    *,
    scenario_name_fn,
    prefix: str = "parametric",
) -> bytes:
    """Create a ZIP with one CSV per run + a small manifest."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest_rows: list[dict[str, Any]] = []
        for idx, (cfg, df) in enumerate(captured):
            scen = str(scenario_name_fn(cfg, df, idx))
            fname = f"{prefix}__{scen}.csv".replace(" ", "_")
            zf.writestr(fname, df.to_csv(index=False))
            manifest_rows.append(
                {
                    "scenario": scen,
                    "speed_kmh": _cfg_speed_kmh(cfg),
                    "dt_s": cfg.get("h_init"),
                    "d0_m": cfg.get("d0"),
                    **{k: cfg.get(k) for k in ("contact_model", "friction_model", "cr_wall", "mu_s", "mu_k")},
                }
            )

        if manifest_rows:
            man = pd.DataFrame(manifest_rows)
            zf.writestr(f"{prefix}__manifest.csv", man.to_csv(index=False))
    return buf.getvalue()


def _fig_to_html_bytes(fig: go.Figure) -> bytes:
    """Export Plotly figure as standalone HTML bytes (no kaleido required)."""
    return fig.to_html(include_plotlyjs="inline", full_html=True).encode("utf-8")


def _force_penetration_time_gradient(df: pd.DataFrame, *, title: str) -> go.Figure:
    """Force–penetration curve colored by time."""
    t, xlab = _get_time_axis(df)
    if "Penetration_mm" not in df.columns or "Impact_Force_MN" not in df.columns:
        return go.Figure()
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["Penetration_mm"],
                y=df["Impact_Force_MN"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=t,
                    showscale=True,
                    colorbar=dict(title=xlab),
                ),
                name="F-δ",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Penetration δ [mm]",
        yaxis_title="Impact force F [MN]",
        height=520,
    )
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        layout="wide",
        page_title="Railway Impact Simulator - DZSF Research",
    )

    # Sidebar parameter UI is shared across all tabs
    params = build_parameter_ui()

    tab_sim, tab_param, tab_about = st.tabs(
        ["Simulator", "Parametric Studies", "About / Documentation"]
    )

    # --------------------------------------------------------------
    # SIMULATOR TAB
    # --------------------------------------------------------------
    with tab_sim:
        st.title("Railway Impact Simulator")
        st.markdown("**HHT-α implicit integration with Bouc–Wen hysteresis**")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration")
            try:
                st.metric("Velocity", f"{-float(params['v0_init']) * 3.6:.1f} km/h")
            except (ValueError, KeyError, TypeError):
                st.metric("Velocity", "—")
            st.metric("Masses", int(params.get("n_masses", 0)))
            st.metric("Time Step", f"{float(params.get('h_init', 0.0))*1000:.2f} ms")
            st.metric("Initial Gap", f"{float(params.get('d0', 0.0)):.3f} m")
            if params.get("case_name"):
                st.caption(f"case_name: **{params['case_name']}**")
            st.markdown("---")
            run_btn = safe_button(st, 
                "**Run Simulation**",
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
                    "Configure parameters in the sidebar and press **Run Simulation**"
                )

    # --------------------------------------------------------------
    # PARAMETRIC STUDIES TAB
    # --------------------------------------------------------------
    with tab_param:
        st.title("Parametric Studies")
        st.markdown(
            "Run reproducible sweeps using the same configuration as the Simulator tab."
        )

        sub_env, sub_contact, sub_friction, sub_sens, sub_dif, sub_solver = st.tabs(
            [
                "Speed envelope",
                "Contact force models",
                "Friction / restitution",
                "Numerics sensitivity",
                "Strain-rate (DIF)",
                "Solver comparison",
            ]
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
            compare_solvers_env = st.checkbox(
                "Compare nonlinear solvers (Newton vs Picard)",
                value=False,
                key="compare_solvers_env",
                help="Runs the same speed envelope twice (solver=newton and solver=picard) and overlays the results.",
            )

            st.markdown("#### Plot options")
            show_speed_histories = st.checkbox(
                "Show per-speed histories (recommended)",
                value=True,
                key="env_show_speed_histories",
                help="Plots one time history per speed scenario. This makes it easier to read the influence of speed than a single envelope curve alone.",
            )
            show_envelope_curves = st.checkbox(
                "Show envelope + weighted mean", 
                value=True,
                key="env_show_envelope_curves",
            )


            if st.button("Run parametric envelope", type="primary", key="run_env"):
                try:
                    speeds = df_scenarios["speed_kmh"].astype(float).tolist()
                    weights = df_scenarios["weight"].astype(float).tolist()

                    if not speeds:
                        st.warning("Please define at least one scenario.")
                    else:
                        base_params = dict(params)

                        if compare_solvers_env:
                            solvers = ["newton", "picard"]
                            env_by_solver: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

                            prog = st.progress(0.0)
                            total = len(solvers)
                            for i, solver in enumerate(solvers, start=1):
                                base_i = dict(base_params)
                                base_i["solver"] = solver
                                scenarios_i = build_speed_scenarios(
                                    base_i,
                                    speeds_kmh=speeds,
                                    weights=weights,
                                    prefix=f"{solver[0]}v",
                                )
                                envelope_df_i, summary_df_i, _ = run_parametric_envelope(
                                    scenarios_i, quantity=quantity
                                )
                                summary_df_i = summary_df_i.copy()
                                summary_df_i["solver"] = solver
                                env_by_solver[solver] = (envelope_df_i, summary_df_i)
                                prog.progress(i / total)

                            fig = go.Figure()
                            for solver, (env_df, _) in env_by_solver.items():
                                fig.add_trace(
                                    go.Scatter(
                                        x=env_df["Time_ms"],
                                        y=env_df[f"{quantity}_envelope"],
                                        mode="lines",
                                        name=f"Envelope ({solver})",
                                    )
                                )
                                fig.add_trace(
                                    go.Scatter(
                                        x=env_df["Time_ms"],
                                        y=env_df[f"{quantity}_weighted_mean"],
                                        mode="lines",
                                        name=f"Weighted mean ({solver})",
                                        line=dict(dash="dot"),
                                    )
                                )

                            fig.update_layout(
                                title=f"{quantity_label} – envelope (Newton vs Picard)",
                                xaxis_title="Time (ms)",
                                yaxis_title=quantity,
                                height=520,
                            )
                            safe_plotly_chart(st, fig, width="stretch")

                            st.markdown("#### Scenario summary (both solvers)")
                            summary_all = pd.concat(
                                [v[1] for v in env_by_solver.values()],
                                ignore_index=True,
                            )
                            try:
                                pivot = summary_all.pivot_table(
                                    index=["scenario", "speed_kmh"],
                                    columns="solver",
                                    values="peak",
                                    aggfunc="first",
                                )
                                if "newton" in pivot.columns and "picard" in pivot.columns:
                                    pivot = pivot.reset_index()
                                    pivot["peak_diff_pct_picard_vs_newton"] = 100.0 * (
                                        (pivot["picard"] - pivot["newton"]) / pivot["newton"]
                                    )
                                    st.dataframe(pivot)
                                else:
                                    st.dataframe(summary_all)
                            except Exception:
                                st.dataframe(summary_all)

                        else:
                            scenarios = build_speed_scenarios(
                                base_params,
                                speeds_kmh=speeds,
                                weights=weights,
                                prefix="v",
                            )

                            envelope_df, summary_df, meta = run_parametric_envelope(
                                scenarios, quantity=quantity
                            )

                            # Persist so the user can tweak plot options without re-running
                            st.session_state["env_last"] = {
                                "quantity": quantity,
                                "quantity_label": quantity_label,
                                "envelope_df": envelope_df,
                                "summary_df": summary_df,
                                "captured": meta.get("captured", []),
                                "speeds": speeds,
                                "weights": weights,
                                "compare_solvers": False,
                            }
                except Exception as exc:
                    st.error(f"Parametric study failed: {exc}")

            # Render last result (if available)
            last = st.session_state.get("env_last", None)
            if last and not last.get("compare_solvers", False):
                envelope_df = last["envelope_df"]
                summary_df = last["summary_df"]
                captured = last.get("captured", [])
                q = str(last.get("quantity", quantity))
                q_label = str(last.get("quantity_label", quantity_label))

                if show_speed_histories and captured:
                    fig_speed = go.Figure()
                    for item in captured:
                        df_i = item.get("df", None)
                        meta_i = item.get("meta", {}) or {}
                        v_kmh = meta_i.get("speed_kmh", None)
                        name = item.get("scenario", "scenario")
                        label = f"{v_kmh:.0f} km/h" if isinstance(v_kmh, (int, float)) else name
                        if isinstance(df_i, pd.DataFrame) and q in df_i.columns:
                            t_ms, _ = _get_time_axis(df_i)
                            fig_speed.add_trace(
                                go.Scatter(x=t_ms, y=df_i[q], mode="lines", name=label)
                            )

                    fig_speed.update_layout(
                        title=f"{q_label} – per-speed histories",
                        xaxis_title="Time (ms)",
                        yaxis_title=q,
                        height=520,
                    )
                    safe_plotly_chart(st, fig_speed, width="stretch")

                    safe_download_button(
                        st,
                        label="Export per-speed overlay (HTML)",
                        data=_fig_to_html_bytes(fig_speed),
                        file_name="speed_histories_overlay.html",
                        mime="text/html",
                        use_container_width=True,
                        key="env_export_speed_overlay_html",
                    )

                # ---------
                # Export bundle (ZIP)
                # ---------
                st.markdown("#### Export study bundle")
                st.caption(
                    "Exports raw runs, summary tables and plots (HTML) in a single ZIP. "
                    "Files are named by speed to make it easy to replace dissertation figures."
                )

                try:
                    # Build derived envelopes for the three canonical quantities
                    env_force = _compute_envelope_from_captured(captured, quantity="Impact_Force_MN")
                    env_acc = _compute_envelope_from_captured(captured, quantity="Acceleration_g")
                    env_pen = _compute_envelope_from_captured(captured, quantity="Penetration_mm")

                    def _per_speed_overlay(qcol: str, title: str, ylab: str) -> go.Figure | None:
                        fig = go.Figure()
                        shown = 0
                        for item in captured:
                            df_i = item.get("df")
                            meta_i = item.get("meta", {}) or {}
                            if not isinstance(df_i, pd.DataFrame) or qcol not in df_i.columns:
                                continue
                            v_kmh = meta_i.get("speed_kmh", None)
                            label = f"{float(v_kmh):.0f} km/h" if isinstance(v_kmh, (int, float)) else str(item.get("scenario", "scenario"))
                            t_ms, _ = _get_time_axis(df_i)
                            fig.add_trace(go.Scatter(x=t_ms, y=df_i[qcol], mode="lines", name=label))
                            shown += 1
                        if shown == 0:
                            return None
                        fig.update_layout(title=title, xaxis_title="Time (ms)", yaxis_title=ylab, height=520)
                        return fig

                    figs: dict[str, bytes] = {}
                    for qcol, title, ylab in (
                        ("Impact_Force_MN", "Impact force – per-speed histories", "Impact force (MN)"),
                        ("Acceleration_g", "Acceleration – per-speed histories", "Acceleration (g)"),
                        ("Penetration_mm", "Penetration – per-speed histories", "Penetration (mm)"),
                    ):
                        fig_i = _per_speed_overlay(qcol, title, ylab)
                        if fig_i is not None:
                            figs[f"per_speed__{qcol}"] = _fig_to_html_bytes(fig_i)

                    # Envelopes plots (if available)
                    def _env_fig(env_df: pd.DataFrame | None, qcol: str, title: str, ylab: str) -> go.Figure | None:
                        if env_df is None or env_df.empty:
                            return None
                        env_col = f"{qcol}_envelope"
                        mean_col = f"{qcol}_weighted_mean"
                        if env_col not in env_df.columns:
                            return None
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=env_df["Time_ms"], y=env_df[env_col], mode="lines", name="Envelope"))
                        if mean_col in env_df.columns:
                            fig.add_trace(go.Scatter(x=env_df["Time_ms"], y=env_df[mean_col], mode="lines", name="Weighted mean", line=dict(dash="dot")))
                        fig.update_layout(title=title, xaxis_title="Time (ms)", yaxis_title=ylab, height=520)
                        return fig

                    for env_df, qcol, title, ylab in (
                        (env_force, "Impact_Force_MN", "Impact force – envelope + weighted mean", "Impact force (MN)"),
                        (env_acc, "Acceleration_g", "Acceleration – envelope + weighted mean", "Acceleration (g)"),
                        (env_pen, "Penetration_mm", "Penetration – envelope + weighted mean", "Penetration (mm)"),
                    ):
                        fig_env_i = _env_fig(env_df, qcol, title, ylab)
                        if fig_env_i is not None:
                            figs[f"envelope__{qcol}"] = _fig_to_html_bytes(fig_env_i)

                    # Runs map: name files by speed for clarity
                    runs: dict[str, pd.DataFrame] = {}
                    for item in captured:
                        df_i = item.get("df")
                        meta_i = item.get("meta", {}) or {}
                        scen = str(item.get("scenario", "run"))
                        v_kmh = meta_i.get("speed_kmh", None)
                        if isinstance(v_kmh, (int, float)):
                            scen_name = f"{scen}__{float(v_kmh):.0f}kmh"
                        else:
                            scen_name = scen
                        runs[sanitize_filename(scen_name)] = df_i

                    # Stacked histories
                    # (scenario, speed_kmh, time, key quantities)
                    long_rows: list[pd.DataFrame] = []
                    for item in captured:
                        df_i = item.get("df")
                        meta_i = item.get("meta", {}) or {}
                        if not isinstance(df_i, pd.DataFrame):
                            continue
                        keep = [c for c in ("Time_ms", "Time_s", "Impact_Force_MN", "Acceleration_g", "Penetration_mm") if c in df_i.columns]
                        if not keep:
                            continue
                        out = df_i[keep].copy()
                        out.insert(0, "speed_kmh", meta_i.get("speed_kmh", np.nan))
                        out.insert(0, "scenario", str(item.get("scenario", "scenario")))
                        long_rows.append(out)
                    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

                    metadata = {
                        "study": "speed_envelope",
                        "speeds_kmh": last.get("speeds"),
                        "weights": last.get("weights"),
                        "base_quantity": q,
                        "base_quantity_label": q_label,
                        "base_speed_kmh": _cfg_speed_kmh(params),
                        "params": {k: v for k, v in params.items() if k not in ("train", "materials")},
                    }
                    bundle = make_bundle_zip(
                        study="speed_envelope",
                        metadata=metadata,
                        dataframes={
                            "summary": summary_df,
                            "histories_long": long_df,
                            "envelope_force": env_force if env_force is not None else pd.DataFrame(),
                            "envelope_acc": env_acc if env_acc is not None else pd.DataFrame(),
                            "envelope_pen": env_pen if env_pen is not None else pd.DataFrame(),
                        },
                        plots_html=figs,
                        runs=runs,
                        extra_files={
                            "replot_matplotlib.py": replot_matplotlib_script(),
                            "README.txt": (
                                "Speed envelope (Umhüllende) export bundle\n"
                                "\n"
                                "Contents:\n"
                                "- data/: summary, stacked histories, envelopes\n"
                                "- runs/: one CSV per speed scenario\n"
                                "- plots/: Plotly HTML overlays/envelopes\n"
                            )
                        },
                    )

                    safe_download_button(
                        st,
                        label="Export speed-envelope bundle (ZIP)",
                        data=bundle,
                        file_name=f"speed_envelope__{sanitize_filename(str(q))}__{utc_timestamp()}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="env_export_bundle_zip",
                    )
                except Exception as exc:
                    st.warning(f"Bundle export not available: {exc}")

                if show_envelope_curves:
                    fig_env = go.Figure()
                    env_col = f"{q}_envelope" if f"{q}_envelope" in envelope_df.columns else q
                    mean_col = f"{q}_weighted_mean" if f"{q}_weighted_mean" in envelope_df.columns else None
                    fig_env.add_trace(
                        go.Scatter(
                            x=envelope_df["Time_ms"],
                            y=envelope_df[env_col],
                            mode="lines",
                            name="Envelope",
                        )
                    )
                    if mean_col:
                        fig_env.add_trace(
                            go.Scatter(
                                x=envelope_df["Time_ms"],
                                y=envelope_df[mean_col],
                                mode="lines",
                                name="Weighted mean",
                                line=dict(dash="dot"),
                            )
                        )
                    fig_env.update_layout(
                        title=f"{q_label} – envelope over defined speeds",
                        xaxis_title="Time (ms)",
                        yaxis_title=q,
                        height=520,
                    )
                    safe_plotly_chart(st, fig_env, width="stretch")

                st.markdown("#### Scenario summary")
                st.dataframe(summary_df)

        # --------------------------
        # Contact force model sweep
        # --------------------------
        with sub_contact:
            st.markdown("### Contact force model sweep")
            st.write(
                "Compare different **contact force laws** (e.g. Anagnostopoulos / Flores / Ye) "
                "with the same base configuration. The tool overlays key time histories and "
                "lets you export the results."
            )

            left, right = st.columns([1, 2])

            with left:
                default_models = ["anagnostopoulos", "flores", "ye"]
                all_models = [
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
                models = st.multiselect(
                    "Contact models",
                    options=all_models,
                    default=[m for m in default_models if m in all_models],
                    help="Runs one simulation per selected contact model.",
                    key="contact_sweep_models",
                )

                max_runs = st.number_input(
                    "Max runs to plot (overlay)",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    key="contact_sweep_max_runs",
                )

                show_force = st.checkbox("Show force–time", value=True, key="contact_sweep_show_force")
                show_acc = st.checkbox("Show acceleration–time", value=True, key="contact_sweep_show_acc")
                show_pen = st.checkbox("Show penetration–time", value=True, key="contact_sweep_show_pen")
                show_fd = st.checkbox("Show force–penetration (time gradient)", value=True, key="contact_sweep_show_fd")

                run_contact = st.button(
                    "Run contact model sweep",
                    type="primary",
                    key="run_contact_sweep",
                    use_container_width=True,
                    disabled=(len(models) == 0),
                )

            with right:
                if run_contact:
                    try:
                        captured: list[tuple[dict, pd.DataFrame]] = []

                        prog = st.progress(0.0)
                        total = max(1, len(models))

                        for i, model in enumerate(models, start=1):
                            cfg = dict(params)
                            cfg["contact_model"] = model
                            # make it easier to identify in exports
                            cfg["case_name"] = f"contact_model={model}"
                            df_i = run_simulation(cfg)
                            captured.append((cfg, df_i))
                            prog.progress(i / total)

                        st.session_state["contact_sweep_captured"] = captured

                    except Exception as exc:
                        st.error(f"Contact model sweep failed: {exc}")

                captured = st.session_state.get("contact_sweep_captured", None)
                if not captured:
                    st.info("Select contact models and run the sweep to see results.")
                else:
                    def scen_name(cfg, df, idx):
                        return str(cfg.get("contact_model", f"run{idx+1}"))

                    # Summary
                    rows = []
                    for idx, (cfg, df_i) in enumerate(captured):
                        name = scen_name(cfg, df_i, idx)
                        row = {"scenario": name}
                        if "Impact_Force_MN" in df_i.columns:
                            q = df_i["Impact_Force_MN"].to_numpy()
                            row["peak_force_MN"] = float(np.nanmax(q))
                            k = int(np.nanargmax(q))
                            if "Time_ms" in df_i.columns:
                                row["t_peak_ms"] = float(df_i["Time_ms"].iloc[k])
                            elif "Time_s" in df_i.columns:
                                row["t_peak_ms"] = 1000.0 * float(df_i["Time_s"].iloc[k])
                        if "Acceleration_g" in df_i.columns:
                            row["max_acc_g"] = float(np.nanmax(df_i["Acceleration_g"]))
                        if "Penetration_mm" in df_i.columns:
                            row["max_pen_mm"] = float(np.nanmax(df_i["Penetration_mm"]))
                        # Friction diagnostics (helps verify μ is actually being used)
                        if "E_diss_friction_J" in df_i.columns:
                            try:
                                row["E_diss_friction_J_end"] = float(df_i["E_diss_friction_J"].iloc[-1])
                            except Exception:
                                pass

                        # Max friction force magnitude over all masses/time
                        try:
                            n_m = int(getattr(df_i, 'attrs', {}).get('n_masses', cfg.get('n_masses', 0)) or 0)
                            max_f = 0.0
                            for mi in range(1, max(n_m, 1) + 1):
                                fx_col = f"Mass{mi}_Force_friction_x_N"
                                fy_col = f"Mass{mi}_Force_friction_y_N"
                                if fx_col in df_i.columns and fy_col in df_i.columns:
                                    fx = df_i[fx_col].to_numpy()
                                    fy = df_i[fy_col].to_numpy()
                                    mag = (fx*fx + fy*fy) ** 0.5
                                    mmax = float(np.nanmax(mag))
                                    if mmax > max_f:
                                        max_f = mmax
                            if max_f > 0.0:
                                row["max_fric_kN"] = max_f / 1e3
                        except Exception:
                            pass
                        if "E_balance_error_J" in df_i.columns:
                            row["max_EB_error_J"] = float(np.nanmax(np.abs(df_i["E_balance_error_J"])))
                        rows.append(row)
                    summary_df = pd.DataFrame(rows)

                    st.markdown("#### Summary")
                    st.dataframe(summary_df, use_container_width=True)

                    if 'E_diss_friction_J_end' in summary_df.columns:
                        try:
                            if float(summary_df['E_diss_friction_J_end'].max()) < 1e-6:
                                st.warning("Friction appears inactive (E_diss_friction_J ~ 0). If you selected LuGre/Dahl, ensure sigma_0>0; also note that with impact angle=0° wall-friction may have little influence on normal force.")
                        except Exception:
                            pass

                    # Overlays
                    st.markdown("#### Overlays")
                    if show_force and any("Impact_Force_MN" in d.columns for _, d in captured):
                        figF = _plot_time_history_overlay(
                            captured,
                            "Impact_Force_MN",
                            int(max_runs),
                            lambda cfg, df, idx: scen_name(cfg, df, idx),
                            title="Impact force – time (overlay)",
                        )
                        safe_plotly_chart(st, figF, width="stretch")
                        safe_download_button(
                            st,
                            "Export force overlay (HTML)",
                            _fig_to_html_bytes(figF),
                            "contact_sweep_force_overlay.html",
                            "text/html",
                            use_container_width=True,
                        )

                    if show_acc and any("Acceleration_g" in d.columns for _, d in captured):
                        figA = _plot_time_history_overlay(
                            captured,
                            "Acceleration_g",
                            int(max_runs),
                            lambda cfg, df, idx: scen_name(cfg, df, idx),
                            title="Acceleration – time (overlay)",
                        )
                        safe_plotly_chart(st, figA, width="stretch")
                        safe_download_button(
                            st,
                            "Export acceleration overlay (HTML)",
                            _fig_to_html_bytes(figA),
                            "contact_sweep_acc_overlay.html",
                            "text/html",
                            use_container_width=True,
                        )

                    if show_pen and any("Penetration_mm" in d.columns for _, d in captured):
                        figP = _plot_time_history_overlay(
                            captured,
                            "Penetration_mm",
                            int(max_runs),
                            lambda cfg, df, idx: scen_name(cfg, df, idx),
                            title="Penetration – time (overlay)",
                        )
                        safe_plotly_chart(st, figP, width="stretch")
                        safe_download_button(
                            st,
                            "Export penetration overlay (HTML)",
                            _fig_to_html_bytes(figP),
                            "contact_sweep_pen_overlay.html",
                            "text/html",
                            use_container_width=True,
                        )

                    if show_fd and any(("Penetration_mm" in d.columns and "Impact_Force_MN" in d.columns) for _, d in captured):
                        scen_opts = [scen_name(c, d, i) for i, (c, d) in enumerate(captured)]
                        pick = st.selectbox("Force–penetration view", scen_opts, index=0, key="contact_sweep_fd_pick")
                        j = scen_opts.index(pick)
                        cfg_j, df_j = captured[j]
                        figFD = _force_penetration_time_gradient(df_j, title=f"Force–penetration (time gradient) – {pick}")
                        safe_plotly_chart(st, figFD, width="stretch")
                        safe_download_button(
                            st,
                            "Export force–penetration (HTML)",
                            _fig_to_html_bytes(figFD),
                            f"contact_sweep_force_penetration__{pick}.html".replace(" ", "_"),
                            "text/html",
                            use_container_width=True,
                        )

                    # Exports (data)
                    st.markdown("#### Export data")
                    c1, c2, c3 = st.columns(3)

                    safe_download_button(
                        c1,
                        "CSV summary",
                        summary_df.to_csv(index=False).encode(),
                        "contact_sweep_summary.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                    safe_download_button(
                        c2,
                        "XLSX summary",
                        to_excel(summary_df),
                        "contact_sweep_summary.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

                    long_df = _stack_parametric_histories(
                        captured,
                        scenario_name_fn=scen_name,
                        columns=["Impact_Force_MN", "Acceleration_g", "Penetration_mm"],
                    )
                    safe_download_button(
                        c3,
                        "ZIP raw runs (CSV)",
                        _zip_parametric_runs(captured, scenario_name_fn=scen_name, prefix="contact_sweep"),
                        "contact_sweep_runs.zip",
                        "application/zip",
                        use_container_width=True,
                    )

                    if not long_df.empty:
                        d1, d2 = st.columns(2)
                        safe_download_button(
                            d1,
                            "CSV stacked histories",
                            long_df.to_csv(index=False).encode(),
                            "contact_sweep_histories_long.csv",
                            "text/csv",
                            use_container_width=True,
                        )
                        safe_download_button(
                            d2,
                            "XLSX stacked histories",
                            to_excel(long_df),
                            "contact_sweep_histories_long.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )

                    # One-click bundle: raw runs + tables + plots (HTML)
                    st.markdown("#### Export bundle (paper-friendly)")
                    st.caption(
                        "Creates a single ZIP containing: raw runs, summary tables, stacked histories, "
                        "and the key overlay plots. Filenames include the current speed for traceability."
                    )
                    try:
                        plots: dict[str, bytes] = {}

                        # Always include the canonical overlays if data exists
                        if any("Impact_Force_MN" in d.columns for _, d in captured):
                            figF2 = _plot_time_history_overlay(
                                captured,
                                "Impact_Force_MN",
                                int(max_runs),
                                lambda cfg, df, idx: scen_name(cfg, df, idx),
                                title="Impact force – time (overlay)",
                            )
                            plots["overlay__Impact_Force_MN"] = _fig_to_html_bytes(figF2)

                        if any("Acceleration_g" in d.columns for _, d in captured):
                            figA2 = _plot_time_history_overlay(
                                captured,
                                "Acceleration_g",
                                int(max_runs),
                                lambda cfg, df, idx: scen_name(cfg, df, idx),
                                title="Acceleration – time (overlay)",
                            )
                            plots["overlay__Acceleration_g"] = _fig_to_html_bytes(figA2)

                        if any("Penetration_mm" in d.columns for _, d in captured):
                            figP2 = _plot_time_history_overlay(
                                captured,
                                "Penetration_mm",
                                int(max_runs),
                                lambda cfg, df, idx: scen_name(cfg, df, idx),
                                title="Penetration – time (overlay)",
                            )
                            plots["overlay__Penetration_mm"] = _fig_to_html_bytes(figP2)

                        # Force–penetration plots for each scenario (time gradient)
                        for i, (cfg_i, df_i) in enumerate(captured[: int(max_runs)]):
                            if not (
                                isinstance(df_i, pd.DataFrame)
                                and "Impact_Force_MN" in df_i.columns
                                and "Penetration_mm" in df_i.columns
                            ):
                                continue
                            name_i = sanitize_filename(scen_name(cfg_i, df_i, i))
                            plots[f"force_penetration__{name_i}"] = _fig_to_html_bytes(
                                _force_penetration_time_gradient(
                                    df_i, title=f"Force–penetration (time gradient) – {name_i}"
                                )
                            )

                        runs_map: dict[str, pd.DataFrame] = {}
                        for i, (cfg_i, df_i) in enumerate(captured):
                            name_i = sanitize_filename(scen_name(cfg_i, df_i, i))
                            sp = _cfg_speed_kmh(cfg_i)
                            if isinstance(sp, (int, float)):
                                name_i = f"v{float(sp):.0f}kmh__{name_i}"
                            runs_map[name_i] = df_i

                        meta = {
                            "study": "contact_force_models",
                            "base_speed_kmh": _cfg_speed_kmh(params),
                            "params": {k: v for k, v in params.items() if k not in ("train", "materials")},
                        }

                        bundle = make_bundle_zip(
                            study="contact_force_models",
                            metadata=meta,
                            dataframes={
                                "summary": summary_df,
                                "histories_long": long_df,
                            },
                            plots_html=plots,
                            runs=runs_map,
                            extra_files={
                                "replot_matplotlib.py": replot_matplotlib_script(),
                                "README.txt": (
                                    "Contact force model sweep export bundle\n\n"
                                    "- data/: summary + stacked histories\n"
                                    "- runs/: raw time histories (one CSV per model)\n"
                                    "- plots/: overlay plots + force-penetration (time gradient)\n"
                                )
                            },
                        )
                        safe_download_button(
                            st,
                            "Export contact-model bundle (ZIP)",
                            bundle,
                            f"contact_models__v{_cfg_speed_kmh(params) or 0:.0f}kmh__{utc_timestamp()}.zip",
                            "application/zip",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.warning(f"Bundle export not available: {exc}")

        # --------------------------
        # Friction / restitution sweep
        # --------------------------
        with sub_friction:
            st.markdown("### Friction / restitution sweep")
            st.write(
                "Vary **cr** (coefficient of restitution) or friction parameters (μ, model) "
                "and compare impact force/acceleration/penetration histories."
            )

            left, right = st.columns([1, 2])
            with left:
                sweep_kind = st.selectbox(
                    "Sweep parameter",
                    [
                        "cr_wall (restitution)",
                        "mu (set mu_s = mu_k)",
                        "mu_s (static)",
                        "mu_k (kinetic)",
                        "friction_model (categorical)",
                    ],
                    index=0,
                    key="fric_sweep_kind",
                )

                if sweep_kind == "friction_model (categorical)":
                    fr_models = st.multiselect(
                        "Friction models",
                        options=["lugre", "dahl", "coulomb", "brown-mcphee"],
                        default=["lugre", "coulomb"],
                        key="fric_sweep_models",
                    )
                    values_str = ""
                else:
                    if sweep_kind.startswith("cr_wall"):
                        values_str = st.text_input("Values", value="0.6,0.8,0.9", key="fric_sweep_values")
                    else:
                        values_str = st.text_input("Values", value="0.2,0.4,0.6", key="fric_sweep_values")
                    fr_models = []

                max_runs = st.number_input(
                    "Max runs to plot (overlay)",
                    min_value=1,
                    max_value=60,
                    value=12,
                    step=1,
                    key="fric_sweep_max_runs",
                )

                show_force = st.checkbox("Show force–time", value=True, key="fric_sweep_show_force")
                show_acc = st.checkbox("Show acceleration–time", value=True, key="fric_sweep_show_acc")
                show_pen = st.checkbox("Show penetration–time", value=True, key="fric_sweep_show_pen")
                show_fd = st.checkbox("Show force–penetration (time gradient)", value=False, key="fric_sweep_show_fd")

                disable_run = False
                if sweep_kind == "friction_model (categorical)" and not fr_models:
                    disable_run = True
                if sweep_kind != "friction_model (categorical)" and not values_str.strip():
                    disable_run = True

                run_fric = st.button(
                    "Run friction/restitution sweep",
                    type="primary",
                    key="run_fric_sweep",
                    use_container_width=True,
                    disabled=disable_run,
                )

            with right:
                if run_fric:
                    try:
                        captured: list[tuple[dict, pd.DataFrame]] = []

                        if sweep_kind == "friction_model (categorical)":
                            scenarios = [("friction_model", m) for m in fr_models]
                        else:
                            vals = parse_floats_csv(values_str)
                            if sweep_kind.startswith("cr_wall"):
                                scenarios = [("cr_wall", float(v)) for v in vals]
                            elif sweep_kind.startswith("mu (set"):
                                scenarios = [("mu", float(v)) for v in vals]
                            elif sweep_kind.startswith("mu_s"):
                                scenarios = [("mu_s", float(v)) for v in vals]
                            else:
                                scenarios = [("mu_k", float(v)) for v in vals]

                        prog = st.progress(0.0)
                        total = max(1, len(scenarios))
                        for i, (k, v) in enumerate(scenarios, start=1):
                            cfg = dict(params)
                            if k == "friction_model":
                                cfg["friction_model"] = str(v)
                                name = f"friction_model={v}"

                                # If we are injecting LuGre into a categorical sweep, make sure
                                # the default is the stable, paper-grade calibration.
                                #
                                # Without this, LuGre runs can inherit lugre_paper_grade=False from
                                # the current UI state if the user selected a non-LuGre base model.
                                if str(v).strip().lower() == "lugre":
                                    cfg["lugre_paper_grade"] = True
                                    if not cfg.get("lugre_bristle_deflection_m"):
                                        cfg["lugre_bristle_deflection_m"] = 1.0e-4
                            elif k == "cr_wall":
                                cfg["cr_wall"] = float(v)
                                name = f"cr={float(v):.3g}"
                            elif k == "mu":
                                cfg["mu_s"] = float(v)
                                cfg["mu_k"] = float(v)
                                name = f"mu={float(v):.3g}"
                            else:
                                cfg[k] = float(v)
                                name = f"{k}={float(v):.3g}"

                            cfg["case_name"] = name

                            # --- Ensure friction model parameters are actually active ---
                            # LuGre/Dahl require sigma_0 > 0 (and sigma_1 for LuGre) to build up
                            # friction within the short impact time window. If the base UI params
                            # left them at zero, the sweep will appear to have "no effect".
                            fm = str(cfg.get("friction_model", "none")).lower()
                            mu_s = float(cfg.get("mu_s", 0.0) or 0.0)
                            mu_k = float(cfg.get("mu_k", 0.0) or 0.0)
                            if fm in ("lugre", "dahl") and (abs(mu_s) > 1e-12 or abs(mu_k) > 1e-12):
                                if abs(float(cfg.get("sigma_0", 0.0) or 0.0)) < 1e-12:
                                    cfg["sigma_0"] = 1.0e6  # N/m (fast bristle build-up)
                                if fm == "lugre" and abs(float(cfg.get("sigma_1", 0.0) or 0.0)) < 1e-12:
                                    cfg["sigma_1"] = 1.0e3  # N·s/m (damping)

                            df_i = run_simulation(cfg)
                            captured.append((cfg, df_i))
                            prog.progress(i / total)

                        st.session_state["fric_sweep_captured"] = captured
                        st.session_state["fric_sweep_kind_last"] = sweep_kind

                    except Exception as exc:
                        st.error(f"Friction/restitution sweep failed: {exc}")

                captured = st.session_state.get("fric_sweep_captured", None)
                if not captured:
                    st.info("Configure a sweep and run it to see results.")
                else:
                    def scen_name(cfg, df, idx):
                        # Prefer case_name if we set it
                        return str(cfg.get("case_name", f"run{idx+1}"))

                    # Summary
                    rows = []
                    for idx, (cfg, df_i) in enumerate(captured):
                        name = scen_name(cfg, df_i, idx)
                        row = {"scenario": name}
                        for k in ("cr_wall", "mu_s", "mu_k", "friction_model", "sigma_0", "sigma_1", "sigma_2"):
                            if k in cfg:
                                row[k] = cfg.get(k)
                        if "Impact_Force_MN" in df_i.columns:
                            q = df_i["Impact_Force_MN"].to_numpy()
                            row["peak_force_MN"] = float(np.nanmax(q))
                            kpk = int(np.nanargmax(q))
                            if "Time_ms" in df_i.columns:
                                row["t_peak_ms"] = float(df_i["Time_ms"].iloc[kpk])
                            elif "Time_s" in df_i.columns:
                                row["t_peak_ms"] = 1000.0 * float(df_i["Time_s"].iloc[kpk])
                        if "Acceleration_g" in df_i.columns:
                            row["max_acc_g"] = float(np.nanmax(df_i["Acceleration_g"]))
                        if "Penetration_mm" in df_i.columns:
                            row["max_pen_mm"] = float(np.nanmax(df_i["Penetration_mm"]))
                        # Friction diagnostics (helps verify μ is actually being used)
                        if "E_diss_friction_J" in df_i.columns:
                            try:
                                row["E_diss_friction_J_end"] = float(df_i["E_diss_friction_J"].iloc[-1])
                            except Exception:
                                pass

                        # Max friction force magnitude over all masses/time
                        try:
                            n_m = int(getattr(df_i, 'attrs', {}).get('n_masses', cfg.get('n_masses', 0)) or 0)
                            max_f = 0.0
                            for mi in range(1, max(n_m, 1) + 1):
                                fx_col = f"Mass{mi}_Force_friction_x_N"
                                fy_col = f"Mass{mi}_Force_friction_y_N"
                                if fx_col in df_i.columns and fy_col in df_i.columns:
                                    fx = df_i[fx_col].to_numpy()
                                    fy = df_i[fy_col].to_numpy()
                                    mag = (fx*fx + fy*fy) ** 0.5
                                    mmax = float(np.nanmax(mag))
                                    if mmax > max_f:
                                        max_f = mmax
                            if max_f > 0.0:
                                row["max_fric_kN"] = max_f / 1e3
                        except Exception:
                            pass
                        rows.append(row)
                    summary_df = pd.DataFrame(rows)

                    st.markdown("#### Summary")
                    st.dataframe(summary_df, use_container_width=True)

                    if 'E_diss_friction_J_end' in summary_df.columns:
                        try:
                            if float(summary_df['E_diss_friction_J_end'].max()) < 1e-6:
                                st.warning("Friction appears inactive (E_diss_friction_J ~ 0). If you selected LuGre/Dahl, ensure sigma_0>0; also note that with impact angle=0° wall-friction may have little influence on normal force.")
                        except Exception:
                            pass

                    st.markdown("#### Overlays")
                    if show_force and any("Impact_Force_MN" in d.columns for _, d in captured):
                        figF = _plot_time_history_overlay(
                            captured,
                            "Impact_Force_MN",
                            int(max_runs),
                            lambda cfg, df, idx: scen_name(cfg, df, idx),
                            title="Impact force – time (overlay)",
                        )
                        safe_plotly_chart(st, figF, width="stretch")
                        safe_download_button(
                            st,
                            "Export force overlay (HTML)",
                            _fig_to_html_bytes(figF),
                            "fric_sweep_force_overlay.html",
                            "text/html",
                            use_container_width=True,
                        )

                    if show_acc and any("Acceleration_g" in d.columns for _, d in captured):
                        figA = _plot_time_history_overlay(
                            captured,
                            "Acceleration_g",
                            int(max_runs),
                            lambda cfg, df, idx: scen_name(cfg, df, idx),
                            title="Acceleration – time (overlay)",
                        )
                        safe_plotly_chart(st, figA, width="stretch")
                        safe_download_button(
                            st,
                            "Export acceleration overlay (HTML)",
                            _fig_to_html_bytes(figA),
                            "fric_sweep_acc_overlay.html",
                            "text/html",
                            use_container_width=True,
                        )

                    if show_pen and any("Penetration_mm" in d.columns for _, d in captured):
                        figP = _plot_time_history_overlay(
                            captured,
                            "Penetration_mm",
                            int(max_runs),
                            lambda cfg, df, idx: scen_name(cfg, df, idx),
                            title="Penetration – time (overlay)",
                        )
                        safe_plotly_chart(st, figP, width="stretch")
                        safe_download_button(
                            st,
                            "Export penetration overlay (HTML)",
                            _fig_to_html_bytes(figP),
                            "fric_sweep_pen_overlay.html",
                            "text/html",
                            use_container_width=True,
                        )

                    if show_fd and any(("Penetration_mm" in d.columns and "Impact_Force_MN" in d.columns) for _, d in captured):
                        scen_opts = [scen_name(c, d, i) for i, (c, d) in enumerate(captured)]
                        pick = st.selectbox("Force–penetration view", scen_opts, index=0, key="fric_sweep_fd_pick")
                        j = scen_opts.index(pick)
                        cfg_j, df_j = captured[j]
                        figFD = _force_penetration_time_gradient(df_j, title=f"Force–penetration (time gradient) – {pick}")
                        safe_plotly_chart(st, figFD, width="stretch")
                        safe_download_button(
                            st,
                            "Export force–penetration (HTML)",
                            _fig_to_html_bytes(figFD),
                            f"fric_sweep_force_penetration__{pick}.html".replace(" ", "_"),
                            "text/html",
                            use_container_width=True,
                        )

                    st.markdown("#### Export data")
                    c1, c2, c3 = st.columns(3)
                    safe_download_button(
                        c1,
                        "CSV summary",
                        summary_df.to_csv(index=False).encode(),
                        "fric_sweep_summary.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                    safe_download_button(
                        c2,
                        "XLSX summary",
                        to_excel(summary_df),
                        "fric_sweep_summary.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                    safe_download_button(
                        c3,
                        "ZIP raw runs (CSV)",
                        _zip_parametric_runs(captured, scenario_name_fn=scen_name, prefix="fric_sweep"),
                        "fric_sweep_runs.zip",
                        "application/zip",
                        use_container_width=True,
                    )

                    long_df = _stack_parametric_histories(
                        captured,
                        scenario_name_fn=scen_name,
                        columns=["Impact_Force_MN", "Acceleration_g", "Penetration_mm"],
                    )
                    if not long_df.empty:
                        d1, d2 = st.columns(2)
                        safe_download_button(
                            d1,
                            "CSV stacked histories",
                            long_df.to_csv(index=False).encode(),
                            "fric_sweep_histories_long.csv",
                            "text/csv",
                            use_container_width=True,
                        )
                        safe_download_button(
                            d2,
                            "XLSX stacked histories",
                            to_excel(long_df),
                            "fric_sweep_histories_long.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )

                    # One-click bundle: raw runs + tables + plots (HTML)
                    st.markdown("#### Export bundle (paper-friendly)")
                    st.caption(
                        "Creates a single ZIP containing: raw runs, summary tables, stacked histories, "
                        "and the key overlay plots. Filenames include the current speed for traceability."
                    )
                    try:
                        plots: dict[str, bytes] = {}

                        if any("Impact_Force_MN" in d.columns for _, d in captured):
                            figF2 = _plot_time_history_overlay(
                                captured,
                                "Impact_Force_MN",
                                int(max_runs),
                                lambda cfg, df, idx: scen_name(cfg, df, idx),
                                title="Impact force – time (overlay)",
                            )
                            plots["overlay__Impact_Force_MN"] = _fig_to_html_bytes(figF2)

                        if any("Acceleration_g" in d.columns for _, d in captured):
                            figA2 = _plot_time_history_overlay(
                                captured,
                                "Acceleration_g",
                                int(max_runs),
                                lambda cfg, df, idx: scen_name(cfg, df, idx),
                                title="Acceleration – time (overlay)",
                            )
                            plots["overlay__Acceleration_g"] = _fig_to_html_bytes(figA2)

                        if any("Penetration_mm" in d.columns for _, d in captured):
                            figP2 = _plot_time_history_overlay(
                                captured,
                                "Penetration_mm",
                                int(max_runs),
                                lambda cfg, df, idx: scen_name(cfg, df, idx),
                                title="Penetration – time (overlay)",
                            )
                            plots["overlay__Penetration_mm"] = _fig_to_html_bytes(figP2)

                        for i, (cfg_i, df_i) in enumerate(captured[: int(max_runs)]):
                            if not (
                                isinstance(df_i, pd.DataFrame)
                                and "Impact_Force_MN" in df_i.columns
                                and "Penetration_mm" in df_i.columns
                            ):
                                continue
                            name_i = sanitize_filename(scen_name(cfg_i, df_i, i))
                            plots[f"force_penetration__{name_i}"] = _fig_to_html_bytes(
                                _force_penetration_time_gradient(
                                    df_i, title=f"Force–penetration (time gradient) – {name_i}"
                                )
                            )

                        runs_map: dict[str, pd.DataFrame] = {}
                        for i, (cfg_i, df_i) in enumerate(captured):
                            name_i = sanitize_filename(scen_name(cfg_i, df_i, i))
                            sp = _cfg_speed_kmh(cfg_i)
                            if isinstance(sp, (int, float)):
                                name_i = f"v{float(sp):.0f}kmh__{name_i}"
                            runs_map[name_i] = df_i

                        meta = {
                            "study": "friction_restitution_sweep",
                            "sweep_kind": sweep_kind,
                            "base_speed_kmh": _cfg_speed_kmh(params),
                            "params": {k: v for k, v in params.items() if k not in ("train", "materials")},
                        }

                        bundle = make_bundle_zip(
                            study="friction_restitution_sweep",
                            metadata=meta,
                            dataframes={
                                "summary": summary_df,
                                "histories_long": long_df,
                            },
                            plots_html=plots,
                            runs=runs_map,
                            extra_files={
                                "replot_matplotlib.py": replot_matplotlib_script(),
                                "README.txt": (
                                    "Friction/restitution sweep export bundle\n\n"
                                    "- data/: summary + stacked histories\n"
                                    "- runs/: raw time histories (one CSV per sweep value)\n"
                                    "- plots/: overlay plots + force-penetration (time gradient)\n"
                                )
                            },
                        )
                        safe_download_button(
                            st,
                            "Export friction/restitution bundle (ZIP)",
                            bundle,
                            f"friction_sweep__{sanitize_filename(str(sweep_kind))}__v{_cfg_speed_kmh(params) or 0:.0f}kmh__{utc_timestamp()}.zip",
                            "application/zip",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.warning(f"Bundle export not available: {exc}")

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

                st.markdown("**Recommended Δt criteria:**")
                peak_tol = st.slider("Peak error (%)", 0.0, 5.0, 1.0, 0.1, key="sens_peak_tol")
                impulse_tol = st.slider("Impulse error (%)", 0.0, 5.0, 1.0, 0.1, key="sens_impulse_tol")
                pen_tol = st.slider("Penetration error (%)", 0.0, 10.0, 3.0, 0.5, key="sens_pen_tol")

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
                        "**Baseline:** Relative errors are computed with respect to the finest "
                        "resolution case (smallest Δt, most negative α, tightest tolerance). "
                        "This shows numerical error vs. the most accurate result."
                    )
                    st.dataframe(summary_df)

                    # Simple peak plot with convergence analysis
                    try:
                        # Use dt_requested if available (new), fallback to dt_s (old)
                        dt_col = "dt_requested" if "dt_requested" in summary_df.columns else "dt_s"

                        # Flag potentially invalid runs
                        # Note: Could add checks for non-convergence, large energy errors, etc.
                        # For now, plot all runs - user can inspect energy_balance_error_J_final column
                        invalid_mask = pd.Series(False, index=summary_df.index)

                        fig_peak = go.Figure()

                        # Plot sampled peaks
                        valid_df = summary_df[~invalid_mask]
                        if len(valid_df) > 0:
                            fig_peak.add_trace(
                                go.Scatter(
                                    x=valid_df[dt_col],
                                    y=valid_df["peak_force_MN"],
                                    mode="markers+lines",
                                    name="Peak force (sampled)",
                                    line=dict(dash="dot"),
                                    marker=dict(size=8),
                                )
                            )

                        # Add interpolated peak if available
                        if "peak_force_interp_MN" in summary_df.columns and len(valid_df) > 0:
                            fig_peak.add_trace(
                                go.Scatter(
                                    x=valid_df[dt_col],
                                    y=valid_df["peak_force_interp_MN"],
                                    mode="markers+lines",
                                    name="Peak force (interpolated)",
                                    marker=dict(size=8),
                                )
                            )

                        # Mark invalid runs if any
                        invalid_df = summary_df[invalid_mask]
                        if len(invalid_df) > 0:
                            fig_peak.add_trace(
                                go.Scatter(
                                    x=invalid_df[dt_col],
                                    y=invalid_df["peak_force_MN"],
                                    mode="markers",
                                    name="Invalid (energy error)",
                                    marker=dict(size=10, symbol="x", color="red"),
                                )
                            )

                        # Compute recommended Δt using user-specified thresholds
                        if "peak_force_interp_rel_to_baseline_pct" in summary_df.columns and len(valid_df) > 2:
                            # Check peak error
                            peak_err = valid_df["peak_force_interp_rel_to_baseline_pct"].abs()
                            meets_peak = peak_err <= peak_tol

                            # Check impulse error
                            meets_impulse = pd.Series(True, index=valid_df.index)
                            if "impulse_MN_s" in valid_df.columns:
                                baseline_impulse = valid_df["impulse_MN_s"].iloc[valid_df[dt_col].argmin()]
                                if baseline_impulse != 0:
                                    impulse_err = 100.0 * (valid_df["impulse_MN_s"] - baseline_impulse).abs() / baseline_impulse
                                    meets_impulse = impulse_err <= impulse_tol

                            # Check penetration error
                            meets_pen = pd.Series(True, index=valid_df.index)
                            if "max_penetration_mm" in valid_df.columns:
                                baseline_pen = valid_df["max_penetration_mm"].iloc[valid_df[dt_col].argmin()]
                                if baseline_pen != 0:
                                    pen_err = 100.0 * (valid_df["max_penetration_mm"] - baseline_pen).abs() / baseline_pen
                                    meets_pen = pen_err <= pen_tol

                            # All criteria must be met
                            converged = meets_peak & meets_impulse & meets_pen
                            if converged.any():
                                # Recommend largest dt that meets ALL criteria (most efficient)
                                recommended_dt = valid_df.loc[converged, dt_col].max()
                                fig_peak.add_vline(
                                    x=recommended_dt,
                                    line_dash="dash",
                                    line_color="green",
                                    annotation_text=f"Recommended Δt = {recommended_dt:.1e}",
                                    annotation_position="top",
                                )

                        fig_peak.update_layout(
                            title=f"Peak force vs Δt (criteria: peak≤{peak_tol}%, impulse≤{impulse_tol}%, pen≤{pen_tol}%)",
                            xaxis_title="Δt (s)",
                            yaxis_title="Peak force (MN)",
                            height=400,
                        )
                        # LOG SCALE for x-axis with clean decade formatting
                        fig_peak.update_xaxes(
                            type="log",
                            tickformat=".0e",
                            exponentformat="e",
                        )
                        safe_plotly_chart(st, fig_peak, width="stretch")
                        st.session_state["dif_fig_peak_html"] = _fig_to_html_bytes(fig_peak)
                        st.session_state["sens_fig_peak_html"] = _fig_to_html_bytes(fig_peak)
                    except Exception as e:
                        st.warning(f"Could not generate peak plot: {e}")

                    # Overlay time histories for the selected quantity
                    if captured:
                        st.markdown("#### Time histories (overlay)")

                        def label_fn(cfg, df, idx):
                            dt = float(cfg.get("h_init", np.nan))
                            a = float(cfg.get("alpha_hht", np.nan))
                            tol = float(cfg.get("newton_tol", np.nan))
                            return f"dt={dt:.1e}s, α={a:+.2f}, tol={tol:.1e}"

                        fig = _plot_time_history_overlay(captured, q_label, max_runs, label_fn)
                        safe_plotly_chart(st, fig, width="stretch")
                        st.session_state["dif_fig_overlay_html"] = _fig_to_html_bytes(fig)
                        st.session_state["sens_fig_overlay_html"] = _fig_to_html_bytes(fig)

                    # Export bundle
                    st.markdown("#### Export bundle")
                    try:
                        long_df = _stack_parametric_histories(
                            captured,
                            scenario_name_fn=lambda cfg, df, idx: label_fn(cfg, df, idx),
                            columns=["Impact_Force_MN", "Acceleration_g", "Penetration_mm"],
                        )

                        plots: dict[str, bytes] = {}
                        fig_peak_html = st.session_state.get("sens_fig_peak_html", None)
                        fig_overlay_html = st.session_state.get("sens_fig_overlay_html", None)
                        if isinstance(fig_peak_html, (bytes, bytearray)):
                            plots["peak_vs_dt"] = bytes(fig_peak_html)
                        if isinstance(fig_overlay_html, (bytes, bytearray)):
                            plots[f"overlay__{q_label}"] = bytes(fig_overlay_html)

                        def _run_name(cfg, df, idx):
                            dt = float(cfg.get("h_init", np.nan))
                            a = float(cfg.get("alpha_hht", np.nan))
                            tol = float(cfg.get("newton_tol", np.nan))
                            return sanitize_filename(f"dt{dt:.2e}__a{a:+.2f}__tol{tol:.1e}")

                        runs_map: dict[str, pd.DataFrame] = { _run_name(cfg, df, i): df for i, (cfg, df) in enumerate(captured) }

                        meta = {
                            "study": "numerics_sensitivity",
                            "base_speed_kmh": _cfg_speed_kmh(params),
                            "quantity": q_label,
                            "criteria": {
                                "peak_tol_pct": float(peak_tol),
                                "impulse_tol_pct": float(impulse_tol),
                                "penetration_tol_pct": float(pen_tol),
                            },
                            "params": {k: v for k, v in params.items() if k not in ("train", "materials")},
                        }

                        bundle = make_bundle_zip(
                            study="numerics_sensitivity",
                            metadata=meta,
                            dataframes={
                                "summary": summary_df,
                                "histories_long": long_df,
                            },
                            plots_html=plots,
                            runs=runs_map,
                            extra_files={
                                "replot_matplotlib.py": replot_matplotlib_script(),
                                "README.txt": (
                                    "Numerics sensitivity export bundle\n\n"
                                    "- data/: summary + stacked histories\n"
                                    "- runs/: raw time histories (one CSV per (dt, alpha, tol))\n"
                                    "- plots/: overlay and peak-vs-dt (HTML)\n"
                                )
                            },
                        )
                        safe_download_button(
                            st,
                            "Export numerics-sensitivity bundle (ZIP)",
                            bundle,
                            f"numerics_sensitivity__{sanitize_filename(str(q_label))}__v{_cfg_speed_kmh(params) or 0:.0f}kmh__{utc_timestamp()}.zip",
                            "application/zip",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.warning(f"Bundle export not available: {exc}")
                        st.session_state["sens_fig_overlay_html"] = _fig_to_html_bytes(fig)


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

                st.markdown("**Strain-rate computation:**")
                L_ref_m = st.slider(
                    "L_ref (m)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="Characteristic length for strain rate: ε̇ ≈ δ̇/L_ref. "
                         "Physical choices: wall thickness, crush zone, buffer stroke. "
                         "Default 1.0 m provides dimensionless rate proxy.",
                    key="dif_L_ref",
                )

                run_dif = st.button("Run DIF study", type="primary", key="run_dif")

            with right:
                if run_dif:
                    try:
                        dif_vals = parse_floats_csv(dif_str) if dif_str.strip() else [1.0]

                        # Add L_ref to params for strain-rate computation
                        params_with_L_ref = dict(params)
                        params_with_L_ref["L_ref_m"] = L_ref_m

                        captured: list[tuple[dict, pd.DataFrame]] = []

                        def _cap_sim(cfg: Dict[str, Any]) -> pd.DataFrame:
                            df = run_simulation(cfg)
                            captured.append((cfg, df))
                            return df

                        summary_df = run_fixed_dif_sensitivity(
                            params_with_L_ref,
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
                        safe_plotly_chart(st, fig_peak, width="stretch")
                        st.session_state["dif_fig_peak_html"] = _fig_to_html_bytes(fig_peak)
                    except (KeyError, ValueError, IndexError) as e:
                        st.info(f"Peak vs DIF plot not available: {e}")

                    # Overlay time histories
                    if captured:
                        st.markdown("#### Time histories (overlay)")

                        def label_fn(cfg, df, idx):
                            # DIF value is in summary_df, not in cfg
                            dif_val = float(summary_df["dif"].iloc[idx]) if (idx < len(summary_df)) else float("nan")
                            return f"DIF={dif_val:.3f}"

                        fig = _plot_time_history_overlay(captured, q_label, max_runs, label_fn)
                        safe_plotly_chart(st, fig, width="stretch")
                        st.session_state["dif_fig_overlay_html"] = _fig_to_html_bytes(fig)

  

                    # Export bundle
                    st.markdown("#### Export bundle")
                    if captured:
                        try:
                            long_df = _stack_parametric_histories(
                                captured,
                                scenario_name_fn=lambda cfg, df, idx: label_fn(cfg, df, idx),
                                columns=["Impact_Force_MN", "Acceleration_g", "Penetration_mm"],
                            )

                            plots: dict[str, bytes] = {}
                            fig_peak_html = st.session_state.get("dif_fig_peak_html", None)
                            fig_overlay_html = st.session_state.get("dif_fig_overlay_html", None)
                            if isinstance(fig_peak_html, (bytes, bytearray)):
                                plots["peak_vs_dif"] = bytes(fig_peak_html)
                            if isinstance(fig_overlay_html, (bytes, bytearray)):
                                plots[f"overlay__{q_label}"] = bytes(fig_overlay_html)

                            runs_map: dict[str, pd.DataFrame] = {}
                            for i, (cfg_i, df_i) in enumerate(captured):
                                dif_val = float(summary_df["dif"].iloc[i]) if ("dif" in summary_df.columns and i < len(summary_df)) else float("nan")
                                name_i = sanitize_filename(f"dif{dif_val:.3f}")
                                sp = _cfg_speed_kmh(cfg_i)
                                if isinstance(sp, (int, float)):
                                    name_i = f"v{float(sp):.0f}kmh__{name_i}"
                                runs_map[name_i] = df_i

                            meta = {
                                "study": "strain_rate_fixed_dif",
                                "base_speed_kmh": _cfg_speed_kmh(params),
                                "quantity": q_label,
                                "difs": [float(x) for x in summary_df["dif"].tolist()] if "dif" in summary_df.columns else [],
                                "k_path": k_path.strip(),
                                "L_ref_m": float(L_ref_m),
                                "params": {k: v for k, v in params.items() if k not in ("train", "materials")},
                            }

                            bundle = make_bundle_zip(
                                study="strain_rate_fixed_dif",
                                metadata=meta,
                                dataframes={
                                    "summary": summary_df,
                                    "histories_long": long_df,
                                },
                                plots_html=plots,
                                runs=runs_map,
                                extra_files={
                                    "replot_matplotlib.py": replot_matplotlib_script(),
                                    "README.txt": (
                                        "Fixed-DIF strain-rate proxy export bundle\n\n"
                                        "- data/: summary + stacked histories\n"
                                        "- runs/: raw time histories (one CSV per DIF)\n"
                                        "- plots/: peak-vs-dif and overlay plot (HTML)\n"
                                        "- extra/replot_matplotlib.py: optional Matplotlib replot helper (PNG/SVG)\n"
                                    )
                                },
                            )
                            safe_download_button(
                                st,
                                "Export DIF bundle (ZIP)",
                                bundle,
                                f"dif_study__{sanitize_filename(k_path.strip())}__v{_cfg_speed_kmh(params) or 0:.0f}kmh__{utc_timestamp()}.zip",
                                "application/zip",
                                use_container_width=True,
                            )
                        except Exception as exc:
                            st.warning(f"Bundle export not available: {exc}")


        # --------------------------
        # Solver comparison (Newton vs Picard)
        # --------------------------
        with sub_solver:
            st.markdown("### Newton–Raphson vs Picard (nonlinear solver comparison)")
            st.caption(
                "Runs identical cases with both solvers and compares outputs and performance. "
                "Useful to validate that Picard and Newton give the same physics (within tolerance)."
            )

            default_speed = float(abs(float(params.get("v0_init", -56.0 / 3.6))) * 3.6)
            speeds_str = st.text_input(
                "Speeds to compare (km/h)",
                value=f"{default_speed:.0f}",
                key="solvercmp_speeds",
                help="Comma-separated list, e.g. `40,56,80`.",
            )

            q_label = st.selectbox(
                "Quantity to compare (time history + peak/impulse)",
                ["Impact_Force_MN", "Acceleration_g", "Penetration_mm"],
                index=0,
                key="solvercmp_quantity",
            )

            max_speeds = st.number_input(
                "Max speeds to run (safety limit)",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="solvercmp_max_speeds",
            )

            run_cmp = st.button("Run solver comparison", type="primary", key="run_solver_cmp")

            if run_cmp:
                try:
                    speeds = parse_floats_csv(speeds_str) if speeds_str.strip() else []
                    speeds = [float(s) for s in speeds][: int(max_speeds)]

                    if not speeds:
                        st.warning("Please provide at least one speed.")
                    else:
                        base = dict(params)
                        solvers = ["newton", "picard"]

                        rows: list[dict[str, float | str]] = []
                        series: dict[tuple[str, float], pd.DataFrame] = {}

                        prog = st.progress(0.0)
                        total = max(1, len(solvers) * len(speeds))
                        done = 0

                        for solver in solvers:
                            for sp in speeds:
                                cfg = dict(base)
                                cfg["solver"] = solver
                                cfg["v0_init"] = -abs(float(sp)) / 3.6

                                t0 = time.perf_counter()
                                df = run_simulation(cfg)
                                runtime_s = float(time.perf_counter() - t0)

                                series[(solver, float(sp))] = df

                                # Time axis for impulse (seconds)
                                t_ms, _ = _get_time_axis(df)
                                t = t_ms / 1000.0

                                y = df[q_label].to_numpy() if q_label in df.columns else np.array([])

                                peak = float(np.nanmax(y)) if y.size else float("nan")
                                # Use trapezoid (NumPy 2.0+) or trapz (NumPy 1.x)
                                trapz_func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
                                impulse = float(trapz_func(y, t)) if y.size else float("nan")

                                rows.append(
                                    {
                                        "speed_kmh": float(sp),
                                        "solver": solver,
                                        "peak": peak,
                                        "impulse": impulse,
                                        "runtime_s": runtime_s,
                                    }
                                )

                                done += 1
                                prog.progress(done / total)

                        summary = pd.DataFrame(rows)
                        st.session_state["solvercmp_summary"] = summary
                        st.session_state["solvercmp_series"] = series
                except Exception as exc:
                    st.error(f"Solver comparison failed: {exc}")

            summary = st.session_state.get("solvercmp_summary", None)
            series = st.session_state.get("solvercmp_series", None)

            if summary is None:
                st.info("Run the comparison to see results.")
            else:
                st.markdown("#### Summary")
                st.dataframe(summary)

                # Pivot for quick peak difference
                try:
                    pivot = summary.pivot_table(
                        index="speed_kmh",
                        columns="solver",
                        values="peak",
                        aggfunc="first",
                    )
                    if "newton" in pivot.columns and "picard" in pivot.columns:
                        pivot = pivot.reset_index()
                        pivot["peak_diff_pct_picard_vs_newton"] = 100.0 * (
                            (pivot["picard"] - pivot["newton"]) / pivot["newton"]
                        )
                        st.markdown("#### Peak difference")
                        st.dataframe(pivot)
                except (KeyError, ValueError) as e:
                    st.debug(f"Peak difference table not available: {e}")

                # Overlay time history for a selected speed
                if isinstance(series, dict) and len(series) > 0:
                    speeds_avail = sorted({float(v) for v in summary["speed_kmh"].unique()})
                    if speeds_avail:
                        sel_speed = st.selectbox(
                            "Overlay time history (select speed)",
                            options=speeds_avail,
                            index=0,
                            key="solvercmp_overlay_speed",
                        )

                        fig = go.Figure()
                        for solver in ["newton", "picard"]:
                            df = series.get((solver, float(sel_speed)))
                            if df is None or q_label not in df.columns:
                                continue

                            t_ms, _ = _get_time_axis(df)

                            fig.add_trace(
                                go.Scatter(
                                    x=t_ms,
                                    y=df[q_label],
                                    mode="lines",
                                    name=f"{solver}",
                                )
                            )

                        fig.update_layout(
                            title=f"{q_label} – Newton vs Picard @ {sel_speed:.0f} km/h",
                            xaxis_title="Time (ms)",
                            yaxis_title=q_label,
                            height=450,
                        )
                        safe_plotly_chart(st, fig, width="stretch")



                # Export bundle
                st.markdown("#### Export bundle")
                if isinstance(series, dict) and len(series) > 0:
                    try:
                        runs_map: dict[str, pd.DataFrame] = {}
                        long_rows: list[pd.DataFrame] = []

                        for (solver, sp), df in series.items():
                            name = sanitize_filename(f"v{float(sp):.0f}kmh__{solver}")
                            runs_map[name] = df

                            keep_cols = [c for c in ("Time_s", "Time_ms") if c in df.columns]
                            keep_cols += [c for c in ("Impact_Force_MN", "Acceleration_g", "Penetration_mm") if c in df.columns]
                            if keep_cols:
                                tmp = df[keep_cols].copy()
                                tmp.insert(0, "solver", str(solver))
                                tmp.insert(0, "speed_kmh", float(sp))
                                tmp.insert(0, "scenario", f"{solver}@{float(sp):.0f}kmh")
                                long_rows.append(tmp)

                        long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

                        plots: dict[str, bytes] = {}

                        # Peak vs speed
                        try:
                            fig_peak = go.Figure()
                            for solver in ["newton", "picard"]:
                                sub = summary[summary["solver"] == solver].sort_values("speed_kmh")
                                if len(sub):
                                    fig_peak.add_trace(go.Scatter(x=sub["speed_kmh"], y=sub["peak"], mode="lines+markers", name=f"peak_{solver}"))
                            fig_peak.update_layout(title=f"Peak vs speed – {q_label}", xaxis_title="Speed (km/h)", yaxis_title="Peak", height=350)
                            plots["peak_vs_speed"] = _fig_to_html_bytes(fig_peak)
                        except Exception:
                            pass

                        # Runtime vs speed
                        try:
                            fig_rt = go.Figure()
                            for solver in ["newton", "picard"]:
                                sub = summary[summary["solver"] == solver].sort_values("speed_kmh")
                                if len(sub):
                                    fig_rt.add_trace(go.Scatter(x=sub["speed_kmh"], y=sub["runtime_s"], mode="lines+markers", name=f"runtime_{solver}"))
                            fig_rt.update_layout(title="Runtime vs speed", xaxis_title="Speed (km/h)", yaxis_title="Runtime (s)", height=350)
                            plots["runtime_vs_speed"] = _fig_to_html_bytes(fig_rt)
                        except Exception:
                            pass

                        # Per-speed overlays
                        try:
                            for sp in sorted({float(v) for v in summary["speed_kmh"].unique()}):
                                fig_ov = go.Figure()
                                for solver in ["newton", "picard"]:
                                    df = series.get((solver, float(sp)))
                                    if df is None or q_label not in df.columns:
                                        continue
                                    t_ms, _ = _get_time_axis(df)
                                    fig_ov.add_trace(go.Scatter(x=t_ms, y=df[q_label], mode="lines", name=f"{solver}"))
                                fig_ov.update_layout(
                                    title=f"{q_label} – Newton vs Picard @ {sp:.0f} km/h",
                                    xaxis_title="Time (ms)",
                                    yaxis_title=q_label,
                                    height=450,
                                )
                                plots[f"overlay__{sanitize_filename(str(q_label))}__v{sp:.0f}kmh"] = _fig_to_html_bytes(fig_ov)
                        except Exception:
                            pass

                        meta = {
                            "study": "solver_comparison",
                            "quantity": q_label,
                            "speeds_kmh": [float(v) for v in sorted({float(v) for v in summary["speed_kmh"].unique()})],
                            "base_speed_kmh": _cfg_speed_kmh(params),
                            "params": {k: v for k, v in params.items() if k not in ("train", "materials")},
                        }

                        bundle = make_bundle_zip(
                            study="solver_comparison",
                            metadata=meta,
                            dataframes={
                                "summary": summary,
                                "histories_long": long_df,
                            },
                            plots_html=plots,
                            runs=runs_map,
                            extra_files={
                                "replot_matplotlib.py": replot_matplotlib_script(),
                                "README.txt": (
                                    "Solver comparison export bundle (Newton vs Picard)\n\n"
                                    "- data/: summary + stacked histories\n"
                                    "- runs/: raw time histories (one CSV per (speed, solver))\n"
                                    "- plots/: peak/runtime vs speed + per-speed overlays (HTML)\n"
                                    "- extra/replot_matplotlib.py: optional Matplotlib replot helper (PNG/SVG)\n"
                                )
                            },
                        )

                        speeds = meta.get("speeds_kmh", [])
                        vmin = min(speeds) if speeds else 0.0
                        vmax = max(speeds) if speeds else 0.0

                        safe_download_button(
                            st,
                            "Export solver-comparison bundle (ZIP)",
                            bundle,
                            f"solver_comparison__{sanitize_filename(str(q_label))}__{vmin:.0f}-{vmax:.0f}kmh__{utc_timestamp()}.zip",
                            "application/zip",
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.warning(f"Bundle export not available: {exc}")


# ABOUT / DOCUMENTATION TAB
    # --------------------------------------------------------------
    with tab_about:
        display_header()
        display_citation()


if __name__ == "__main__":
    main()
