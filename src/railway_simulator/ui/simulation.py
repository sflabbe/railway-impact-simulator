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

from railway_simulator.ui.st_compat import safe_plotly_chart, safe_download_button
from railway_simulator.core.engine import run_simulation, GRAVITY
from railway_simulator.config.presets import (
    load_en15227_presets,
    resolve_scenario_preset,
    resolve_partner_preset,
    resolve_interface_preset,
)
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
from .plotting import (
    create_mass_kinematics_plots,
    create_mass_force_displacement_plots,
    create_results_plots,
    create_spring_plots,
    create_nodal_field_surface,
    estimate_nodal_field_color_range,
    export_nodal_field_heatmap_bytes,
)
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
    Ejecuta la simulación y gestiona el caché tanto de la física (core)
    como del post-proceso (edificio/SDOF) para evitar recargas innecesarias.

    - Core physics (impacto tren-muro) se ejecuta solo cuando:
        * run_new=True (botón), o
        * no hay resultados cacheados.
    - Building SDOF se cachea por parámetros para que cambiar widgets
      (p.ej. selectbox de masa / resorte) no dispare CPU innecesaria.
    """
    # ------------------------------------------------------
    # 1) Simulación base de impacto (Core Physics)
    # ------------------------------------------------------
    df_core = st.session_state.get("sim_results", None)

    # Si se pide run_new, forzamos la ejecución.
    # Si df_core no existe, forzamos la ejecución.
    if run_new or df_core is None:
        with st.spinner("Running HHT-α simulation..."):
            try:
                df_core = run_simulation(params)
            except Exception as e:
                st.error(f"Simulation error: {e}")
                return

        # Guardamos resultados y parámetros en Session State
        st.session_state["sim_results"] = df_core
        st.session_state["sim_params_core"] = params

        # Reset de notificación (para no spamear en reruns)
        st.session_state["sim_results_notified"] = False

        # IMPORTANTE: invalidamos el caché del edificio porque la física cambió
        for k in ("sim_building_results", "last_building_params"):
            if k in st.session_state:
                del st.session_state[k]

        st.success("✅ Complete!")
    else:
        # Aquí entra cuando cambias selectboxes/sliders de UI.
        if not st.session_state.get("sim_results_notified", False):
            st.info("Using cached impact history.")
            st.session_state["sim_results_notified"] = True

    if df_core is None:
        st.error("No simulation results available.")
        return

    # ------------------------------------------------------
    # 2) SDOF del edificio (Optimizado con Caché)
    # ------------------------------------------------------
    building_enabled = (
        params.get("building_enable", False)
        and params.get("k_wall", 0.0) > 0.0
        and params.get("building_mass", 0.0) > 0.0
    )

    building_df = None

    if building_enabled:
        # Clave única para el estado actual de los parámetros del edificio
        current_build_params = {
            "k_wall": float(params.get("k_wall", 0.0)),
            "m_build": float(params.get("building_mass", 0.0)),
            "zeta": float(params.get("building_zeta", 0.0)),
            "model": params.get("building_model", "Linear elastic SDOF"),
            "uy_mm": float(params.get("building_uy_mm", 10.0)),
            "alpha": float(params.get("building_alpha", 0.05)),
            "gamma": float(params.get("building_gamma", 0.4)),
            # Incluimos la longitud para asegurar que es la misma simulación
            "core_len": int(len(df_core)),
        }

        cached_build_df = st.session_state.get("sim_building_results", None)
        cached_build_params = st.session_state.get("last_building_params", {})

        # Comparamos: ¿Tenemos datos cacheados Y son los mismos parámetros?
        if (cached_build_df is not None) and (cached_build_params == current_build_params):
            building_df = cached_build_df
        else:
            # Si no coincide, recalculamos (solo aquí se gasta CPU)
            try:
                building_df = compute_building_sdof_response(
                    df_core,
                    k_wall=float(params.get("k_wall", 0.0)),
                    m_build=float(params.get("building_mass", 0.0)),
                    zeta=float(params.get("building_zeta", 0.0)),
                    model=params.get("building_model", "Linear elastic SDOF"),
                    uy_mm=float(params.get("building_uy_mm", 10.0)),
                    alpha=float(params.get("building_alpha", 0.05)),
                    gamma=float(params.get("building_gamma", 0.4)),
                )
                st.session_state["sim_building_results"] = building_df
                st.session_state["last_building_params"] = current_build_params
            except Exception as e:
                st.warning(f"Building SDOF response could not be computed: {e}")
                building_df = None

    # Unimos los dataframes para visualización
    df = df_core.copy()
    if building_df is not None and not building_df.empty:
        df = pd.concat([df_core, building_df], axis=1)

    # ------------------------------------------------------
    # 3) Tabs de resultados
    # ------------------------------------------------------
    tab_global, tab_batch, tab_building, tab_train, tab_springs, tab_nodal = st.tabs(
        [
            "📈 Global Results",
            "🧪 Batch Compare",
            "🏢 Building Response (SDOF)",
            "🚃 Train Configuration",
            "🧷 Springs & Masses",
            "🧊 Nodal Fields",
        ]
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
        safe_plotly_chart(st, fig, width="stretch")

        # Optional deeper energy / dissipation breakdown
        with st.expander("🔬 Advanced energy breakdown", expanded=False):
            cols = [
                "E_kin_J",
                "E_pot_contact_J",
                "E_pot_spring_J",
                "E_diss_rayleigh_J",
                "E_diss_bw_J",
                "E_diss_softening_J",
                "E_diss_contact_damp_J",
                "E_diss_friction_J",
                "E_diss_mass_contact_J",
                "E_diss_total_J",
                "E_num_J",
            ]
            present = [c for c in cols if c in df.columns]
            if present:
                import plotly.graph_objects as go

                fig_e = go.Figure()
                for c in present:
                    fig_e.add_trace(go.Scatter(x=df["Time_ms"], y=df[c] / 1e6, mode="lines", name=c.replace("_J", "")))
                fig_e.update_layout(
                    height=450,
                    margin=dict(t=30, b=30, l=60, r=30),
                    xaxis_title="Time (ms)",
                    yaxis_title="Energy (MJ)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                safe_plotly_chart(st, fig_e, width="stretch")
            else:
                st.info("Energy bookkeeping columns are not available in these results.")

    # ----- Batch compare (EN15227 / multi-YAML) -----
    with tab_batch:
        from pathlib import Path
        import yaml
        import zipfile
        import io
        import plotly.graph_objects as go

        st.markdown(
            "Run and compare multiple YAML configs (especially **configs/en15227/**) and overlay key responses."
        )

        # Locate configs directory
        cfg_dir = Path.cwd() / "configs"
        if not cfg_dir.is_dir():
            try:
                cfg_dir = Path(__file__).resolve().parents[3] / "configs"
            except Exception:
                cfg_dir = Path.cwd() / "configs"

        all_yaml = sorted(list(cfg_dir.rglob("*.yml")) + list(cfg_dir.rglob("*.yaml")), key=lambda p: str(p).lower())
        if not all_yaml:
            st.info("No YAML configs found under ./configs.")
        else:
            # Default: EN15227 variants
            en_files = [p for p in all_yaml if "en15227" in str(p).lower() or "__en15227_" in p.name.lower()]
            default_files = en_files[: min(12, len(en_files))] if en_files else all_yaml[: min(8, len(all_yaml))]

            # Filters
            case_filter = st.selectbox(
                "Filter by EN15227 case",
                options=["All", "C1", "C2", "C3"],
                index=0,
                help="Filters filenames containing __EN15227_C1/C2/C3.",
            )

            def _match_case(p: Path) -> bool:
                if case_filter == "All":
                    return True
                return f"__en15227_{case_filter.lower()}" in p.name.lower()

            filtered = [p for p in (en_files if en_files else all_yaml) if _match_case(p)]

            # Build labels with relative path
            def _label(p: Path) -> str:
                try:
                    rel = p.relative_to(cfg_dir)
                    return str(rel)
                except Exception:
                    return str(p)

            selected = st.multiselect(
                "Select YAML configs",
                options=filtered,
                default=default_files if case_filter == "All" else [p for p in default_files if _match_case(p)],
                format_func=_label,
            )

            col_run, col_clear, col_dl = st.columns([1, 1, 1])
            run_batch = col_run.button("▶️ Run batch", use_container_width=True)
            if col_clear.button("🧹 Clear batch cache", use_container_width=True):
                st.session_state.pop("batch_results", None)
                st.session_state.pop("batch_meta", None)
                st.session_state.pop("batch_fp", None)

            # Fingerprint to reuse cached runs
            fp_items = []
            for p in selected:
                try:
                    fp_items.append(f"{p}|{p.stat().st_mtime_ns}")
                except Exception:
                    fp_items.append(str(p))
            fp = "|".join(fp_items)

            if run_batch and selected:
                presets = load_en15227_presets(Path.cwd())
                prog = st.progress(0)
                results: dict[str, pd.DataFrame] = {}
                meta_rows: list[dict[str, Any]] = []

                for i, p in enumerate(selected):
                    try:
                        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                        if not isinstance(cfg, dict):
                            cfg = {}
                    except Exception as e:
                        st.warning(f"Failed to load {p}: {e}")
                        continue

                    label = cfg.get("case_name") or p.stem

                    # Resolve EN15227 info if present
                    scenario_id = None
                    try:
                        col = cfg.get("collision") or {}
                        if isinstance(col, dict):
                            scenario_id = col.get("scenario")
                    except Exception:
                        scenario_id = None

                    scen = resolve_scenario_preset(presets, scenario_id) if scenario_id else None
                    partner_id = None
                    interface_id = None
                    speed_kmh = None
                    if scen:
                        speed_kmh = scen.get("speed_kmh")
                        try:
                            partner_id = (scen.get("partner") or {}).get("preset_id")
                            interface_id = (scen.get("interface") or {}).get("preset_id")
                        except Exception:
                            pass

                    partner = resolve_partner_preset(presets, partner_id) if partner_id else None
                    interface = resolve_interface_preset(presets, interface_id) if interface_id else None

                    M_train = None
                    try:
                        masses = cfg.get("masses")
                        if masses is not None:
                            M_train = float(sum(float(x) for x in masses))
                    except Exception:
                        M_train = None

                    M_ref = None
                    if partner and isinstance(partner.get("mass_kg"), (int, float)):
                        M_ref = float(partner["mass_kg"])

                    # Run
                    with st.spinner(f"Running: {label}"):
                        try:
                            df_i = run_simulation(cfg)
                        except Exception as e:
                            st.error(f"Run failed for {label}: {e}")
                            continue
                    results[label] = df_i

                    # Summary
                    row = {
                        "case": label,
                        "yaml": _label(p),
                        "scenario": scenario_id or "",
                        "speed_kmh": speed_kmh,
                        "train_mass_t": (M_train / 1000.0) if M_train else None,
                        "ref_mass_t": (M_ref / 1000.0) if M_ref else None,
                        "max_force_MN": float(df_i["Impact_Force_MN"].max()) if "Impact_Force_MN" in df_i else None,
                        "max_pen_mm": float(df_i["Penetration_mm"].max()) if "Penetration_mm" in df_i else None,
                        "max_acc_g": float(df_i["Acceleration_g"].max()) if "Acceleration_g" in df_i else None,
                        "peak_E_num_%": (100.0 * float(df_i["E_num_ratio"].max())) if "E_num_ratio" in df_i else None,
                    }
                    if interface and isinstance(interface.get("metadata"), dict):
                        row["expected_energy_kJ"] = float(interface["metadata"].get("expected_energy_J", 0.0)) / 1e3
                    meta_rows.append(row)

                    prog.progress((i + 1) / max(1, len(selected)))

                st.session_state["batch_results"] = results
                st.session_state["batch_meta"] = meta_rows
                st.session_state["batch_fp"] = fp
                st.success(f"Batch complete: {len(results)} run(s)")

            # Show cached
            batch_results = st.session_state.get("batch_results")
            batch_meta = st.session_state.get("batch_meta")
            batch_fp = st.session_state.get("batch_fp")

            if batch_results and batch_fp == fp:
                st.markdown("### Summary")
                meta_df = pd.DataFrame(batch_meta)
                if not meta_df.empty:
                    st.dataframe(meta_df.sort_values(by=["max_force_MN"], ascending=False), use_container_width=True)

                st.markdown("### Overlay plots")
                y_choice = st.selectbox(
                    "Signal",
                    options=[
                        "Impact_Force_MN vs Time_ms",
                        "Penetration_mm vs Time_ms",
                        "Acceleration_g vs Time_ms",
                        "Hysteresis (Force vs Penetration)",
                        "Energy Balance Quality (E_num_ratio %)",
                    ],
                    index=0,
                )

                fig_o = go.Figure()
                for label, dfi in batch_results.items():
                    try:
                        if y_choice == "Impact_Force_MN vs Time_ms":
                            fig_o.add_trace(go.Scatter(x=dfi["Time_ms"], y=dfi["Impact_Force_MN"], mode="lines", name=label))
                            fig_o.update_xaxes(title_text="Time (ms)")
                            fig_o.update_yaxes(title_text="Impact force (MN)")
                        elif y_choice == "Penetration_mm vs Time_ms":
                            fig_o.add_trace(go.Scatter(x=dfi["Time_ms"], y=dfi["Penetration_mm"], mode="lines", name=label))
                            fig_o.update_xaxes(title_text="Time (ms)")
                            fig_o.update_yaxes(title_text="Penetration (mm)")
                        elif y_choice == "Acceleration_g vs Time_ms":
                            fig_o.add_trace(go.Scatter(x=dfi["Time_ms"], y=dfi["Acceleration_g"], mode="lines", name=label))
                            fig_o.update_xaxes(title_text="Time (ms)")
                            fig_o.update_yaxes(title_text="Acceleration (g)")
                        elif y_choice == "Hysteresis (Force vs Penetration)":
                            fig_o.add_trace(go.Scatter(x=dfi["Penetration_mm"], y=dfi["Impact_Force_MN"], mode="lines", name=label))
                            fig_o.update_xaxes(title_text="Penetration (mm)")
                            fig_o.update_yaxes(title_text="Impact force (MN)")
                        else:
                            fig_o.add_trace(go.Scatter(x=dfi["Time_ms"], y=100.0 * dfi["E_num_ratio"], mode="lines", name=label))
                            fig_o.update_xaxes(title_text="Time (ms)")
                            fig_o.update_yaxes(title_text="|E_num| / E0 (%)")
                    except Exception:
                        continue

                fig_o.update_layout(height=520, margin=dict(t=30, b=30, l=60, r=30))
                safe_plotly_chart(st, fig_o, width="stretch")

                # Zip download (CSV per run)
                if col_dl.button("📦 Download batch CSV zip", use_container_width=True):
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                        for label, dfi in batch_results.items():
                            z.writestr(f"{label}.csv", dfi.to_csv(index=False))
                        if batch_meta:
                            z.writestr("summary.csv", pd.DataFrame(batch_meta).to_csv(index=False))
                    buf.seek(0)
                    safe_download_button(st, "Download", buf.read(), "batch_results.zip", "application/zip", width="stretch")
            else:
                st.info("Select some YAML files and click **Run batch** to generate overlay plots.")

        st.subheader("📥 Export")
        e1, e2, e3 = st.columns(3)

        safe_download_button(e1, 
            "📄 CSV",
            df.to_csv(index=False).encode(),
            "results.csv",
            "text/csv",
            width="stretch",
        )

        safe_download_button(e2, 
            "📝 TXT",
            df.to_string(index=False).encode(),
            "results.txt",
            "text/plain",
            width="stretch",
        )

        safe_download_button(e3, 
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
            safe_plotly_chart(st, fig_b, width="stretch")

            st.markdown("#### Building hysteresis (restoring force vs displacement)")
            try:
                fig_h = create_building_hysteresis_plot(df)
                safe_plotly_chart(st, fig_h, width="stretch")
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
                        safe_plotly_chart(st, anim_fig, width="stretch")
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
                    safe_plotly_chart(st, fig_spec, width="stretch")

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
                        safe_plotly_chart(st, fig_multi, width="stretch")

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

        # EN15227 helper (only if current run came from YAML with collision info)
        try:
            yaml_cfg = st.session_state.get("yaml_example_cfg")
        except Exception:
            yaml_cfg = None

        with st.expander("🇪🇺 EN 15227 helper", expanded=False):
            presets = load_en15227_presets(Path.cwd())
            scenario_id = None
            if isinstance(yaml_cfg, dict):
                col = yaml_cfg.get("collision")
                if isinstance(col, dict):
                    scenario_id = col.get("scenario")

            if not scenario_id:
                st.caption("No EN15227 scenario found in the selected YAML.")
            else:
                scen = resolve_scenario_preset(presets, scenario_id)
                if not scen:
                    st.warning(f"Unknown scenario: {scenario_id}")
                else:
                    partner_id = (scen.get("partner") or {}).get("preset_id")
                    interface_id = (scen.get("interface") or {}).get("preset_id")
                    speed_kmh = scen.get("speed_kmh")
                    partner = resolve_partner_preset(presets, partner_id) if partner_id else None
                    interface = resolve_interface_preset(presets, interface_id) if interface_id else None

                    cA, cB, cC = st.columns(3)
                    cA.metric("Scenario", scenario_id)
                    if speed_kmh is not None:
                        cB.metric("Reference speed", f"{float(speed_kmh):.0f} km/h")
                    if partner and partner.get("mass_kg") is not None:
                        cC.metric("Reference mass", f"{float(partner['mass_kg'])/1000.0:.1f} t")

                    if interface and isinstance(interface.get("points"), list):
                        import plotly.graph_objects as go
                        pts = interface["points"]
                        x_m = [float(p[0]) for p in pts]
                        f_MN = [float(p[1]) / 1e6 for p in pts]
                        fig_if = go.Figure()
                        fig_if.add_trace(go.Scatter(x=x_m, y=f_MN, mode="lines+markers", name=interface_id))
                        fig_if.update_layout(height=320, margin=dict(t=10, b=10, l=60, r=20))
                        fig_if.update_xaxes(title_text="Displacement (m)")
                        fig_if.update_yaxes(title_text="Force (MN)")
                        safe_plotly_chart(st, fig_if, width="stretch")

                        # Energy (area under curve)
                        try:
                            import numpy as _np
                            E = 0.0
                            for (x0, f0), (x1, f1) in zip(pts[:-1], pts[1:]):
                                E += 0.5 * (float(f0) + float(f1)) * (float(x1) - float(x0))
                            st.caption(f"Curve energy (area) ≈ {E/1e3:.1f} kJ")
                            md = interface.get("metadata") or {}
                            if isinstance(md, dict) and md.get("expected_energy_J") is not None:
                                st.caption(f"Expected energy (metadata) ≈ {float(md['expected_energy_J'])/1e3:.1f} kJ")
                        except Exception:
                            pass

                    # Mass consistency quick check
                    try:
                        masses = np.asarray(params.get("masses", []), dtype=float)
                        M = float(masses.sum())
                        if partner and partner.get("mass_kg") is not None:
                            M_ref = float(partner["mass_kg"])
                            st.caption(f"Current train mass: {M/1000.0:.1f} t · target: {M_ref/1000.0:.1f} t · ratio: {M/M_ref:.3f}")
                    except Exception:
                        pass
        fig_train = create_train_geometry_plot(params)
        safe_plotly_chart(st, fig_train, width="stretch")

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

    
    # ----- Springs and masses results -----
    with tab_springs:
        st.markdown("### Spring and mass response histories")

        # Robust n_masses detection:
        # 1) Prefer DataFrame columns (reliable even if df.attrs are lost in caching/pickling)
        # 2) Fallback to df.attrs / params
        import re

        mass_ids = []
        for c in df.columns:
            mm = re.match(r"Mass(\d+)_Position_x_m$", str(c))
            if mm:
                mass_ids.append(int(mm.group(1)))

        n_masses_cols = max(mass_ids) if mass_ids else 0
        n_masses = int(
            n_masses_cols
            or df.attrs.get("n_masses", 0)
            or params.get("n_masses", len(params.get("masses", [])))
            or 0
        )

        if n_masses <= 0:
            st.info("No mass results available for the current simulation.")
        else:
            mass_index = st.selectbox(
                "Mass index",
                options=list(range(n_masses)),
                format_func=lambda i: f"Mass {i} (node {i + 1})",
                key="ui_mass_index",
            )

            mass_fig = create_mass_kinematics_plots(df, mass_index)
            

            if mass_fig is None:
                st.info("Mass kinematics columns are not available in these results.")
            else:
                safe_plotly_chart(st, mass_fig, width="stretch")

            st.markdown("#### Mass force–displacement")

            mode_label = st.selectbox(
                "Force used for mass loop",
                [
                    "Auto (prefer core total)",
                    "Core total (x)",
                    "Core internal (x)",
                    "Net (left - right)",
                    "Left spring",
                    "Right spring",
                ],
                key="ui_mass_force_mode",
                help=(
                    "Forces for the force–displacement loop. 'Core total' uses the nodal force exported by the solver "
                    "(includes wall contact, friction and mass-to-mass contact). 'Core internal' is springs-only. "
                    "If core force columns are not available, the plot falls back to spring reconstruction."
                ),
            )
            mode = {
                "Auto (prefer core total)": "auto",
                "Core total (x)": "total",
                "Core internal (x)": "internal",
                "Net (left - right)": "net",
                "Left spring": "left",
                "Right spring": "right",
            }[mode_label]

            mfd_fig = create_mass_force_displacement_plots(df, mass_index, mode=mode)
            if mfd_fig is None:
                st.info(
                    "Mass force–displacement could not be computed (required force columns not available for this mass)."
                )
            else:
                safe_plotly_chart(st, mfd_fig, width="stretch")

            if n_masses < 2:
                st.info("Spring results are not available for a single-mass model.")
            else:
                st.markdown("#### Spring response")
                spring_index = st.selectbox(
                    "Spring index",
                    options=list(range(n_masses - 1)),
                    format_func=lambda i: f"Spring {i} (between masses {i + 1} and {i + 2})",
                    key="ui_spring_index",
                )

                spring_fig = create_spring_plots(df, spring_index)
                if spring_fig is None:
                    st.info("Spring response columns are not available in these results.")
                else:
                    safe_plotly_chart(st, spring_fig, width="stretch")

                show_neighbors = st.checkbox(
                    "Show neighbor springs for selected mass",
                    value=False,
                    help="Displays springs connected to the selected mass (i-1 and i).",
                )
                if show_neighbors:
                    neighbor_indices = []
                    if mass_index > 0:
                        neighbor_indices.append(mass_index - 1)
                    if mass_index < n_masses - 1:
                        neighbor_indices.append(mass_index)

                    if not neighbor_indices:
                        st.info("No neighboring springs for the selected mass.")
                    else:
                        for neighbor_idx in neighbor_indices:
                            st.markdown(
                                f"**Neighbor spring {neighbor_idx}** "
                                f"(between masses {neighbor_idx + 1} and {neighbor_idx + 2})"
                            )
                            neighbor_fig = create_spring_plots(df, neighbor_idx)
                            if neighbor_fig is None:
                                st.info(
                                    f"Spring {neighbor_idx} response columns are not available."
                                )
                            else:
                                safe_plotly_chart(st, neighbor_fig, width="stretch")

    with tab_nodal:
        st.markdown("### Nodal fields (node vs time)")

        # --------- Paper default ---------
        # heatmap + contours + log_color + to_g

        import re

        mass_ids = []
        for c in df.columns:
            mm = re.match(r"Mass(\d+)_Position_x_m$", str(c))
            if mm:
                mass_ids.append(int(mm.group(1)))
        n_masses_cols = max(mass_ids) if mass_ids else 0
        n_masses = int(
            n_masses_cols
            or df.attrs.get("n_masses", 0)
            or params.get("n_masses", len(params.get("masses", [])))
            or 0
        )

        if n_masses <= 0:
            st.info("No nodal (per-mass) histories available in these results.")
        else:
            row1 = st.columns(4)
            with row1[0]:
                quantity = st.selectbox(
                    "Quantity",
                    ["acceleration", "velocity", "position"],
                    index=0,
                    format_func=lambda s: s.title(),
                    key="ui_nodal_quantity",
                )
            with row1[1]:
                component = st.selectbox(
                    "Component",
                    ["magnitude", "x", "y"],
                    index=0,
                    format_func=lambda s: s.title(),
                    key="ui_nodal_component",
                )
            with row1[2]:
                plot_type = st.selectbox(
                    "Plot type",
                    ["heatmap", "surface"],
                    index=0,  # paper default
                    format_func=lambda s: "2D heatmap (paper)" if s == "heatmap" else "3D surface (wow)",
                    key="ui_nodal_plot_type",
                )
            with row1[3]:
                time_unit = st.selectbox(
                    "Time axis",
                    ["ms", "s"],
                    index=0,  # paper default
                    format_func=lambda s: "ms" if s == "ms" else "s",
                    key="ui_nodal_time_unit",
                )

            row2 = st.columns(5)
            with row2[0]:
                log_color = st.checkbox("Log color (log10|·|)", value=True, key="ui_nodal_log")
            with row2[1]:
                to_g = st.checkbox("Acceleration in g", value=True, key="ui_nodal_to_g")
            with row2[2]:
                add_contours = st.checkbox(
                    "Contours",
                    value=True,  # paper default
                    disabled=(plot_type != "heatmap"),
                    key="ui_nodal_contours",
                )
            with row2[3]:
                downsample_mode = st.selectbox(
                    "Downsampling",
                    ["impact", "uniform"],
                    index=0,  # paper default
                    format_func=lambda s: "Impact-focused" if s == "impact" else "Uniform",
                    key="ui_nodal_downsample_mode",
                )
            with row2[4]:
                impact_window_ms = st.number_input(
                    "Impact window (ms)",
                    min_value=1.0,
                    max_value=300.0,
                    value=30.0,
                    step=5.0,
                    disabled=(downsample_mode != "impact"),
                    key="ui_nodal_impact_window_ms",
                )

            row3 = st.columns(4)
            with row3[0]:
                max_time_points = st.number_input(
                    "Max time points",
                    min_value=200,
                    max_value=5000,
                    value=1400 if plot_type == "heatmap" else 800,
                    step=100,
                    key="ui_nodal_max_t",
                )
            with row3[1]:
                max_nodes_cap = min(400, max(1, int(n_masses)))
                default_nodes = min(120, max_nodes_cap)

                prev_nodes = st.session_state.get("ui_nodal_max_nodes", default_nodes)
                try:
                    prev_nodes = int(prev_nodes)
                except Exception:
                    prev_nodes = default_nodes
                prev_nodes = max(1, min(prev_nodes, max_nodes_cap))
                st.session_state["ui_nodal_max_nodes"] = prev_nodes

                max_nodes = st.number_input(
                    "Max nodes",
                    min_value=1,
                    max_value=max_nodes_cap,
                    value=prev_nodes,
                    step=1,
                    key="ui_nodal_max_nodes",
                )
            with row3[2]:
                lock_scale = st.checkbox(
                    "Fixed color scale",
                    value=True,  # paper default
                    key="ui_nodal_lock_scale",
                )
            with row3[3]:
                if st.button("Reset scale", width="stretch", key="ui_nodal_reset_scale"):
                    # Clear all stored scales
                    for k in list(st.session_state.keys()):
                        if str(k).startswith("ui_nodal_scale::"):
                            del st.session_state[k]

            # Fixed color scale (robust quantile-based)
            cmin = None
            cmax = None
            if lock_scale:
                scale_key = f"ui_nodal_scale::{quantity}::{component}::{int(log_color)}::{int(to_g)}"
                if scale_key not in st.session_state:
                    est = estimate_nodal_field_color_range(
                        df,
                        quantity=quantity,
                        component=component,
                        log_color=log_color,
                        to_g=to_g,
                    )
                    if est is not None:
                        st.session_state[scale_key] = tuple(est)

                est = st.session_state.get(scale_key, None)
                if est is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        cmin = st.number_input(
                            "cmin",
                            value=float(est[0]),
                            key=f"{scale_key}::cmin",
                            format="%.6f",
                        )
                    with c2:
                        cmax = st.number_input(
                            "cmax",
                            value=float(est[1]),
                            key=f"{scale_key}::cmax",
                            format="%.6f",
                        )
                    # Persist edits
                    st.session_state[scale_key] = (float(cmin), float(cmax))

            fig = create_nodal_field_surface(
                df,
                quantity=quantity,
                component=component,
                plot_type=plot_type,
                log_color=log_color,
                to_g=to_g,
                time_unit=time_unit,
                add_contours=bool(add_contours),
                cmin=cmin,
                cmax=cmax,
                downsample_mode=downsample_mode,
                impact_window_ms=float(impact_window_ms),
                max_time_points=int(max_time_points),
                max_nodes=int(max_nodes),
            )

            if fig is None:
                st.info(
                    "Required nodal columns were not found (need per-mass position/velocity/acceleration exports)."
                )
            else:
                safe_plotly_chart(st, fig, width="stretch")

            st.markdown("#### Export figure")
            if plot_type != "heatmap":
                st.caption("Export is currently enabled for the 2D heatmap (paper view). Switch plot type to heatmap.")
            else:
                export_col1, export_col2 = st.columns(2)
                base = f"nodal_{quantity}_{component}_{time_unit}"

                png_bytes = export_nodal_field_heatmap_bytes(
                    df,
                    quantity=quantity,
                    component=component,
                    log_color=log_color,
                    to_g=to_g,
                    time_unit=time_unit,
                    add_contours=bool(add_contours),
                    cmin=cmin,
                    cmax=cmax,
                    max_time_points=int(max_time_points),
                    max_nodes=int(max_nodes),
                    downsample_mode=downsample_mode,
                    impact_window_ms=float(impact_window_ms),
                    fmt="png",
                    dpi=300,
                )

                svg_bytes = export_nodal_field_heatmap_bytes(
                    df,
                    quantity=quantity,
                    component=component,
                    log_color=log_color,
                    to_g=to_g,
                    time_unit=time_unit,
                    add_contours=bool(add_contours),
                    cmin=cmin,
                    cmax=cmax,
                    max_time_points=int(max_time_points),
                    max_nodes=int(max_nodes),
                    downsample_mode=downsample_mode,
                    impact_window_ms=float(impact_window_ms),
                    fmt="svg",
                    dpi=300,
                )

                if png_bytes is not None:
                    safe_download_button(
                        export_col1,
                        "🖼️ Export PNG (300 dpi)",
                        png_bytes,
                        f"{base}.png",
                        "image/png",
                        width="stretch",
                    )
                else:
                    export_col1.info("PNG export not available (missing nodal data).")

                if svg_bytes is not None:
                    safe_download_button(
                        export_col2,
                        "🧾 Export SVG",
                        svg_bytes,
                        f"{base}.svg",
                        "image/svg+xml",
                        width="stretch",
                    )
                else:
                    export_col2.info("SVG export not available (missing nodal data).")

