"""
Parameter input UI components for the Railway Impact Simulator.

Provides the sidebar interface for configuring simulation parameters including:
- Train configuration (masses, positions, velocities, stiffnesses)
- Contact model selection and parameters
- Friction model selection and parameters
- Material properties (Bouc-Wen hysteresis, Rayleigh damping)
- Solver settings (time step, integration method, tolerances)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from railway_simulator.core.engine import TrainBuilder, TrainConfig



def _invalidate_sim_cache() -> None:
    """Clear cached simulation outputs when parameters/config change."""
    for k in ("sim_results", "sim_params_core"):
        st.session_state.pop(k, None)


def _normalize_yaml_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize YAML-loaded configuration for internal use (types/arrays)."""
    out: Dict[str, Any] = dict(cfg) if isinstance(cfg, dict) else {}

    # arrays
    for k in ("masses", "x_init", "y_init", "fy", "uy"):
        if k in out and out.get(k) is not None:
            try:
                out[k] = np.asarray(out[k], dtype=float)
            except Exception:
                pass

    # scalars / tuples
    if "n_masses" in out and out.get("n_masses") is not None:
        try:
            out["n_masses"] = int(out["n_masses"])
        except Exception:
            pass

    if "T_int" in out and isinstance(out.get("T_int"), (list, tuple)) and len(out["T_int"]) == 2:
        try:
            out["T_int"] = (float(out["T_int"][0]), float(out["T_int"][1]))
        except Exception:
            pass

    return out


def _get_cfg_dir() -> Path:
    """Locate ./configs directory both when running from repo root and from source tree."""
    cfg_dir = Path.cwd() / "configs"
    if cfg_dir.is_dir():
        return cfg_dir
    return Path(__file__).resolve().parents[3] / "configs"


def _load_active_yaml_cfg() -> Dict[str, Any] | None:
    """If YAML mode is active, load the selected YAML and store it in session_state."""
    if st.session_state.get("train_config_mode") != "YAML example configs":
        return None

    cfg_dir = _get_cfg_dir()
    yaml_files = sorted(cfg_dir.glob("*.yml"))
    if not yaml_files:
        return None

    idx = int(st.session_state.get("yaml_config_idx", 0) or 0)
    idx = max(0, min(idx, len(yaml_files) - 1))
    yaml_path = yaml_files[idx]

    try:
        d = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(d, dict):
            d = {}
    except Exception:
        d = {}

    d = _normalize_yaml_cfg(d)
    st.session_state["yaml_example_cfg"] = d
    st.session_state["yaml_config_path"] = str(yaml_path)
    return d


def _sync_time_widgets_from_yaml(cfg: Dict[str, Any]) -> None:
    """Sync sidebar time/integration widgets to YAML values (so UI matches what runs)."""
    def _set_if_present(key: str, value):
        if value is None:
            return
        st.session_state[key] = value

    # velocity (km/h) from v0_init (m/s)
    v0 = cfg.get("v0_init", None)
    if v0 is not None:
        try:
            v0_kmh = float(abs(float(v0)) * 3.6)
            v0_kmh = float(np.clip(v0_kmh, 10.0, 200.0))
            _set_if_present("ui_v0_kmh", int(round(v0_kmh)))
        except Exception:
            pass

    # time step (ms) from h_init (s)
    h = cfg.get("h_init", None)
    if h is not None:
        try:
            h_ms = float(float(h) * 1000.0)
            h_ms = float(np.clip(h_ms, 0.01, 1.0))
            _set_if_present("ui_h_ms", h_ms)
        except Exception:
            pass

    # max time (s)
    T_max = cfg.get("T_max", None)
    if T_max is not None:
        try:
            _set_if_present("ui_T_max", float(np.clip(float(T_max), 0.1, 10.0)))
        except Exception:
            pass

    # initial gap (cm) from d0 (m)
    d0 = cfg.get("d0", None)
    if d0 is not None:
        try:
            d0_cm = float(float(d0) * 100.0)
            d0_cm = float(np.clip(d0_cm, 0.0, 100.0))
            _set_if_present("ui_d0_cm", d0_cm)
        except Exception:
            pass

    # impact angle (deg) from angle_rad
    ang = cfg.get("angle_rad", None)
    if ang is not None:
        try:
            ang_deg = float(float(ang) * 180.0 / np.pi)
            ang_deg = float(np.clip(ang_deg, 0.0, 45.0))
            _set_if_present("ui_angle_deg", ang_deg)
        except Exception:
            pass

    # HHT-alpha
    alpha = cfg.get("alpha_hht", None)
    if alpha is not None:
        try:
            a = float(np.clip(float(alpha), -0.3, 0.0))
            _set_if_present("ui_alpha_hht", a)
        except Exception:
            pass

    # solver and related
    solver = cfg.get("solver", None)
    if solver is not None:
        solver = str(solver).strip().lower()
        if solver not in ("newton", "picard"):
            solver = "newton"
        _set_if_present("ui_solver", solver)

    jac_mode = cfg.get("newton_jacobian_mode", None)
    if jac_mode is not None:
        jac_mode = str(jac_mode).strip().lower()
        if jac_mode not in ("per_step", "each_iter"):
            jac_mode = "per_step"
        _set_if_present("ui_newton_jacobian_mode", jac_mode)

    tol = cfg.get("newton_tol", None)
    if tol is not None:
        try:
            _set_if_present("ui_newton_tol", float(tol))
        except Exception:
            pass

    mi = cfg.get("max_iter", None)
    if mi is not None:
        try:
            _set_if_present("ui_max_iter", int(mi))
        except Exception:
            pass



def build_parameter_ui() -> Dict[str, Any]:
    """Build parameter input UI in sidebar."""
    with st.sidebar:
        st.header("⚙️ Parameters")

        # YAML mode pre-pass: load selected YAML early so the sidebar widgets match what will run.
        yaml_cfg = _load_active_yaml_cfg()
        yaml_active = isinstance(yaml_cfg, dict) and bool(yaml_cfg)

        apply_full = bool(st.session_state.get("yaml_apply_full", False))
        use_time = bool(st.session_state.get("yaml_use_time", True))
        use_material = bool(st.session_state.get("yaml_use_material", True))
        use_contact = bool(st.session_state.get("yaml_use_contact", True))

        if yaml_active:
            fp = f"{st.session_state.get('yaml_config_path', '')}|{int(apply_full)}|{int(use_time)}|{int(use_material)}|{int(use_contact)}"
            if st.session_state.get("_yaml_fp") != fp:
                st.session_state["_yaml_fp"] = fp
                _invalidate_sim_cache()
                if apply_full or use_time:
                    _sync_time_widgets_from_yaml(yaml_cfg)
        else:
            st.session_state.pop("_yaml_fp", None)

        params: Dict[str, Any] = {}

        # Time & Integration
        time_lock = yaml_active and (apply_full or use_time)
        with st.expander("🕐 Time & Integration", expanded=True):
            if time_lock:
                st.caption("🔒 Locked by YAML (Time & Integration)")

            v0_kmh = st.slider(
                "Impact Velocity (km/h)",
                10,
                200,
                56,
                1,
                key="ui_v0_kmh",
                on_change=_invalidate_sim_cache,
                disabled=time_lock,
            )
            params["v0_init"] = -v0_kmh / 3.6

            h_ms = st.number_input(
                "Time Step Δt (ms)",
                0.01,
                1.0,
                0.1,
                0.01,
                help="Time step size in milliseconds",
                key="ui_h_ms",
                on_change=_invalidate_sim_cache,
                disabled=time_lock,
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
                key="ui_T_max",
                on_change=_invalidate_sim_cache,
                disabled=time_lock,
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
                key="ui_d0_cm",
                on_change=_invalidate_sim_cache,
                disabled=time_lock,
            )
            params["d0"] = d0_cm / 100.0

            angle_deg = st.number_input(
                "Impact Angle (°)",
                0.0,
                45.0,
                0.0,
                0.1,
                key="ui_angle_deg",
                on_change=_invalidate_sim_cache,
                disabled=time_lock,
            )
            params["angle_rad"] = angle_deg * np.pi / 180

            params["alpha_hht"] = st.slider(
                "HHT-α parameter",
                -0.3,
                0.0,
                -0.1,
                0.01,
                help="Negative values add numerical damping",
                key="ui_alpha_hht",
                on_change=_invalidate_sim_cache,
                disabled=time_lock,
            )

            params["solver"] = st.selectbox(
                "Nonlinear solver",
                options=["newton", "picard"],
                index=0,
                format_func=lambda s: "Newton–Raphson (recommended)" if s == "newton" else "Picard (legacy)",
                help=(
                    "Choose the nonlinear solver used inside each implicit HHT-α step. "
                    "Newton is more robust near contact/strong nonlinearity; Picard is a legacy fixed-point iteration."
                ),
                key="ui_solver",
                on_change=_invalidate_sim_cache,
                disabled=time_lock,
            )

            # Newton Jacobian update strategy (only relevant for Newton solver)
            if params["solver"] == "newton":
                params["newton_jacobian_mode"] = st.selectbox(
                    "Newton Jacobian",
                    options=["per_step", "each_iter"],
                    index=0,
                    format_func=lambda s: "Per step (fast)" if s == "per_step" else "Each iteration (pure, slow)",
                    help=(
                        "Controls how often the Newton Jacobian is rebuilt when using the finite-difference Jacobian. "
                        "'Per step' reuses J within the time step (modified Newton, much faster). "
                        "'Each iteration' rebuilds J every Newton iteration (pure but very slow)."
                    ),
                    key="ui_newton_jacobian_mode",
                    on_change=_invalidate_sim_cache,
                    disabled=time_lock,
                )
            else:
                params["newton_jacobian_mode"] = "per_step"

            params["newton_tol"] = st.number_input(
                "Convergence tolerance",
                1e-8,
                1e-2,
                1e-4,
                format="%.1e",
                key="ui_newton_tol",
                on_change=_invalidate_sim_cache,
                disabled=time_lock,
            )

            params["max_iter"] = st.number_input(
                "Max iterations",
                5,
                100,
                50,
                1,
                key="ui_max_iter",
                on_change=_invalidate_sim_cache,
                disabled=time_lock,
            )

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
                "newton_jacobian_mode",
                "solver",
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
                "friction_model",
                "mu_s",
                "mu_k",
                "sigma_0",
                "sigma_1",
                "sigma_2",
                # building keys (optional)
                "building_enable",
                "building_mass",
                "building_zeta",
                "building_height",
                "building_model",
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
            key="train_config_mode",
            on_change=_invalidate_sim_cache,
        )

        # Clear YAML selection when leaving YAML mode (avoid stale overrides).
        if config_mode != "YAML example configs":
            # Clear YAML selection when leaving YAML mode (avoid stale overrides).
            for k in (
                "yaml_example_cfg",
                "yaml_apply_full",
                "yaml_use_time",
                "yaml_use_material",
                "yaml_use_contact",
                "yaml_config_path",
                "yaml_config_idx",
                "_yaml_fp",
            ):
                st.session_state.pop(k, None)
            _invalidate_sim_cache()

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
            parsed.append(_normalize_yaml_cfg(d))

        idx = st.selectbox(
            "YAML config",
            options=list(range(len(yaml_files))),
            format_func=lambda i: labels[i],
            index=0,
            key="yaml_config_idx",
            on_change=_invalidate_sim_cache,
        )

        yaml_path = yaml_files[idx]
        yaml_cfg = parsed[idx] if isinstance(parsed[idx], dict) else {}

        st.session_state["yaml_example_cfg"] = yaml_cfg
        st.session_state["yaml_config_path"] = str(yaml_path)

        case_name = yaml_cfg.get("case_name", yaml_path.stem)

        st.caption(f"Selected YAML: **{yaml_path.name}**  ·  case_name: **{case_name}**")

        apply_full = st.checkbox(
            "Apply full YAML config (overrides all sidebar params)",
            value=bool(st.session_state.get("yaml_apply_full", False)),
            help="If enabled, the simulation will run exactly with the YAML values (time/material/contact/etc).",
            key="yaml_apply_full",
            on_change=_invalidate_sim_cache,
        )

        use_time = st.checkbox(
            "Use time/integration values from YAML (v0_init, T_max, h_init, α, tol, ...)",
            value=bool(st.session_state.get("yaml_use_time", True)),
            disabled=apply_full,
            key="yaml_use_time",
            on_change=_invalidate_sim_cache,
        )
        use_material = st.checkbox(
            "Use train material values from YAML (fy/uy/Bouc–Wen)",
            value=bool(st.session_state.get("yaml_use_material", True)),
            disabled=apply_full,
            key="yaml_use_material",
            on_change=_invalidate_sim_cache,
        )
        use_contact = st.checkbox(
            "Use contact/friction values from YAML (k_wall/contact model/μ)",
            value=bool(st.session_state.get("yaml_use_contact", True)),
            disabled=apply_full,
            key="yaml_use_contact",
            on_change=_invalidate_sim_cache,
        )

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


