#!/usr/bin/env python3
"""Deprecated legacy generator for the historical locomotive/full-train table.

TODO(legacy): keep this script frozen for reproducibility. New production
workflows should use the neutral Project Workbench / study services instead.

This script keeps the deterministic runout/geometric filter separate from the
1D/2D reduced multibody impact model.  The engine receives the wall-normal
impact velocity ``v_n``; the runout path and beta cap only determine that input.

Scope: reduced multibody proxy / parametric full-consist model.  The generated
TRAXX and ICE4 consist definitions are not calibrated against crash-test data.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from railway_simulator.config.loader import load_simulation_config
from railway_simulator.core.engine import get_default_simulation_params, run_simulation
from railway_simulator.hazard.sdof import equivalent_static_force_sdof

G = 9.80665


@dataclass(frozen=True)
class VehicleGeometry:
    name: str
    L_m: float
    Dzpa_m: float
    b_m: float
    beta_train_cap_deg: float

    @property
    def l_eff_m(self) -> float:
        return 0.5 * (self.L_m + self.Dzpa_m)


TRAXX_GEOM = VehicleGeometry(
    name="TRAXX BR187",
    L_m=18.90,
    Dzpa_m=10.44,
    b_m=2.977,
    beta_train_cap_deg=7.2,
)

ICE4_GEOM = VehicleGeometry(
    name="ICE4 Endwagen proxy",
    L_m=29.11,
    Dzpa_m=19.10,
    b_m=2.852,
    beta_train_cap_deg=10.7,
)


class PicardWarningCounter(logging.Handler):
    """Logging handler that counts Picard non-convergence warnings."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - tiny handler
        msg = record.getMessage()
        if "Picard solver did not converge" in msg:
            self.count += 1
            if len(self.messages) < 10:
                self.messages.append(msg)


@contextlib.contextmanager
def capture_picard_warnings() -> Iterable[PicardWarningCounter]:
    logger = logging.getLogger("railway_simulator.core.engine")
    old_level = logger.level
    old_propagate = logger.propagate
    handler = PicardWarningCounter()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def _round_step(T_max: float, h: float) -> int:
    return int(round(float(T_max) / float(h)))


def _base_engine_settings() -> dict[str, Any]:
    cfg = load_simulation_config(ROOT / "configs" / "traxx_freight.yml")
    cfg.update(
        {
            "solver": "picard",
            "h_init": 5.0e-5,
            "T_max": 0.15,
            "T_int": (0.0, 0.15),
            "step": _round_step(0.15, 5.0e-5),
            "picard_max_iters": 30,
            "picard_tol": 1.0e-5,
            "contact_model": "lankarani-nikravesh",
            "emit_peak_diagnostics": False,
            "angle_rad": 0.0,
            # Keep the leading lumped mass just outside the wall at t=0.
            "d0": 0.01,
        }
    )
    return cfg


def _with_time_grid(cfg: dict[str, Any], *, T_max: float = 0.15, h: float = 5.0e-5) -> dict[str, Any]:
    out = dict(cfg)
    out["h_init"] = float(h)
    out["T_max"] = float(T_max)
    out["T_int"] = (0.0, float(T_max))
    out["step"] = _round_step(T_max, h)
    return out


def _k_train_from_fy_uy(fy: list[float], uy: list[float]) -> list[float]:
    return [float(f) / float(u) if float(u) != 0.0 else 0.0 for f, u in zip(fy, uy)]


def build_traxx_lok_solo() -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    cfg = _base_engine_settings()
    cfg["case_name"] = "traxx_br187_lok_solo"
    fy = [float(x) for x in cfg["fy"]]
    uy = [float(x) for x in cfg["uy"]]
    cfg["k_train"] = _k_train_from_fy_uy(fy, uy)
    meta = {
        "mass_total_t": float(sum(cfg["masses"]) / 1000.0),
        "n_masses": int(cfg["n_masses"]),
        "coupling_spring_indices_1based": [],
        "assumptions": ["Existing configs/traxx_freight.yml distribution preserved for Lok_solo."],
    }
    return cfg, [], meta


def build_traxx_full_train() -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    cfg = _base_engine_settings()
    masses = [float(x) for x in cfg["masses"]]
    x = [float(v) for v in cfg["x_init"]]
    y = [float(v) for v in cfg["y_init"]]
    fy = [float(v) for v in cfg["fy"]]
    uy = [float(v) for v in cfg["uy"]]

    # Prompt-defined coupling.  Wagon internal springs are kept equal to the
    # existing TRAXX freight spring law to avoid unconnected wagon lumped masses.
    gap_between_units = 0.5
    wagon_length_assumed_m = 18.0
    wagon_internal_fy = 20.0e6
    wagon_internal_uy = 0.05
    fy_coupling = 2.5e6
    uy_coupling = 0.05
    coupling_indices: list[int] = []

    last_x = x[-1]
    for _wagon in range(4):
        start = last_x + gap_between_units
        x.extend([start, start + 0.5 * wagon_length_assumed_m, start + wagon_length_assumed_m])
        y.extend([0.0, 0.0, 0.0])
        masses.extend([20_000.0, 40_000.0, 20_000.0])
        # Spring order: coupling to previous unit, then two intra-wagon springs.
        coupling_indices.append(len(fy) + 1)  # 1-based spring index in exported DataFrame.
        fy.extend([fy_coupling, wagon_internal_fy, wagon_internal_fy])
        uy.extend([uy_coupling, wagon_internal_uy, wagon_internal_uy])
        last_x = start + wagon_length_assumed_m

    cfg.update(
        {
            "case_name": "traxx_br187_full_gueterzug_proxy",
            "n_masses": len(masses),
            "masses": masses,
            "x_init": x,
            "y_init": y,
            "fy": fy,
            "uy": uy,
            "k_train": _k_train_from_fy_uy(fy, uy),
        }
    )
    meta = {
        "mass_total_t": float(sum(masses) / 1000.0),
        "n_masses": len(masses),
        "coupling_spring_indices_1based": coupling_indices,
        "fy_coupling_N": fy_coupling,
        "uy_coupling_m": uy_coupling,
        "gap_between_units_m": gap_between_units,
        "assumed_wagon_length_m": wagon_length_assumed_m,
        "assumptions": [
            "TRAXX locomotive mass distribution preserved from configs/traxx_freight.yml.",
            "Each 80 t wagon is represented by 3 masses [20, 40, 20] t.",
            "Wagon length is not specified in the prompt; 18.0 m is used only for initial spring lengths.",
            "Wagon internal springs use the existing TRAXX freight spring law; inter-unit couplings use the prompt fy/uy.",
        ],
    }
    return cfg, coupling_indices, meta


def _ice_base_engine_settings() -> dict[str, Any]:
    # Reuse wall/contact/friction/numerics from the TRAXX contact-fix benchmark
    # so the ICE4 proxy differs primarily through mass layout and geometry.
    cfg = _base_engine_settings()
    return cfg


def build_ice4_lok_solo() -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    cfg = _ice_base_engine_settings()
    L = ICE4_GEOM.L_m
    masses = [15_000.0, 30_000.0, 15_000.0]
    x = [0.0, 0.5 * L, L]
    y = [0.0, 0.0, 0.0]
    # ICE4 proxy internal crush links: intentionally generic, not calibrated.
    fy = [15.0e6, 15.0e6]
    uy = [0.20, 0.20]
    cfg.update(
        {
            "case_name": "ice4_endwagen_solo_proxy",
            "n_masses": len(masses),
            "masses": masses,
            "x_init": x,
            "y_init": y,
            "fy": fy,
            "uy": uy,
            "k_train": _k_train_from_fy_uy(fy, uy),
            "bw_a": 1.0,
            "bw_A": 1.0,
            "bw_beta": 0.5,
            "bw_gamma": 0.5,
            "bw_n": 2,
        }
    )
    meta = {
        "mass_total_t": 60.0,
        "n_masses": len(masses),
        "coupling_spring_indices_1based": [],
        "assumptions": [
            "ICE4 Endwagen solo is a 60 t / 3-mass reduced proxy.",
            "Internal crush links are generic linear proxy links, not calibrated ICE4 data.",
        ],
    }
    return cfg, [], meta


def build_ice4_full_train() -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    cfg = _ice_base_engine_settings()
    end_L = ICE4_GEOM.L_m
    middle_length_assumed_m = 24.0
    gap_between_units = 0.3
    fy_coupling = 2.0e6
    uy_coupling = 0.05
    internal_fy = 15.0e6
    internal_uy = 0.20

    masses = [15_000.0, 30_000.0, 15_000.0]
    x = [0.0, 0.5 * end_L, end_L]
    y = [0.0, 0.0, 0.0]
    fy = [internal_fy, internal_fy]
    uy = [internal_uy, internal_uy]
    coupling_indices: list[int] = []

    last_x = x[-1]
    for _middle in range(2):
        start = last_x + gap_between_units
        x.extend([start, start + middle_length_assumed_m])
        y.extend([0.0, 0.0])
        masses.extend([22_500.0, 22_500.0])
        coupling_indices.append(len(fy) + 1)
        fy.extend([fy_coupling, internal_fy])
        uy.extend([uy_coupling, internal_uy])
        last_x = start + middle_length_assumed_m

    cfg.update(
        {
            "case_name": "ice4_full_proxy_endwagen_plus_2_mittelwagen",
            "n_masses": len(masses),
            "masses": masses,
            "x_init": x,
            "y_init": y,
            "fy": fy,
            "uy": uy,
            "k_train": _k_train_from_fy_uy(fy, uy),
            "bw_a": 1.0,
            "bw_A": 1.0,
            "bw_beta": 0.5,
            "bw_gamma": 0.5,
            "bw_n": 2,
        }
    )
    meta = {
        "mass_total_t": float(sum(masses) / 1000.0),
        "n_masses": len(masses),
        "coupling_spring_indices_1based": coupling_indices,
        "fy_coupling_N": fy_coupling,
        "uy_coupling_m": uy_coupling,
        "gap_between_units_m": gap_between_units,
        "assumed_middle_car_length_m": middle_length_assumed_m,
        "assumptions": [
            "ICE4 full proxy is Endwagen + 2 Mittelwagen, total 150 t / 7 masses.",
            "Each Mittelwagen is represented by 2 masses [22.5, 22.5] t.",
            "Mittelwagen length is not specified in the prompt; 24.0 m is used only for initial spring lengths.",
            "Internal crush links are generic proxy links; inter-unit couplings use the prompt fy/uy.",
        ],
    }
    return cfg, coupling_indices, meta


def beta_eff_for_distance(a_m: float, geom: VehicleGeometry, *, apply_train_cap: bool = True) -> tuple[float | None, str]:
    a = float(a_m)
    if a <= 0.5 * geom.b_m:
        return None, "geometry_invalid"
    arg = (a - 0.5 * geom.b_m) / geom.l_eff_m
    arg = min(1.0, max(0.0, arg))
    beta_max = math.asin(arg)
    if apply_train_cap:
        beta_eff = min(beta_max, math.radians(geom.beta_train_cap_deg))
    else:
        beta_eff = beta_max
    return beta_eff, "ok"


def runout(v0_kmh: float, a_m: float, mu: float, beta_eff_rad: float) -> tuple[float, float]:
    v0 = float(v0_kmh) / 3.6
    s = float(a_m) / max(math.sin(beta_eff_rad), 1.0e-15)
    v_imp = math.sqrt(max(v0 * v0 - 2.0 * float(mu) * G * s, 0.0))
    vn = v_imp * math.sin(beta_eff_rad)
    return v_imp, vn


def pad_force_history(time: np.ndarray, force: np.ndarray, *, pad_to_s: float = 0.50) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float)
    f = np.asarray(force, dtype=float)
    if t.size < 2:
        return t, f
    dt = float(np.median(np.diff(t)))
    if t[-1] >= pad_to_s:
        return t, f
    tail = np.arange(t[-1] + dt, pad_to_s + 0.5 * dt, dt)
    if tail.size == 0:
        return t, f
    return np.concatenate([t, tail]), np.concatenate([f, np.zeros_like(tail)])


def coupling_diagnostics(df: pd.DataFrame, coupling_indices_1based: list[int]) -> dict[str, Any]:
    if not coupling_indices_1based:
        return {
            "max_coupling_force_N": np.nan,
            "coupling_activation_time_s": np.nan,
            "coupling_max_displacement_m": np.nan,
            "coupling_diagnostics_note": "no coupling springs for solo mode",
        }

    force_cols = [f"Spring{i}_Force_N" for i in coupling_indices_1based if f"Spring{i}_Force_N" in df.columns]
    disp_cols = [f"Spring{i}_Disp_m" for i in coupling_indices_1based if f"Spring{i}_Disp_m" in df.columns]
    if not force_cols or not disp_cols:
        return {
            "max_coupling_force_N": np.nan,
            "coupling_activation_time_s": np.nan,
            "coupling_max_displacement_m": np.nan,
            "coupling_diagnostics_note": "coupling diagnostics not exported by current engine",
        }

    f_abs = df[force_cols].abs()
    d_abs = df[disp_cols].abs()
    row_max = f_abs.max(axis=1)
    threshold = 1.0e3
    active = np.flatnonzero(row_max.to_numpy(dtype=float) > threshold)
    activation_time = float(df["Time_s"].iloc[int(active[0])]) if active.size else np.nan
    return {
        "max_coupling_force_N": float(f_abs.max().max()),
        "coupling_activation_time_s": activation_time,
        "coupling_max_displacement_m": float(d_abs.max().max()),
        "coupling_diagnostics_note": "ok",
    }


def run_engine_row(
    *,
    base_cfg: dict[str, Any],
    coupling_indices_1based: list[int],
    vn_ms: float,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    cfg = _with_time_grid(base_cfg)
    cfg.update(
        {
            "v0_init": -float(vn_ms),
            "angle_rad": 0.0,
            "solver": "picard",
            "picard_max_iters": 30,
            "picard_tol": 1.0e-5,
            "contact_model": "lankarani-nikravesh",
        }
    )
    step_count = int(cfg["step"])
    start = time.perf_counter()
    try:
        with capture_picard_warnings() as counter:
            # Suppress optional stdout/stderr chatter from diagnostics; warnings are
            # counted via the logging handler above.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                df = run_simulation(cfg, emit_peak_diagnostics=False)
        elapsed = time.perf_counter() - start
        fpeak = float(df["Impact_Force_MN"].max())
        t_pad, f_pad = pad_force_history(
            df["Time_s"].to_numpy(dtype=float),
            df["Impact_Force_MN"].to_numpy(dtype=float),
            pad_to_s=0.50,
        )
        feq = float(equivalent_static_force_sdof(t_pad, f_pad, Tn_s=0.1, zeta=0.05))
        frac = float(counter.count / max(step_count, 1))
        picard_gt_10pct = bool(frac > 0.10)
        diag = coupling_diagnostics(df, coupling_indices_1based)
        return {
            "engine_status": "ok",
            "Fpeak_MN": fpeak,
            "Feq_100ms_MN": feq,
            "picard_nonconverged_count": int(counter.count),
            "picard_nonconverged_fraction": frac,
            "picard_exhausted_gt_10pct": picard_gt_10pct,
            "runtime_s": elapsed,
            **diag,
        }, df
    except Exception as exc:  # Keep table rows; do not silently drop failures.
        elapsed = time.perf_counter() - start
        return {
            "engine_status": "engine_failed",
            "engine_error": repr(exc),
            "Fpeak_MN": np.nan,
            "Feq_100ms_MN": np.nan,
            "picard_nonconverged_count": np.nan,
            "picard_nonconverged_fraction": np.nan,
            "picard_exhausted_gt_10pct": False,
            "runtime_s": elapsed,
            "max_coupling_force_N": np.nan,
            "coupling_activation_time_s": np.nan,
            "coupling_max_displacement_m": np.nan,
            "coupling_diagnostics_note": "engine failed before diagnostics",
        }, None


def hooke_single_mass_sanity(outdir: Path) -> dict[str, Any]:
    base = get_default_simulation_params()
    m = 13_000.0
    k = 45.0e6
    vn = 4.0
    T_max = 0.20
    h = 2.5e-5
    cfg = dict(base)
    cfg.update(
        {
            "n_masses": 1,
            "masses": [m],
            "x_init": [0.0],
            "y_init": [0.0],
            "fy": [],
            "uy": [],
            "k_train": [],
            "v0_init": -vn,
            "angle_rad": 0.0,
            "d0": 0.01,
            "k_wall": k,
            "cr_wall": 1.0,
            "contact_model": "hooke",
            "friction_model": "none",
            "mu_s": 0.0,
            "mu_k": 0.0,
            "sigma_0": 0.0,
            "sigma_1": 0.0,
            "sigma_2": 0.0,
            "solver": "picard",
            "h_init": h,
            "T_max": T_max,
            "T_int": (0.0, T_max),
            "step": _round_step(T_max, h),
            "picard_max_iters": 30,
            "picard_tol": 1.0e-6,
        }
    )
    with capture_picard_warnings():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = run_simulation(cfg, emit_peak_diagnostics=False)
    fpeak_N = float(df["Impact_Force_MN"].max() * 1.0e6)
    expected_N = float(vn * math.sqrt(k * m))
    rel_error = abs(fpeak_N - expected_N) / expected_N
    ok = bool(rel_error <= 0.05)
    df[["Time_s", "Impact_Force_MN", "Penetration_mm"]].to_csv(
        outdir / "hooke_single_mass_sanity_history.csv", index=False
    )
    return {
        "model": "single_mass_hooke",
        "mass_kg": m,
        "k_wall_N_per_m": k,
        "vn_ms": vn,
        "Fpeak_engine_MN": fpeak_N / 1.0e6,
        "Fpeak_expected_MN": expected_N / 1.0e6,
        "relative_error": rel_error,
        "passed_5pct": ok,
    }


def add_ratios_and_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Fpeak_ratio"] = np.nan
    out["Feq_ratio"] = np.nan
    out["significant_train_effect"] = False

    key_cols = ["Fahrzeug", "a_m", "v0_kmh"]
    for keys, sub in out.groupby(key_cols, sort=False):
        idx_lok = sub.index[sub["mode"] == "Lok_solo"].tolist()
        idx_zug = sub.index[sub["mode"] == "Zug_full"].tolist()
        if not idx_lok or not idx_zug:
            continue
        i_lok, i_zug = idx_lok[0], idx_zug[0]
        lok_ok = out.at[i_lok, "engine_status"] == "ok"
        zug_ok = out.at[i_zug, "engine_status"] == "ok"
        if lok_ok and zug_ok:
            fpeak_lok = float(out.at[i_lok, "Fpeak_MN"])
            feq_lok = float(out.at[i_lok, "Feq_100ms_MN"])
            fpeak_zug = float(out.at[i_zug, "Fpeak_MN"])
            feq_zug = float(out.at[i_zug, "Feq_100ms_MN"])
            fpeak_ratio = fpeak_zug / fpeak_lok if fpeak_lok > 0.0 else np.nan
            feq_ratio = feq_zug / feq_lok if feq_lok > 0.0 else np.nan
            out.loc[[i_lok, i_zug], "Fpeak_ratio"] = fpeak_ratio
            out.loc[[i_lok, i_zug], "Feq_ratio"] = feq_ratio
            out.loc[[i_lok, i_zug], "significant_train_effect"] = bool(fpeak_ratio > 1.3)
        elif (not lok_ok) and zug_ok:
            out.loc[[i_lok, i_zug], "status"] = "ratio_unavailable_lok_failed"
        elif lok_ok and (not zug_ok):
            out.loc[i_zug, "status"] = "zug_engine_failed"
    return out


def make_response_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(["Fahrzeug", "a_m", "v0_kmh"], sort=True):
        fahrzeug, a_m, v0 = keys
        lok = sub[sub["mode"] == "Lok_solo"]
        zug = sub[sub["mode"] == "Zug_full"]
        if lok.empty or zug.empty:
            continue
        lok_row = lok.iloc[0]
        z = zug.iloc[0]
        rows.append(
            {
                "Fahrzeug": fahrzeug,
                "a_m": a_m,
                "v0_kmh": v0,
                "beta_eff_deg": z["beta_eff_deg"],
                "vimp_kmh": z["vimp_kmh"],
                "vn_ms": z["vn_ms"],
                "Fpeak_Lok_solo_MN": lok_row["Fpeak_MN"],
                "Fpeak_Zug_full_MN": z["Fpeak_MN"],
                "Feq_100ms_Lok_solo_MN": lok_row["Feq_100ms_MN"],
                "Feq_100ms_Zug_full_MN": z["Feq_100ms_MN"],
                "Fpeak_ratio": z["Fpeak_ratio"],
                "Feq_ratio": z["Feq_ratio"],
                "significant_train_effect": z["significant_train_effect"],
                "Lok_status": lok_row["status"],
                "Zug_status": z["status"],
                "Zug_max_coupling_force_MN": z["max_coupling_force_N"] / 1.0e6
                if pd.notna(z["max_coupling_force_N"])
                else np.nan,
                "Zug_coupling_activation_time_ms": z["coupling_activation_time_s"] * 1000.0
                if pd.notna(z["coupling_activation_time_s"])
                else np.nan,
                "Zug_coupling_max_displacement_mm": z["coupling_max_displacement_m"] * 1000.0
                if pd.notna(z["coupling_max_displacement_m"])
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def monotonicity_check(df: pd.DataFrame, tol: float = 1.0e-3) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    ok = df[(df["status"] == "ok") & pd.notna(df["Fpeak_MN"])].copy()
    for (veh, mode), sub in ok.groupby(["Fahrzeug", "mode"], sort=True):
        sub = sub.sort_values("vn_ms")
        values = sub["Fpeak_MN"].to_numpy(dtype=float)
        vn = sub["vn_ms"].to_numpy(dtype=float)
        for i in range(1, len(values)):
            if values[i] + tol < values[i - 1]:
                violations.append(
                    {
                        "Fahrzeug": veh,
                        "mode": mode,
                        "prev_vn_ms": float(vn[i - 1]),
                        "prev_Fpeak_MN": float(values[i - 1]),
                        "curr_vn_ms": float(vn[i]),
                        "curr_Fpeak_MN": float(values[i]),
                    }
                )
    return violations


def known_lok_solo_check(df: pd.DataFrame) -> dict[str, Any]:
    row = df[
        (df["Fahrzeug"] == "TRAXX BR187")
        & (df["mode"] == "Lok_solo")
        & (df["a_m"] == 3.0)
        & (df["v0_kmh"] == 160.0)
    ]
    if row.empty:
        return {"passed": False, "reason": "row_missing"}
    r = row.iloc[0]
    fpeak = float(r["Fpeak_MN"])
    feq = float(r["Feq_100ms_MN"])
    return {
        "passed": bool(2.0 <= fpeak <= 4.0),
        "Fpeak_MN": fpeak,
        "Feq_100ms_MN": feq,
        "expected_Fpeak_range_MN": [2.0, 4.0],
        "previous_reference_Fpeak_MN": 2.38,
        "previous_reference_Feq_MN": 3.40,
    }


def _round_for_markdown(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    round_map = {
        "a_m": 3,
        "v0_kmh": 0,
        "beta_eff_deg": 3,
        "vimp_kmh": 3,
        "vn_ms": 3,
        "Fpeak_MN": 3,
        "Feq_100ms_MN": 3,
        "Fpeak_ratio": 3,
        "Feq_ratio": 3,
        "picard_nonconverged_fraction": 5,
        "max_coupling_force_N": 0,
        "coupling_activation_time_s": 5,
        "coupling_max_displacement_m": 5,
    }
    for col, nd in round_map.items():
        if col in out.columns:
            out[col] = out[col].round(nd)
    return out


def write_markdown_outputs(
    *,
    outdir: Path,
    full_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    metadata: dict[str, Any],
    known_check: dict[str, Any],
    mono_violations: list[dict[str, Any]],
    hooke_check: dict[str, Any],
) -> None:
    main_cols = [
        "a_m",
        "Fahrzeug",
        "mode",
        "v0_kmh",
        "beta_eff_deg",
        "vimp_kmh",
        "vn_ms",
        "Fpeak_MN",
        "Feq_100ms_MN",
        "Fpeak_ratio",
        "Feq_ratio",
        "significant_train_effect",
        "status",
    ]
    main_table = _round_for_markdown(full_df[main_cols]).rename(columns={
        "a_m": "a [m]",
        "v0_kmh": "v0 [km/h]",
        "beta_eff_deg": "beta_eff [deg]",
        "vimp_kmh": "vimp [km/h]",
        "vn_ms": "vn [m/s]",
        "Fpeak_MN": "Fpeak [MN]",
        "Feq_100ms_MN": "Feq_100ms [MN]",
    })
    (outdir / "stempi_full_train_comparison_mu0p30.md").write_text(
        "# Stempi full-train comparison, μ = 0.30\n\n"
        "Scope: reduced multibody proxy / parametric full-consist model. "
        "`FEN = 6 MN` is used only as an illustrative equivalent force reference, not as a capacity.\n\n"
        + main_table.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )

    fen_peak = full_df[(full_df["status"] == "ok") & (full_df["Fpeak_MN"] > 6.0)]
    fen_feq = full_df[(full_df["status"] == "ok") & (full_df["Feq_100ms_MN"] > 6.0)]
    picard_rows = full_df[full_df["status"].astype(str).str.contains("picard_warning", na=False)]

    summary_lines: list[str] = []
    summary_lines.append("# Stempi full-train summary\n")
    summary_lines.append("## Scope and modelling status\n")
    summary_lines.append(
        "This is a reduced multibody proxy / parametric full-consist model. "
        "It is not calibrated to real TRAXX or ICE4 crash tests. The added cars do not directly contact the wall; "
        "their effect enters through the inter-unit coupling springs.\n"
    )
    summary_lines.append("The geometric filter and runout model are unchanged from the Stempi runout setup. For side-by-side comparability, the Zug cap is applied to both `Lok_solo` and `Zug_full` rows.\n")
    summary_lines.append("## Generated configurations\n")
    for key, value in metadata["configurations"].items():
        summary_lines.append(f"- `{key}`: mass = {value['mass_total_t']:.1f} t, n_masses = {value['n_masses']}, coupling springs = {value['coupling_spring_indices_1based']}\n")
        for assumption in value.get("assumptions", []):
            summary_lines.append(f"  - {assumption}\n")
    summary_lines.append("\n## Sanity checks\n")
    summary_lines.append(f"- Known TRAXX Lok_solo check: `{known_check}`\n")
    summary_lines.append(f"- Hooke single-mass check: `{hooke_check}`\n")
    if mono_violations:
        summary_lines.append("- Monotonicity check: violations found:\n")
        summary_lines.append(pd.DataFrame(mono_violations).to_markdown(index=False) + "\n")
    else:
        summary_lines.append("- Monotonicity check: passed for all vehicle/mode groups with successful rows.\n")

    summary_lines.append("\n## Engineering interpretation\n")
    successful_grid = grid_df[pd.notna(grid_df["Fpeak_ratio"])].copy()
    if successful_grid.empty:
        summary_lines.append("- No successful Lok/Zug pairs were available for ratio interpretation.\n")
    else:
        max_fpeak_ratio = successful_grid.loc[successful_grid["Fpeak_ratio"].idxmax()]
        max_feq_ratio = successful_grid.loc[successful_grid["Feq_ratio"].idxmax()]
        sig = successful_grid[successful_grid["significant_train_effect"]]
        summary_lines.append(
            f"- Maximum Fpeak ratio: {max_fpeak_ratio['Fpeak_ratio']:.3f} "
            f"({max_fpeak_ratio['Fahrzeug']}, a={max_fpeak_ratio['a_m']} m, v0={max_fpeak_ratio['v0_kmh']} km/h).\n"
        )
        summary_lines.append(
            f"- Maximum Feq ratio: {max_feq_ratio['Feq_ratio']:.3f} "
            f"({max_feq_ratio['Fahrzeug']}, a={max_feq_ratio['a_m']} m, v0={max_feq_ratio['v0_kmh']} km/h).\n"
        )
        if sig.empty:
            summary_lines.append("- No row satisfies `significant_train_effect = Fpeak_ratio > 1.3`. In this run, the added consist mass does not materially increase first-contact peak force.\n")
        else:
            summary_lines.append("- Rows with `significant_train_effect = True`:\n")
            summary_lines.append(_round_for_markdown(sig).to_markdown(index=False) + "\n")
        summary_lines.append("- The train effect appears primarily in coupling-force history, rebound and tail response; Fpeak remains controlled by the leading vehicle/wall contact in this reduced end-on model.\n")

    if fen_peak.empty:
        summary_lines.append("- Rows with Fpeak > illustrative FEN = 6 MN: none.\n")
    else:
        summary_lines.append("- Rows with Fpeak > illustrative FEN = 6 MN:\n")
        summary_lines.append(_round_for_markdown(fen_peak[main_cols]).to_markdown(index=False) + "\n")
    if fen_feq.empty:
        summary_lines.append("- Rows with Feq_100ms > illustrative FEN = 6 MN: none.\n")
    else:
        summary_lines.append("- Rows with Feq_100ms > illustrative FEN = 6 MN:\n")
        summary_lines.append(_round_for_markdown(fen_feq[main_cols]).to_markdown(index=False) + "\n")
    if picard_rows.empty:
        summary_lines.append("- Picard warnings above 10% of steps: none.\n")
    else:
        summary_lines.append("- Rows with Picard warning status:\n")
        summary_lines.append(_round_for_markdown(picard_rows[main_cols + ["picard_nonconverged_fraction"]]).to_markdown(index=False) + "\n")

    summary_lines.append("\n## Output files\n")
    for fname in [
        "stempi_full_train_comparison_full.csv",
        "stempi_full_train_comparison_mu0p30.md",
        "stempi_full_train_summary.md",
        "response_grid_lok_vs_zug.csv",
        "feq_lok_vs_zug.png",
        "fpeak_lok_vs_zug.png",
        "fpeak_ratio_heatmap_traxx.png",
        "fpeak_ratio_heatmap_ice4.png",
    ]:
        summary_lines.append(f"- `{fname}`\n")

    (outdir / "stempi_full_train_summary.md").write_text("".join(summary_lines), encoding="utf-8")


def make_plots(outdir: Path, grid_df: pd.DataFrame) -> None:
    plot_df = grid_df.copy()
    plot_df["case"] = plot_df["a_m"].map(lambda x: f"a={x:g} m") + ", " + plot_df["v0_kmh"].map(lambda x: f"v0={x:g}")

    for metric, fname, ylabel in [
        ("Fpeak", "fpeak_lok_vs_zug.png", "Fpeak [MN]"),
        ("Feq_100ms", "feq_lok_vs_zug.png", "Feq_100ms [MN]"),
    ]:
        fig = plt.figure(figsize=(11.0, 5.2))
        xpos = np.arange(len(plot_df))
        width = 0.38
        plt.bar(xpos - width / 2.0, plot_df[f"{metric}_Lok_solo_MN"], width, label="Lok_solo")
        plt.bar(xpos + width / 2.0, plot_df[f"{metric}_Zug_full_MN"], width, label="Zug_full")
        plt.axhline(6.0, linestyle="--", linewidth=1.0, label="FEN = 6 MN illustrative reference")
        plt.xticks(xpos, plot_df["Fahrzeug"].str.replace(" proxy", "") + "\n" + plot_df["case"], rotation=80, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"{metric}: Lok_solo vs Zug_full")
        plt.legend()
        plt.tight_layout()
        fig.savefig(outdir / fname, dpi=180)
        plt.close(fig)

    for vehicle, fname in [
        ("TRAXX BR187", "fpeak_ratio_heatmap_traxx.png"),
        ("ICE4 Endwagen proxy", "fpeak_ratio_heatmap_ice4.png"),
    ]:
        sub = grid_df[grid_df["Fahrzeug"] == vehicle].copy()
        pivot = sub.pivot(index="a_m", columns="v0_kmh", values="Fpeak_ratio").sort_index()
        fig = plt.figure(figsize=(6.5, 4.8))
        data = pivot.to_numpy(dtype=float)
        im = plt.imshow(data, aspect="auto", origin="lower")
        plt.colorbar(im, label="Fpeak ratio Zug/Lok")
        plt.xticks(np.arange(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
        plt.yticks(np.arange(len(pivot.index)), [f"{i:g}" for i in pivot.index])
        plt.xlabel("v0 [km/h]")
        plt.ylabel("a [m]")
        plt.title(f"Fpeak ratio heatmap: {vehicle}")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i, j]):
                    plt.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center")
        plt.tight_layout()
        fig.savefig(outdir / fname, dpi=180)
        plt.close(fig)


def write_generated_configs(outdir: Path, configs: dict[str, dict[str, Any]]) -> None:
    cfg_dir = outdir / "generated_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs.items():
        safe = name.lower().replace(" ", "_").replace("/", "_")
        serializable = {}
        for k, v in cfg.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, tuple):
                serializable[k] = list(v)
            elif k != "contact_law":
                serializable[k] = v
        with (cfg_dir / f"{safe}.yml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(serializable, f, sort_keys=False)



def evaluate_case_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Evaluate one scenario row. Top-level so it can be used by ProcessPoolExecutor."""
    case_no = int(payload["case_no"])
    total_cases = int(payload["total_cases"])
    vehicle = payload["vehicle"]
    mode = payload["mode"]
    geom: VehicleGeometry = payload["geom"]
    cfg = payload["cfg"]
    cidx = payload["coupling_indices"]
    a = float(payload["a_m"])
    v0 = float(payload["v0_kmh"])
    mu = float(payload["mu"])

    beta_eff, geom_status = beta_eff_for_distance(a, geom, apply_train_cap=True)
    base_row: dict[str, Any] = {
        "a_m": float(a),
        "Fahrzeug": vehicle,
        "mode": mode,
        "v0_kmh": float(v0),
        "mu": mu,
        "beta_eff_deg": np.nan,
        "vimp_kmh": np.nan,
        "vn_ms": np.nan,
        "Fpeak_MN": np.nan,
        "Feq_100ms_MN": np.nan,
        "Fpeak_ratio": np.nan,
        "Feq_ratio": np.nan,
        "significant_train_effect": False,
        "status": geom_status,
        "engine_status": "not_run",
    }
    if beta_eff is None:
        return base_row, f"[{case_no}/{total_cases}] {vehicle} {mode} a={a:g} v0={v0:g}: geometry_invalid"

    v_imp, vn = runout(v0, a, mu, beta_eff)
    base_row.update(
        {
            "beta_eff_deg": math.degrees(beta_eff),
            "vimp_kmh": v_imp * 3.6,
            "vn_ms": vn,
        }
    )
    if vn <= 0.0:
        base_row["status"] = "runout_stops_before_wall"
        return base_row, f"[{case_no}/{total_cases}] {vehicle} {mode} a={a:g} v0={v0:g}: runout stops"

    result, _df = run_engine_row(base_cfg=cfg, coupling_indices_1based=cidx, vn_ms=vn)
    base_row.update(result)
    if result["engine_status"] == "ok":
        status = "ok"
        if bool(result.get("picard_exhausted_gt_10pct", False)):
            status = "picard_warning"
        base_row["status"] = status
    else:
        base_row["status"] = "engine_failed"
    log = (
        f"[{case_no}/{total_cases}] {vehicle} {mode} a={a:g} v0={v0:g} "
        f"beta={math.degrees(beta_eff):.3f} vn={vn:.3f} m/s -> "
        f"{base_row['status']}: Fpeak={base_row['Fpeak_MN']:.3f} MN, "
        f"Feq={base_row['Feq_100ms_MN']:.3f} MN, "
        f"Picard frac={base_row.get('picard_nonconverged_fraction', np.nan):.4f}"
    )
    return base_row, log

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "stempi_full_train")
    parser.add_argument("--workers", type=int, default=1, help="Parallel engine workers for the scenario grid.")
    args = parser.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    builders = {
        ("TRAXX BR187", "Lok_solo"): (TRAXX_GEOM, build_traxx_lok_solo),
        ("TRAXX BR187", "Zug_full"): (TRAXX_GEOM, build_traxx_full_train),
        ("ICE4 Endwagen proxy", "Lok_solo"): (ICE4_GEOM, build_ice4_lok_solo),
        ("ICE4 Endwagen proxy", "Zug_full"): (ICE4_GEOM, build_ice4_full_train),
    }

    speeds = [80.0, 120.0, 160.0]
    distances = [3.0, 5.0, 10.0]
    mu = 0.30

    configs: dict[tuple[str, str], dict[str, Any]] = {}
    couplings: dict[tuple[str, str], list[int]] = {}
    config_meta: dict[str, Any] = {}
    for key, (_geom, builder) in builders.items():
        cfg, cidx, meta = builder()
        configs[key] = cfg
        couplings[key] = cidx
        config_meta[f"{key[0]} / {key[1]}"] = meta

    write_generated_configs(
        outdir,
        {f"{vehicle}_{mode}": cfg for (vehicle, mode), cfg in configs.items()},
    )

    cases: list[dict[str, Any]] = []
    total_cases = len(builders) * len(distances) * len(speeds)
    case_no = 0
    for (vehicle, mode), (geom, _builder) in builders.items():
        cfg = configs[(vehicle, mode)]
        cidx = couplings[(vehicle, mode)]
        for a in distances:
            for v0 in speeds:
                case_no += 1
                cases.append(
                    {
                        "case_no": case_no,
                        "total_cases": total_cases,
                        "vehicle": vehicle,
                        "mode": mode,
                        "geom": geom,
                        "cfg": cfg,
                        "coupling_indices": cidx,
                        "a_m": a,
                        "v0_kmh": v0,
                        "mu": mu,
                    }
                )

    rows: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for payload in cases:
            row, log = evaluate_case_payload(payload)
            rows.append(row)
            print(log, flush=True)
    else:
        print(f"Running {len(cases)} cases with {workers} worker processes...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for row, log in executor.map(evaluate_case_payload, cases):
                rows.append(row)
                print(log, flush=True)

    full_df = pd.DataFrame(rows)
    full_df = add_ratios_and_status(full_df)

    # Stable output ordering.
    full_df = full_df.sort_values(["Fahrzeug", "a_m", "v0_kmh", "mode"]).reset_index(drop=True)
    grid_df = make_response_grid(full_df)

    full_df.to_csv(outdir / "stempi_full_train_comparison_full.csv", index=False)
    grid_df.to_csv(outdir / "response_grid_lok_vs_zug.csv", index=False)

    known_check = known_lok_solo_check(full_df)
    mono_violations = monotonicity_check(full_df)
    hooke_check = hooke_single_mass_sanity(outdir)

    metadata = {
        "scope": "reduced multibody proxy / parametric full consist model",
        "not_calibrated_to_crash_tests": True,
        "mu_runout": mu,
        "speeds_kmh": speeds,
        "distances_m": distances,
        "engine_settings": {
            "solver": "picard",
            "h_init": 5.0e-5,
            "T_max": 0.15,
            "picard_max_iters": 30,
            "picard_tol": 1.0e-5,
            "contact_model": "lankarani-nikravesh",
            "emit_peak_diagnostics": False,
        },
        "geometry": {
            "TRAXX BR187": TRAXX_GEOM.__dict__ | {"l_eff_m": TRAXX_GEOM.l_eff_m},
            "ICE4 Endwagen proxy": ICE4_GEOM.__dict__ | {"l_eff_m": ICE4_GEOM.l_eff_m},
            "zug_cap_applied_to_both_modes_for_side_by_side_comparability": True,
        },
        "configurations": config_meta,
        "known_lok_solo_check": known_check,
        "monotonicity_violations": mono_violations,
        "hooke_single_mass_sanity": hooke_check,
    }
    (outdir / "stempi_full_train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    make_plots(outdir, grid_df)
    write_markdown_outputs(
        outdir=outdir,
        full_df=full_df,
        grid_df=grid_df,
        metadata=metadata,
        known_check=known_check,
        mono_violations=mono_violations,
        hooke_check=hooke_check,
    )

    print(f"Wrote: {outdir / 'stempi_full_train_comparison_full.csv'}")
    print(f"Wrote: {outdir / 'stempi_full_train_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
