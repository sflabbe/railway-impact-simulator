from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .sdof import equivalent_static_force_sdof


def _positive_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def build_one_mass_hooke_engine_params(
    mass_kg: float,
    k_wall_N_m: float,
    v_n_ms: float,
    dt_s: float = 1e-4,
    t_max_s: float = 0.10,
) -> dict:
    """
    Build run_simulation() params for the canonical one-mass Hooke oracle case.

    Analytical oracle:
        omega = sqrt(k_wall / mass)
        u_peak = v_n / omega
        F_peak = v_n * sqrt(k_wall * mass)

    Validation:
        mass_kg > 0
        k_wall_N_m > 0
        v_n_ms > 0
        dt_s > 0
        t_max_s > 0

    Important:
        Do not require t_max_s >= contact duration. The truncation test
        deliberately uses t_max_s < t_contact.

    For a complete contact pulse, t_max_s should be at least pi / omega.

    Return a dict ready for railway_simulator.core.engine.run_simulation().
    """
    mass = _positive_float("mass_kg", mass_kg)
    k_wall = _positive_float("k_wall_N_m", k_wall_N_m)
    velocity = _positive_float("v_n_ms", v_n_ms)
    dt = _positive_float("dt_s", dt_s)
    t_max = _positive_float("t_max_s", t_max_s)

    step = int(np.ceil(t_max / dt))

    return {
        # Train geometry
        "n_masses": 1,
        "masses": [mass],
        "x_init": [0.0],
        "y_init": [0.0],
        # Impact conditions
        "v0_init": -velocity,
        "d0": 0.0,
        "angle_rad": 0.0,
        # No inter-mass springs
        "k_train": [],
        "fy": [],
        "uy": [],
        # Wall contact
        "contact_model": "hooke",
        "k_wall": k_wall,
        "cr_wall": 0.0,
        # Friction disabled
        "friction_model": "none",
        "mu_s": 0.0,
        "mu_k": 0.0,
        "sigma_0": 0.0,
        "sigma_1": 0.0,
        "sigma_2": 0.0,
        # Integration
        "alpha_hht": 0.0,
        "solver": "newton",
        "newton_tol": 1e-6,
        "max_iter": 25,
        # Time control
        "h_init": dt,
        "T_max": t_max,
        "step": step,
        "T_int": (0.0, t_max),
        "case_name": "hooke_oracle_test",
    }


def extract_engine_force_history(
    df: pd.DataFrame,
    force_col: str = "Impact_Force_MN",
    time_col: str = "Time_s",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract finite numpy arrays (time_s, force) from run_simulation() output.

    Validation:
        df is non-empty
        required columns exist
        arrays are finite
        time is strictly increasing

    Return:
        time_s: float ndarray
        force: float ndarray, in the units of force_col. Default is MN.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df must be non-empty")

    missing = [col for col in (time_col, force_col) if col not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")

    try:
        time_s = df[time_col].to_numpy(dtype=float)
        force = df[force_col].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("time and force columns must be numeric") from exc

    if time_s.ndim != 1 or force.ndim != 1:
        raise ValueError("time and force columns must be 1D")
    if time_s.size == 0 or force.size == 0:
        raise ValueError("time and force arrays must be non-empty")
    if time_s.shape != force.shape:
        raise ValueError("time and force arrays must have the same length")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("all time values must be finite")
    if not np.all(np.isfinite(force)):
        raise ValueError("all force values must be finite")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time must be strictly increasing")

    return time_s, force


def equivalent_static_force_from_engine_df(
    df: pd.DataFrame,
    Tn_s: float = 0.100,
    zeta: float = 0.05,
    force_col: str = "Impact_Force_MN",
    time_col: str = "Time_s",
) -> float:
    """
    Compute Feq from engine DataFrame.

    Pipeline:
        extract_engine_force_history -> equivalent_static_force_sdof

    With the default force column, return Feq in MN.
    """
    time_s, force = extract_engine_force_history(
        df,
        force_col=force_col,
        time_col=time_col,
    )
    return equivalent_static_force_sdof(time_s, force, Tn_s=Tn_s, zeta=zeta)
