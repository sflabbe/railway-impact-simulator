from __future__ import annotations

import math

import numpy as np


def _finite_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def half_sine_pulse(
    Tp: float,
    F_peak: float,
    dt: float = 1e-4,
    n_periods: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    F(t) = F_peak * sin(pi*t/Tp) for 0 <= t <= Tp, else 0.

    Validation:
        Tp > 0
        F_peak finite
        dt > 0
        n_periods > 1
    """
    pulse_duration = _finite_float("Tp", Tp)
    if pulse_duration <= 0.0:
        raise ValueError("Tp must be > 0")
    peak_force = _finite_float("F_peak", F_peak)
    time_step = _finite_float("dt", dt)
    if time_step <= 0.0:
        raise ValueError("dt must be > 0")
    duration_factor = _finite_float("n_periods", n_periods)
    if duration_factor <= 1.0:
        raise ValueError("n_periods must be > 1")

    end_time = duration_factor * pulse_duration
    time_s = np.arange(0.0, end_time + 0.5 * time_step, time_step, dtype=float)
    force = np.zeros_like(time_s)
    active = time_s <= pulse_duration
    force[active] = peak_force * np.sin(math.pi * time_s[active] / pulse_duration)
    return time_s, force


def one_mass_hooke_oracle(
    mass_kg: float,
    k_wall_N_m: float,
    v_n_ms: float,
) -> dict:
    """
    Exact one-mass Hooke wall-contact oracle.

    omega = sqrt(k_wall / mass)
    u_peak = v_n / omega
    F_peak = v_n * sqrt(k_wall * mass)
    t_contact = pi / omega

    Inputs:
        mass_kg > 0
        k_wall_N_m > 0
        v_n_ms > 0

    Returns keys:
        omega_rad_s
        u_peak_m
        F_peak_N
        t_contact_s
    """
    mass = _finite_float("mass_kg", mass_kg)
    if mass <= 0.0:
        raise ValueError("mass_kg must be > 0")
    stiffness = _finite_float("k_wall_N_m", k_wall_N_m)
    if stiffness <= 0.0:
        raise ValueError("k_wall_N_m must be > 0")
    velocity = _finite_float("v_n_ms", v_n_ms)
    if velocity <= 0.0:
        raise ValueError("v_n_ms must be > 0")

    omega = math.sqrt(stiffness / mass)
    return {
        "omega_rad_s": omega,
        "u_peak_m": velocity / omega,
        "F_peak_N": velocity * math.sqrt(stiffness * mass),
        "t_contact_s": math.pi / omega,
    }
