from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _finite_1d_array(name: str, value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if result.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"all {name} values must be finite")
    return result


def _finite_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def termination_ratio(force_history: np.ndarray) -> float:
    """
    Compute abs(F(t_end)) / max(abs(F)).

    Quality criterion:
        ratio < 0.01 means the contact force has essentially terminated.

    Behavior:
        - force_history must be finite, 1D, and non-empty.
        - Return 0.0 if max(abs(force_history)) == 0.
        - Return native Python float.
    """
    force = _finite_1d_array("force_history", force_history)
    force_abs = np.abs(force)
    max_force = float(np.max(force_abs))
    if max_force == 0.0:
        return 0.0
    return float(force_abs[-1] / max_force)


def force_history_is_terminated(
    force_history: np.ndarray,
    threshold: float = 0.01,
) -> bool:
    """
    Return True if termination_ratio(force_history) < threshold.

    threshold must be > 0.
    Return native Python bool.
    """
    threshold_value = _finite_float("threshold", threshold)
    if threshold_value <= 0.0:
        raise ValueError("threshold must be > 0")
    return bool(termination_ratio(force_history) < threshold_value)


def equivalent_static_force_sdof(
    time_s: np.ndarray,
    force: np.ndarray,
    Tn_s: float = 0.100,
    zeta: float = 0.05,
    oscillator_mass: float = 1.0,
) -> float:
    """
    Response-spectrum equivalent static force Feq.

    Equation of motion:
        u'' + 2*zeta*omega_n*u' + omega_n**2*u = force(t) / oscillator_mass

    Return:
        Feq = oscillator_mass * omega_n**2 * max(abs(u(t)))

    Unit rule:
        Feq is returned in the same force unit as `force`.
        For physical SI: force in N and oscillator_mass in kg.
        For normalized work: force in MN and oscillator_mass=1.0 gives Feq in MN.
        The linear scaling makes Feq independent of the numerical value of
        oscillator_mass if force and oscillator_mass are used consistently.

    Inputs:
        time_s: finite 1D array, length >= 2, strictly increasing.
        force: finite 1D array, same length as time_s.
        Tn_s > 0
        0 <= zeta < 1
        oscillator_mass > 0

    Implementation:
        Use Newmark average acceleration, beta=0.25, gamma=0.5.
        Initial conditions: u(0)=0, v(0)=0.
        For each time step h = time[i+1] - time[i], allow non-uniform h.
        External load per unit mass is p = force / oscillator_mass.
        Compute acceleration from equilibrium.
        Track max(abs(u)) including all time steps.

    Return native Python float.
    """
    time = _finite_1d_array("time_s", time_s)
    force_values = _finite_1d_array("force", force)
    if time.shape != force_values.shape:
        raise ValueError("time_s and force must have the same length")
    if time.size < 2:
        raise ValueError("time_s and force must have length >= 2")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be strictly increasing")

    Tn = _finite_float("Tn_s", Tn_s)
    if Tn <= 0.0:
        raise ValueError("Tn_s must be > 0")
    damping_ratio = _finite_float("zeta", zeta)
    if damping_ratio < 0.0 or damping_ratio >= 1.0:
        raise ValueError("zeta must be in [0, 1)")
    mass = _finite_float("oscillator_mass", oscillator_mass)
    if mass <= 0.0:
        raise ValueError("oscillator_mass must be > 0")

    omega_n = 2.0 * math.pi / Tn
    stiffness = omega_n**2
    damping = 2.0 * damping_ratio * omega_n
    load = force_values / mass

    beta = 0.25
    gamma = 0.5

    displacement = 0.0
    velocity = 0.0
    acceleration = float(load[0] - damping * velocity - stiffness * displacement)
    max_abs_displacement = abs(displacement)

    for i in range(time.size - 1):
        h = float(time[i + 1] - time[i])

        a0 = 1.0 / (beta * h**2)
        a1 = gamma / (beta * h)
        a2 = 1.0 / (beta * h)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = h * (gamma / (2.0 * beta) - 1.0)

        effective_stiffness = stiffness + a0 + damping * a1
        effective_load = (
            load[i + 1]
            + a0 * displacement
            + a2 * velocity
            + a3 * acceleration
            + damping * (a1 * displacement + a4 * velocity + a5 * acceleration)
        )

        next_displacement = effective_load / effective_stiffness
        next_acceleration = (
            a0 * (next_displacement - displacement) - a2 * velocity - a3 * acceleration
        )
        next_velocity = velocity + h * (
            (1.0 - gamma) * acceleration + gamma * next_acceleration
        )

        displacement = float(next_displacement)
        velocity = float(next_velocity)
        acceleration = float(next_acceleration)
        max_abs_displacement = max(max_abs_displacement, abs(displacement))

    return float(mass * stiffness * max_abs_displacement)


def compute_response_spectrum(
    time_s: np.ndarray,
    force: np.ndarray,
    Tn_grid_ms: np.ndarray | None = None,
    zeta: float = 0.05,
    oscillator_mass: float = 1.0,
) -> pd.DataFrame:
    """
    Compute Feq(Tn) over a natural-period grid.

    Parameters:
        Tn_grid_ms:
            If None, use 30 log-spaced values from 10 to 3000 ms.
            Otherwise must be finite, positive, 1D.

    Return:
        pd.DataFrame with columns exactly ["Tn_ms", "Feq"].
        Feq is in the same force unit as input `force`.
    """
    if Tn_grid_ms is None:
        periods_ms = np.logspace(math.log10(10.0), math.log10(3000.0), 30)
    else:
        periods_ms = _finite_1d_array("Tn_grid_ms", Tn_grid_ms)
        if np.any(periods_ms <= 0.0):
            raise ValueError("all Tn_grid_ms values must be > 0")

    feq = [
        equivalent_static_force_sdof(
            time_s,
            force,
            Tn_s=float(period_ms) / 1000.0,
            zeta=zeta,
            oscillator_mass=oscillator_mass,
        )
        for period_ms in periods_ms
    ]
    return pd.DataFrame({"Tn_ms": periods_ms.astype(float), "Feq": feq})
