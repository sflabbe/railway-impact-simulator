from __future__ import annotations

import math

import numpy as np

G = 9.80665  # m/s^2


def _finite_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _positive_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _beta_d_rad(beta_d_deg: float) -> float:
    beta = _finite_float("beta_d_deg", beta_d_deg)
    if beta <= 0.0 or beta > 90.0:
        raise ValueError("beta_d_deg must be in (0, 90]")
    return math.radians(beta)


def _pow_positive(base: float, exponent: float) -> float:
    try:
        return math.exp(exponent * math.log(base))
    except OverflowError:
        return math.inf


def kmh_to_ms(v_kmh: float) -> float:
    """Convert km/h to m/s. v_kmh must be >= 0."""
    return float(_nonnegative_float("v_kmh", v_kmh) / 3.6)


def ms_to_kmh(v_ms: float) -> float:
    """Convert m/s to km/h. v_ms must be >= 0."""
    return float(_nonnegative_float("v_ms", v_ms) * 3.6)


def max_lateral_reach_m(v0_kmh: float, mu: float, beta_d_deg: float) -> float:
    """
    Maximum lateral runout distance [m] under constant friction.

        a_max = v0^2 * sin(beta_d) / (2 * mu * g)

    Inputs:
        v0_kmh >= 0
        mu > 0
        beta_d_deg in (0, 90]

    Returns 0.0 if v0_kmh == 0.
    """
    v0_ms = kmh_to_ms(v0_kmh)
    friction = _positive_float("mu", mu)
    beta_rad = _beta_d_rad(beta_d_deg)
    if v0_ms == 0.0:
        return 0.0
    return float(v0_ms**2 * math.sin(beta_rad) / (2.0 * friction * G))


def runout_path_length_m(a_m: float, beta_d_deg: float) -> float:
    """
    Runout path length:

        s = a / sin(beta_d)

    Inputs:
        a_m >= 0
        beta_d_deg in (0, 90]
    """
    a = _nonnegative_float("a_m", a_m)
    beta_rad = _beta_d_rad(beta_d_deg)
    return float(a / math.sin(beta_rad))


def impact_velocity_ms(v0_kmh: float, a_m: float, mu: float, beta_d_deg: float) -> float:
    """
    Post-runout impact speed [m/s].

        v_imp = sqrt(max(v0^2 - 2 * mu * g * s(a), 0))

    Returns 0.0 if kinetic energy is fully dissipated before reaching `a_m`.
    """
    v0_ms = kmh_to_ms(v0_kmh)
    a = _nonnegative_float("a_m", a_m)
    friction = _positive_float("mu", mu)
    beta_rad = _beta_d_rad(beta_d_deg)
    path_length = a / math.sin(beta_rad)
    remaining_v2 = v0_ms**2 - 2.0 * friction * G * path_length
    return float(math.sqrt(max(remaining_v2, 0.0)))


def normal_velocity_ms(v_imp_ms: float, beta_wall_deg: float) -> float:
    """
    Normal component of impact velocity:

        v_n = v_imp * sin(beta_wall)

    Inputs:
        v_imp_ms >= 0
        beta_wall_deg in [0, 90]
    """
    v_imp = _nonnegative_float("v_imp_ms", v_imp_ms)
    beta = _finite_float("beta_wall_deg", beta_wall_deg)
    if beta < 0.0 or beta > 90.0:
        raise ValueError("beta_wall_deg must be in [0, 90]")
    return float(v_imp * math.sin(math.radians(beta)))


def bounded_weibull_reach_probability(
    a_m: float,
    v0_kmh: float,
    mu: float,
    beta_d_deg: float,
    chi: float,
    shape_k: float,
) -> float:
    """
    Truncated Weibull survival reach probability P(reach a).

        ell = chi * a_max

        P = [exp(-(a/ell)^k) - exp(-(a_max/ell)^k)]
            / [1 - exp(-(a_max/ell)^k)]

    Validation order:
        1. Validate all parameters first.
        2. Then return 1.0 if a_m == 0 and parameters are valid.

    Inputs:
        a_m >= 0
        v0_kmh >= 0
        mu > 0
        beta_d_deg in (0, 90]
        chi in (0, 1]
        shape_k > 0

    Returns:
        0.0 if a_m >= a_max or a_max == 0
        1.0 if a_m == 0 and all parameters are valid
        otherwise a finite float clipped/guarded into [0, 1]

    Numerical rule:
        Guard division by zero. Never return NaN. Use expm1 or stable algebra
        when exp(-x) is close to 1.
    """
    a = _nonnegative_float("a_m", a_m)
    v0 = _nonnegative_float("v0_kmh", v0_kmh)
    friction = _positive_float("mu", mu)
    beta_rad = _beta_d_rad(beta_d_deg)
    chi_value = _positive_float("chi", chi)
    if chi_value > 1.0:
        raise ValueError("chi must be in (0, 1]")
    k = _positive_float("shape_k", shape_k)

    if a == 0.0:
        return 1.0

    v0_ms = v0 / 3.6
    a_max = 0.0
    if v0_ms > 0.0:
        a_max = v0_ms**2 * math.sin(beta_rad) / (2.0 * friction * G)

    if a_max == 0.0 or a >= a_max:
        return 0.0

    ell = chi_value * a_max
    if ell <= 0.0:
        return 0.0

    x_a = _pow_positive(a / ell, k)
    x_max = _pow_positive(a_max / ell, k)

    if x_a == math.inf:
        return 0.0

    denominator = -math.expm1(-x_max)
    if denominator <= 0.0 or not math.isfinite(denominator):
        denominator = 1.0

    diff = x_max - x_a
    if diff == math.inf:
        tail_factor = 1.0
    else:
        tail_factor = -math.expm1(-diff)
    numerator = math.exp(-x_a) * tail_factor
    probability = numerator / denominator

    if not math.isfinite(probability):
        probability = 0.0
    return float(np.clip(probability, 0.0, 1.0))
