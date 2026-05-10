from __future__ import annotations

import math

import numpy as np


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


def _probability(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _nonnegative_array(name: str, value: np.ndarray | float) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"all {name} values must be finite")
    if np.any(result < 0.0):
        raise ValueError(f"all {name} values must be >= 0")
    return result


def base_occurrence_rate_per_year(
    n_trains_year: float,
    exposure_km: float,
    derailment_rate_per_train_km: float,
    derailment_rate_factor: float = 1.0,
) -> float:
    """
    Base annual scenario rate before reach and angle probabilities.

        lambda_0 = N_trains * L_exp * r_derail * factor

    Reference:
        100 * 0.05 * 0.12e-6 = 6.0e-7 [1/year]

    Inputs must be >= 0.
    """
    n_trains = _nonnegative_float("n_trains_year", n_trains_year)
    exposure = _nonnegative_float("exposure_km", exposure_km)
    rate = _nonnegative_float("derailment_rate_per_train_km", derailment_rate_per_train_km)
    factor = _nonnegative_float("derailment_rate_factor", derailment_rate_factor)
    return float(n_trains * exposure * rate * factor)


def scenario_occurrence_rate(
    lambda0_per_year: float,
    p_reach: float,
    p_wall: float,
) -> float:
    """
    Annual scenario occurrence rate:

        lambda_s = lambda_0 * P(reach a) * P(beta_wall)

    Inputs:
        lambda0_per_year >= 0
        p_reach in [0, 1]
        p_wall in [0, 1]
    """
    lambda0 = _nonnegative_float("lambda0_per_year", lambda0_per_year)
    reach = _probability("p_reach", p_reach)
    wall = _probability("p_wall", p_wall)
    return float(lambda0 * reach * wall)


def lambda_en_crit(
    lambda_s_per_year: float,
    feq: float,
    fen: float,
    alpha: float,
) -> float:
    """
    Required normative anchor frequency:

        lambda_EN_crit = lambda_s * (Feq / FEN)**alpha

    Inputs:
        lambda_s_per_year >= 0
        feq >= 0
        fen > 0
        alpha >= 0

    Return 0.0 if lambda_s_per_year == 0 or feq == 0.
    """
    lambda_s = _nonnegative_float("lambda_s_per_year", lambda_s_per_year)
    feq_value = _nonnegative_float("feq", feq)
    fen_value = _finite_float("fen", fen)
    if fen_value <= 0.0:
        raise ValueError("fen must be > 0")
    alpha_value = _nonnegative_float("alpha", alpha)
    if lambda_s == 0.0 or feq_value == 0.0:
        return 0.0
    return float(lambda_s * (feq_value / fen_value) ** alpha_value)


def iso_demand_mask(
    feq: np.ndarray | float,
    fen: float,
    delta: float = 0.10,
) -> np.ndarray | bool:
    """
    True where abs(Feq/FEN - 1) <= delta.

    Inputs:
        fen > 0
        delta >= 0
        all feq >= 0

    Scalar input must return native Python bool, not numpy.bool_.
    Array input must return np.ndarray with dtype bool.
    """
    feq_values = _nonnegative_array("feq", feq)
    fen_value = _finite_float("fen", fen)
    if fen_value <= 0.0:
        raise ValueError("fen must be > 0")
    delta_value = _nonnegative_float("delta", delta)
    result = np.abs(feq_values / fen_value - 1.0) <= delta_value
    return bool(result) if np.ndim(feq) == 0 else result


def exceedance_mask(
    feq: np.ndarray | float,
    fen: float,
) -> np.ndarray | bool:
    """
    True where Feq > FEN, strict inequality.

    Scalar input must return native Python bool, not numpy.bool_.
    Array input must return np.ndarray with dtype bool.
    """
    feq_values = _nonnegative_array("feq", feq)
    fen_value = _finite_float("fen", fen)
    if fen_value <= 0.0:
        raise ValueError("fen must be > 0")
    result = feq_values > fen_value
    return bool(result) if np.ndim(feq) == 0 else result
