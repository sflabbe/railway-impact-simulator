from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_TABLE2: dict[int, tuple[float, float, float]] = {
    30: (0.733, 0.982, 0.964),
    100: (0.933, 1.007, 0.997),
    300: (0.918, 0.958, 0.983),
}


@dataclass(frozen=True)
class PowerLawCoeffs:
    Tn_ms: float
    A: float
    p: float
    R2_loocv: float


def _finite_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def get_default_coeffs(Tn_ms: int) -> PowerLawCoeffs:
    """Return Table 2 coefficients for Tn in {30, 100, 300} ms. Raise KeyError otherwise."""
    A, p, r2 = _TABLE2[Tn_ms]
    return PowerLawCoeffs(Tn_ms=float(Tn_ms), A=A, p=p, R2_loocv=r2)


def power_law_feq(vn_kmh: float | np.ndarray, A: float, p: float) -> float | np.ndarray:
    """
    Evaluate Feq = A * vn**p.

    Inputs:
        all vn_kmh >= 0
        A finite
        p finite

    Return 0.0 where vn_kmh == 0.
    Scalar input returns native Python float.
    Array input returns np.ndarray.
    """
    A_value = _finite_float("A", A)
    p_value = _finite_float("p", p)

    values = np.asarray(vn_kmh, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("all vn_kmh values must be finite")
    if np.any(values < 0.0):
        raise ValueError("all vn_kmh values must be >= 0")

    if np.ndim(vn_kmh) == 0:
        value = float(values)
        return 0.0 if value == 0.0 else float(A_value * value**p_value)

    result = np.zeros_like(values, dtype=float)
    positive = values > 0.0
    result[positive] = A_value * np.power(values[positive], p_value)
    return result


def fit_power_law(
    vn_kmh: np.ndarray,
    feq_mn: np.ndarray,
) -> tuple[float, float]:
    """
    Fit Feq = A * vn**p by OLS in log-log space.

    Required behavior:
        - Convert inputs to 1D arrays.
        - Require same shape.
        - Filter finite positive pairs only.
        - Raise ValueError if fewer than 2 valid pairs remain.
        - Return finite (A, p).
    """
    vn = np.asarray(vn_kmh, dtype=float)
    feq = np.asarray(feq_mn, dtype=float)
    if vn.ndim != 1 or feq.ndim != 1:
        raise ValueError("vn_kmh and feq_mn must be 1D arrays")
    if vn.shape != feq.shape:
        raise ValueError("vn_kmh and feq_mn must have the same shape")

    valid = np.isfinite(vn) & np.isfinite(feq) & (vn > 0.0) & (feq > 0.0)
    if int(np.count_nonzero(valid)) < 2:
        raise ValueError("at least two finite positive pairs are required")

    p, log_a = np.polyfit(np.log(vn[valid]), np.log(feq[valid]), 1)
    A = float(math.exp(float(log_a)))
    p = float(p)
    if not math.isfinite(A) or not math.isfinite(p):
        raise ValueError("fitted coefficients must be finite")
    return A, p
