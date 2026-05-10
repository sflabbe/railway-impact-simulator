from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from railway_simulator.hazard import (
    PowerLawCoeffs,
    fit_power_law,
    get_default_coeffs,
    power_law_feq,
)


@pytest.mark.parametrize(
    ("Tn_ms", "A", "p", "R2_loocv"),
    [
        (30, 0.733, 0.982, 0.964),
        (100, 0.933, 1.007, 0.997),
        (300, 0.918, 0.958, 0.983),
    ],
)
def test_table2_default_coefficients(Tn_ms: int, A: float, p: float, R2_loocv: float) -> None:
    coeffs = get_default_coeffs(Tn_ms)
    assert coeffs == PowerLawCoeffs(Tn_ms=float(Tn_ms), A=A, p=p, R2_loocv=R2_loocv)


def test_unknown_period_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_default_coeffs(50)


def test_power_law_coeffs_is_frozen() -> None:
    coeffs = get_default_coeffs(30)
    with pytest.raises(FrozenInstanceError):
        coeffs.A = 1.0


def test_power_law_feq_zero_array_scaling_and_negative_velocity() -> None:
    assert power_law_feq(0.0, A=2.0, p=3.0) == 0.0
    assert power_law_feq(3.0, A=2.0, p=3.0) == pytest.approx(54.0)

    velocities = np.array([0.0, 2.0, 4.0])
    np.testing.assert_allclose(power_law_feq(velocities, A=1.5, p=2.0), [0.0, 6.0, 24.0])

    with pytest.raises(ValueError):
        power_law_feq(-1.0, A=1.0, p=1.0)


def test_fit_power_law_recovers_exact_synthetic_coefficients() -> None:
    A = 0.733
    p = 0.982
    vn = np.array([10.0, 20.0, 40.0, 80.0])
    feq = A * vn**p

    fitted_A, fitted_p = fit_power_law(vn, feq)
    assert fitted_A == pytest.approx(A)
    assert fitted_p == pytest.approx(p)


def test_fit_power_law_ignores_zero_nan_and_inf_pairs() -> None:
    A = 1.25
    p = 1.1
    valid_vn = np.array([15.0, 30.0, 60.0])
    valid_feq = A * valid_vn**p
    vn = np.array([0.0, np.nan, np.inf, *valid_vn])
    feq = np.array([10.0, 20.0, 30.0, *valid_feq])

    fitted_A, fitted_p = fit_power_law(vn, feq)
    assert fitted_A == pytest.approx(A)
    assert fitted_p == pytest.approx(p)


def test_fit_power_law_rejects_shape_mismatch_non_1d_and_too_few_valid_pairs() -> None:
    with pytest.raises(ValueError):
        fit_power_law(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError):
        fit_power_law(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError):
        fit_power_law(np.array([0.0, np.nan, 2.0]), np.array([1.0, 2.0, 4.0]))
