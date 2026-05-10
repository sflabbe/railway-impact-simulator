from __future__ import annotations

import math

import numpy as np
import pytest

from helpers.reduced_oracles import half_sine_pulse, one_mass_hooke_oracle
from railway_simulator.hazard import (
    compute_response_spectrum,
    equivalent_static_force_sdof,
    force_history_is_terminated,
    termination_ratio,
)


def test_completed_contact_pulse() -> None:
    _, F = half_sine_pulse(Tp=0.10, F_peak=1.0, dt=1e-4, n_periods=3.0)
    assert termination_ratio(F) < 0.01
    assert force_history_is_terminated(F)


def test_truncated_pulse_fails() -> None:
    _, F = half_sine_pulse(Tp=0.10, F_peak=1.0, dt=1e-4)
    F_trunc = F[: int(0.05 / 1e-4)]
    assert termination_ratio(F_trunc) > 0.01
    assert not force_history_is_terminated(F_trunc)


def test_termination_ratio_uses_absolute_final_force() -> None:
    F = np.array([0.0, 1.0, -0.5])
    assert termination_ratio(F) == pytest.approx(0.5)
    assert not force_history_is_terminated(F, threshold=0.01)


def test_zero_history_is_terminated() -> None:
    F = np.zeros(100)
    assert termination_ratio(F) == 0.0
    assert force_history_is_terminated(F)


@pytest.mark.parametrize(
    "force_history",
    [
        np.array([]),
        np.array([[0.0, 1.0]]),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
    ],
)
def test_termination_ratio_rejects_invalid_histories(force_history: np.ndarray) -> None:
    with pytest.raises(ValueError):
        termination_ratio(force_history)
    with pytest.raises(ValueError):
        force_history_is_terminated(force_history)


@pytest.mark.parametrize("threshold", [0.0, -0.01, math.nan, math.inf])
def test_force_history_is_terminated_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        force_history_is_terminated(np.array([0.0, 0.0]), threshold=threshold)


def test_sdof_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        equivalent_static_force_sdof(np.array([0.0, 0.1]), np.array([0.0]))


def test_sdof_rejects_length_less_than_two() -> None:
    with pytest.raises(ValueError):
        equivalent_static_force_sdof(np.array([0.0]), np.array([0.0]))


@pytest.mark.parametrize(
    "time_s",
    [
        np.array([0.0, 0.1, 0.1]),
        np.array([0.0, 0.2, 0.1]),
    ],
)
def test_sdof_rejects_non_increasing_time(time_s: np.ndarray) -> None:
    with pytest.raises(ValueError):
        equivalent_static_force_sdof(time_s, np.zeros_like(time_s))


@pytest.mark.parametrize("bad_value", [math.nan, math.inf])
def test_sdof_rejects_nan_or_inf_in_time(bad_value: float) -> None:
    with pytest.raises(ValueError):
        equivalent_static_force_sdof(np.array([0.0, bad_value]), np.array([0.0, 0.0]))


@pytest.mark.parametrize("bad_value", [math.nan, math.inf])
def test_sdof_rejects_nan_or_inf_in_force(bad_value: float) -> None:
    with pytest.raises(ValueError):
        equivalent_static_force_sdof(np.array([0.0, 0.1]), np.array([0.0, bad_value]))


@pytest.mark.parametrize("Tn_s", [0.0, -0.1])
def test_sdof_rejects_nonpositive_natural_period(Tn_s: float) -> None:
    with pytest.raises(ValueError):
        equivalent_static_force_sdof(np.array([0.0, 0.1]), np.zeros(2), Tn_s=Tn_s)


@pytest.mark.parametrize("zeta", [-0.01, 1.0, 1.2])
def test_sdof_rejects_invalid_damping_ratio(zeta: float) -> None:
    with pytest.raises(ValueError):
        equivalent_static_force_sdof(np.array([0.0, 0.1]), np.zeros(2), zeta=zeta)


@pytest.mark.parametrize("oscillator_mass", [0.0, -1.0])
def test_sdof_rejects_nonpositive_oscillator_mass(oscillator_mass: float) -> None:
    with pytest.raises(ValueError):
        equivalent_static_force_sdof(
            np.array([0.0, 0.1]),
            np.zeros(2),
            oscillator_mass=oscillator_mass,
        )


def test_unit_scaling_N_to_kN() -> None:
    t, F_N = half_sine_pulse(Tp=0.10, F_peak=1000.0, dt=1e-4)
    F_kN = F_N / 1000.0
    feq_N = equivalent_static_force_sdof(t, F_N, Tn_s=0.10, zeta=0.0)
    feq_kN = equivalent_static_force_sdof(t, F_kN, Tn_s=0.10, zeta=0.0)
    assert feq_kN == pytest.approx(feq_N / 1000.0, rel=1e-6)


def test_mass_parameter_does_not_affect_feq_unit() -> None:
    t, F = half_sine_pulse(Tp=0.10, F_peak=1.0, dt=1e-4)
    feq_m1 = equivalent_static_force_sdof(
        t,
        F,
        Tn_s=0.10,
        zeta=0.0,
        oscillator_mass=1.0,
    )
    feq_m1000 = equivalent_static_force_sdof(
        t,
        F,
        Tn_s=0.10,
        zeta=0.0,
        oscillator_mass=1000.0,
    )
    assert feq_m1 == pytest.approx(feq_m1000, rel=1e-6)


def test_rigid_limit() -> None:
    Tp, F_peak = 0.10, 1.0
    t, F = half_sine_pulse(Tp=Tp, F_peak=F_peak, dt=1e-4, n_periods=3.0)
    feq = equivalent_static_force_sdof(t, F, Tn_s=Tp / 100.0, zeta=0.0)
    assert feq / F_peak == pytest.approx(1.0, rel=0.05)


def test_daf_half_sine_tn_equals_tp_undamped() -> None:
    Tp, F_peak = 0.10, 1.0
    t, F = half_sine_pulse(Tp=Tp, F_peak=F_peak, dt=1e-4, n_periods=3.0)
    feq = equivalent_static_force_sdof(t, F, Tn_s=Tp, zeta=0.0)
    assert feq / F_peak == pytest.approx(1.73, rel=0.05)


def test_zero_force() -> None:
    t = np.linspace(0, 1.0, 10000)
    F = np.zeros_like(t)
    assert equivalent_static_force_sdof(t, F, Tn_s=0.10) == pytest.approx(0.0, abs=1e-14)


def test_amplitude_linearity() -> None:
    t, F1 = half_sine_pulse(Tp=0.10, F_peak=1.0, dt=1e-4)
    _, F2 = half_sine_pulse(Tp=0.10, F_peak=2.0, dt=1e-4)
    feq1 = equivalent_static_force_sdof(t, F1, Tn_s=0.10)
    feq2 = equivalent_static_force_sdof(t, F2, Tn_s=0.10)
    assert feq2 == pytest.approx(2.0 * feq1, rel=1e-6)


def test_damping_reduces_feq() -> None:
    t, F = half_sine_pulse(Tp=0.10, F_peak=1.0, dt=1e-4, n_periods=3.0)
    feq_0 = equivalent_static_force_sdof(t, F, Tn_s=0.10, zeta=0.0)
    feq_05 = equivalent_static_force_sdof(t, F, Tn_s=0.10, zeta=0.05)
    feq_20 = equivalent_static_force_sdof(t, F, Tn_s=0.10, zeta=0.20)
    assert feq_0 >= feq_05 >= feq_20


def test_response_spectrum_default_grid_columns_length_and_nonnegative_feq() -> None:
    t, F = half_sine_pulse(Tp=0.10, F_peak=1.0, dt=1e-4)
    spectrum = compute_response_spectrum(t, F)
    assert list(spectrum.columns) == ["Tn_ms", "Feq"]
    assert len(spectrum) == 30
    assert np.all(spectrum["Feq"].to_numpy() >= 0.0)


def test_response_spectrum_tn_100_ms_equals_scalar_call() -> None:
    t, F = half_sine_pulse(Tp=0.10, F_peak=1.0, dt=1e-4)
    spectrum = compute_response_spectrum(t, F, Tn_grid_ms=np.array([50.0, 100.0, 200.0]))
    scalar = equivalent_static_force_sdof(t, F, Tn_s=0.100)
    assert spectrum.loc[1, "Feq"] == pytest.approx(scalar)


@pytest.mark.parametrize(
    "Tn_grid_ms",
    [
        np.array([]),
        np.array([[100.0]]),
        np.array([100.0, math.nan]),
        np.array([100.0, math.inf]),
        np.array([0.0, 100.0]),
        np.array([-10.0, 100.0]),
    ],
)
def test_response_spectrum_rejects_invalid_tn_grid(Tn_grid_ms: np.ndarray) -> None:
    t, F = half_sine_pulse(Tp=0.10, F_peak=1.0, dt=1e-4)
    with pytest.raises(ValueError):
        compute_response_spectrum(t, F, Tn_grid_ms=Tn_grid_ms)


def test_one_mass_hooke_oracle() -> None:
    oracle = one_mass_hooke_oracle(mass_kg=4.0, k_wall_N_m=100.0, v_n_ms=3.0)
    assert oracle["omega_rad_s"] == pytest.approx(5.0)
    assert oracle["u_peak_m"] == pytest.approx(0.6)
    assert oracle["F_peak_N"] == pytest.approx(60.0)
    assert oracle["t_contact_s"] == pytest.approx(math.pi / 5.0)
