from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pytest

from railway_simulator.core.engine import run_simulation
from railway_simulator.hazard import (
    SDOFSettings,
    build_lateral_yaw_params,
    build_single_mass_reference_params,
    default_velocity_families,
    equivalent_demands_from_engine_df,
    grid_summary,
    lateral_initial_coordinates,
    lateral_t_max_s,
    scenario_exceedance_from_vstar,
    scenario_grid,
    scenario_weight_grid,
    single_mass_reference_feq_MN,
    wall_reaction_histories_N,
)


def quiet_run(params: dict):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        return run_simulation(params, emit_peak_diagnostics=False)


def small_base_config() -> dict:
    return {
        "n_masses": 3,
        "masses": [100.0, 100.0, 100.0],
        "x_init": [0.0, 1.0, 2.0],
        "y_init": [0.0, 0.0, 0.0],
        "v0_init": -2.0,
        "angle_rad": 0.0,
        "d0": 0.005,
        "fy": [0.0, 0.0],
        "uy": [1.0, 1.0],
        "k_train": [0.0, 0.0],
        "contact_model": "hooke",
        "k_wall": 1.0e5,
        "cr_wall": 0.0,
        "mu_s": 0.0,
        "mu_k": 0.0,
        "sigma_0": 0.0,
        "sigma_1": 0.0,
        "sigma_2": 0.0,
        "friction_model": "none",
        "bw_a": 1.0,
        "bw_A": 1.0,
        "bw_beta": 0.5,
        "bw_gamma": 0.5,
        "bw_n": 2,
        "alpha_hht": 0.0,
        "newton_tol": 1.0e-6,
        "max_iter": 25,
        "solver": "newton",
        "h_init": 5.0e-4,
        "T_max": 0.2,
        "building_enable": False,
        "building_mass": 1.0,
        "building_zeta": 0.05,
        "building_height": 1.0,
        "building_model": "linear",
        "building_uy": 0.05,
        "building_uy_mm": 50.0,
        "building_alpha": 0.0,
        "building_gamma": 0.0,
    }


def test_scenario_grid_has_90_entries_and_weight_grid_has_30() -> None:
    weights = scenario_weight_grid()
    assert len(weights) == 30
    for w in weights:
        assert w.w_guided + w.w_excursion + w.w_severe == pytest.approx(1.0)
    assert len(default_velocity_families()) == 3
    assert len(scenario_grid()) == 90


def test_scenario_exceedance_analytical_truncates_and_renormalizes() -> None:
    weights = scenario_weight_grid(w_s_values=(0.25,), rho_values=(0.5,))[0]
    family = default_velocity_families()[1]  # V_N: severe starts at 6 m/s
    # At 10 km/h, v_imp=2.777... m/s, severe infeasible and excursion is truncated.
    result = scenario_exceedance_from_vstar(
        v_imp_ms=10.0 / 3.6,
        v_star_ms=1.7,
        weights=weights,
        family=family,
    )
    assert "severe" not in result.active_regimes
    assert result.feasible_weights["guided"] + result.feasible_weights["excursion"] == pytest.approx(1.0)
    assert 0.0 <= result.pi <= 1.0
    assert result.truncated_intervals["excursion"][1] == pytest.approx(10.0 / 3.6)


def test_grid_summary_reports_grid_dependent_extremes() -> None:
    summary = grid_summary([0.0, 0.25, 0.5, 1.0])
    assert summary["pi_min_grid"] == pytest.approx(0.0)
    assert summary["pi_max_grid"] == pytest.approx(1.0)
    assert summary["n_zero"] == 1
    assert summary["n_active"] == 3


def test_lateral_initial_coordinates_psi_zero_are_simultaneous() -> None:
    x, y = lateral_initial_coordinates(small_base_config(), psi_rad=0.0)
    assert np.allclose(x, 0.0)
    assert np.allclose(y, [0.0, 1.0, 2.0])


def test_lateral_initial_coordinates_yaw_offsets_closest_mass_to_zero() -> None:
    psi = math.radians(10.0)
    x, y = lateral_initial_coordinates(small_base_config(), psi_rad=psi)
    assert float(np.min(x)) == pytest.approx(0.0)
    assert float(np.max(x)) > 0.0
    assert y.shape == x.shape


def test_lateral_t_max_uses_last_contact_plus_tail() -> None:
    x = np.array([0.0, 0.5, 1.0])
    tmax = lateral_t_max_s(x, d0_m=0.01, v_n_ms=2.0, Tn_s=0.1, tail_min_s=0.2)
    assert tmax == pytest.approx((1.0 + 0.01) / 2.0 + 0.3)


@pytest.mark.slow
def test_single_mass_reference_and_lateral_psi_zero_integrity() -> None:
    base = small_base_config()
    vn = 2.0
    sdof = SDOFSettings(Tn_s=0.05, zeta=0.02)

    single_params = build_single_mass_reference_params(
        base,
        v_n_ms=vn,
        mass_kg=100.0,
        t_max_s=0.20,
        h_init_s=base["h_init"],
    )
    single_df = quiet_run(single_params)
    single_feq = single_mass_reference_feq_MN(single_df, sdof=sdof)
    assert single_feq > 0.0

    lateral_params = build_lateral_yaw_params(
        base,
        v_n_ms=vn,
        psi_rad=0.0,
        Tn_s=sdof.Tn_s,
        tail_min_s=0.10,
        h_init_s=base["h_init"],
    )
    lateral_df = quiet_run(lateral_params)
    time_s, reactions = wall_reaction_histories_N(lateral_df)
    assert reactions.shape[0] == 3
    # Simultaneous contact: per-mass force histories should match closely.
    assert np.max(np.abs(reactions[0] - reactions[1])) < 1.0e-6 * max(1.0, np.max(reactions[0]))
    assert np.max(np.abs(reactions[1] - reactions[2])) < 1.0e-6 * max(1.0, np.max(reactions[1]))

    result = equivalent_demands_from_engine_df(
        lateral_df,
        sdof=sdof,
        reference_single_MN=single_feq,
        contact_threshold_N=1.0,
    )
    assert result.local_feq_MN == pytest.approx(single_feq, rel=0.03)
    assert result.total_feq_MN == pytest.approx(3.0 * single_feq, rel=0.03)
    assert result.eta is not None
    assert max(abs(e - 1.0) for e in result.eta) < 0.03
    assert result.n_contacts == (1, 1, 1)
    assert result.first_contact_vx_m_s is not None
    assert all(v is not None for v in result.first_contact_vx_m_s)
