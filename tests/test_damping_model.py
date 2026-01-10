from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from railway_simulator.core.engine import get_default_simulation_params, run_simulation


def _base_params() -> dict:
    params = get_default_simulation_params()
    params.update(
        {
            "n_masses": 2,
            "masses": [1000.0, 1000.0],
            "x_init": [0.0, 1.0],
            "y_init": [0.0, 0.0],
            "fy": [1.0e6],
            "uy": [1.0e-3],
            "mu_s": 0.0,
            "mu_k": 0.0,
            "friction_model": "none",
            "damping_model": "stiffness",
            "damping_zeta": 0.05,
            "alpha_hht": 0.0,
            "newton_tol": 1.0e-5,
            "max_iter": 50,
        }
    )
    return params


def test_stiffness_damping_does_not_brake_rigid_translation() -> None:
    params = _base_params()
    params.update(
        {
            "v0_init": -10.0,
            "d0": 5.0,
            "T_max": 0.1,
            "h_init": 1.0e-3,
            "step": 100,
            "T_int": (0.0, 0.1),
        }
    )

    df = run_simulation(params)
    v0 = df["Velocity_m_s"].iloc[0]
    v_end = df["Velocity_m_s"].iloc[-1]
    np.testing.assert_allclose(v_end, v0, rtol=1e-3, atol=1e-3)


def test_stiffness_damping_allows_contact_for_large_distance() -> None:
    params = _base_params()
    params.update(
        {
            "v0_init": -22.22,
            "d0": 12.0,
            "T_max": 1.0,
            "h_init": 1.0e-3,
            "step": 1000,
            "T_int": (0.0, 1.0),
        }
    )

    df = run_simulation(params)
    assert float(df["Position_x_m"].min()) <= 0.0
