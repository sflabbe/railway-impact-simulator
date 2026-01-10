from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from railway_simulator.core.engine import ImpactSimulator, SimulationParams, get_default_simulation_params


def _base_two_mass_params() -> dict:
    params = get_default_simulation_params()
    params.update(
        {
            "n_masses": 2,
            "masses": [1000.0, 1000.0],
            "x_init": [0.0, 1.0],
            "y_init": [0.0, 0.0],
            "fy": [1.0e6],
            "uy": [1.0e-3],
            "v0_init": 0.0,
            "d0": 0.0,
            "mu_s": 0.0,
            "mu_k": 0.0,
            "friction_model": "none",
            "h_init": 1.0e-3,
            "T_max": 1.0e-3,
            "step": 1,
            "T_int": (0.0, 1.0e-3),
        }
    )
    return params


def test_mass_contact_vector_indexing() -> None:
    params = SimulationParams(**_base_two_mass_params())
    sim = ImpactSimulator(params)

    u_spring = np.zeros((1, 2))
    u_spring[0, 1] = -0.96

    R_mass_contact = np.zeros((4, 2))
    q_vec = np.array([0.0, 0.0, 0.0, 0.0])
    qp_vec = np.array([0.0, -1.0, 0.0, 0.0])

    sim._compute_mass_contact(0, u_spring, q_vec, qp_vec, R_mass_contact)

    assert sim.mass_contact_active[0]
    assert R_mass_contact[0, 1] < 0.0
    assert R_mass_contact[1, 1] > 0.0
    np.testing.assert_allclose(R_mass_contact[0, 1], -R_mass_contact[1, 1])
