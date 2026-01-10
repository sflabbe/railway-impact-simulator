from __future__ import annotations

import sys

sys.path.insert(0, "src")

from railway_simulator.core.engine import ImpactSimulator, get_default_simulation_params, run_simulation


def test_picard_fallback_to_newton(monkeypatch) -> None:
    params = get_default_simulation_params()
    params.update(
        {
            "n_masses": 2,
            "masses": [1000.0, 1000.0],
            "x_init": [0.0, 1.0],
            "y_init": [0.0, 0.0],
            "fy": [1.0e6],
            "uy": [1.0e-3],
            "v0_init": -5.0,
            "d0": 2.0,
            "mu_s": 0.0,
            "mu_k": 0.0,
            "friction_model": "none",
            "solver": "picard",
            "solver_fail_policy": "switch",
            "max_iter": 5,
            "newton_tol": 1.0e-6,
            "alpha_hht": 0.0,
            "T_max": 1.0e-3,
            "h_init": 1.0e-3,
            "step": 1,
            "T_int": (0.0, 1.0e-3),
        }
    )

    monkeypatch.setattr(
        ImpactSimulator,
        "_check_convergence",
        staticmethod(lambda *_args, **_kwargs: 1.0e9),
    )

    df = run_simulation(params)
    assert df.attrs.get("fallback_used") is True
    assert df.attrs.get("converged_all_steps") is True
