from __future__ import annotations

import sys

sys.path.insert(0, "src")

from railway_simulator.core.engine import (
    ImpactSimulator,
    get_default_simulation_params,
    run_simulation,
    NonConvergenceError,
)


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


def test_nonconvergence_error_has_diagnostics():
    """Test that NonConvergenceError provides diagnostic information."""
    err = NonConvergenceError(
        "Test failure message",
        step_idx=100,
        t=0.05,
        residual_norm=1e3,
        iter_count=25,
        dt_effective=1e-4,
        solver_type="newton",
        failure_stage="newton_fallback",
        state_snapshot={"x_front_last": -0.01, "in_contact": True},
        dt_reductions_used=3,
        fallback_attempted=True,
    )

    # Check attributes
    assert err.step_idx == 100
    assert err.t == 0.05
    assert err.residual_norm == 1e3
    assert err.failure_stage == "newton_fallback"
    assert err.fallback_attempted is True
    assert err.dt_reductions_used == 3

    # Check diagnostics dict
    diag = err.to_diagnostics_dict()
    assert diag["error_type"] == "NonConvergenceError"
    assert diag["step_idx"] == 100
    assert diag["t_last"] == 0.05
    assert diag["failure_stage"] == "newton_fallback"
    assert diag["x_front_last"] == -0.01
    assert diag["in_contact"] is True


def test_nonconvergence_error_raises_with_strict_policy(monkeypatch):
    """Test that solver raises NonConvergenceError when policy is 'raise'."""
    import pytest

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
            "solver_fail_policy": "raise",  # Strict policy
            "max_iter": 3,  # Very few iterations to force failure
            "newton_tol": 1.0e-12,  # Very strict tolerance
            "alpha_hht": 0.0,
            "T_max": 1.0e-3,
            "h_init": 1.0e-3,
            "step": 1,
            "T_int": (0.0, 1.0e-3),
        }
    )

    # Force Picard to never converge
    monkeypatch.setattr(
        ImpactSimulator,
        "_check_convergence",
        staticmethod(lambda *_args, **_kwargs: 1.0e9),
    )

    with pytest.raises(NonConvergenceError) as exc_info:
        run_simulation(params)

    # Check that exception has diagnostic info
    assert exc_info.value.failure_stage == "picard"
    assert exc_info.value.fallback_attempted is False
