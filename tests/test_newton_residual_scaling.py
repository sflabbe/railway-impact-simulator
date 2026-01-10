"""Tests for Newton solver residual scaling fix.

This test verifies that the Newton solver correctly handles the case where
the system starts in equilibrium (R_total_old ≈ 0), which previously caused
pathologically small residual references and false non-convergence in step 0.

Bug description:
- When system starts in equilibrium (no spring deformation, no contact),
  R_total_old = 0 at step 0
- With stiffness-proportional damping and uniform initial velocity (rigid body mode),
  C @ v0 = 0 as well
- The old reference: ref = ||R_total_old|| + 1.0 = 1.0
- This made the relative error huge even for small absolute residuals
- Fix: Use max(||R_total_old||, ||force_rhs||, 1.0) to include current RHS forces

Author: Claude Code Audit
Date: 2026-01-10
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from railway_simulator.core.engine import (
    run_simulation,
    NonConvergenceError,
    get_default_simulation_params,
)


class TestNewtonResidualScaling:
    """Test cases for Newton residual scaling fix."""

    def test_newton_step0_converges_with_rayleigh_damping(self):
        """Newton should converge at step 0 with Rayleigh damping.

        This tests the fix for the pathological residual scaling bug where
        ref = ||R_total_old|| + 1.0 was too small. With Rayleigh damping,
        the mass-proportional term ensures non-zero damping forces even
        for rigid body modes.
        """
        params = get_default_simulation_params()

        # Configure a case that starts in equilibrium
        # - No initial contact (d0 > 0)
        # - No spring deformation (uniform x positions from config)
        # - Rayleigh damping (has mass-proportional term)
        params.update({
            "n_masses": 3,
            "masses": [1000.0, 1000.0, 1000.0],
            "x_init": [0.0, 10.0, 20.0],
            "y_init": [0.0, 0.0, 0.0],
            "fy": [1e6, 1e6],  # 2 springs between 3 masses
            "uy": [0.01, 0.01],
            "v0_init": -10.0,  # Uniform velocity = rigid body mode
            "d0": 5.0,  # Gap to wall - no initial contact
            "k_wall": 1e8,
            "cr_wall": 0.8,
            "contact_model": "hooke",
            "damping_model": "rayleigh",  # Rayleigh damping has mass term
            "damping_zeta": 0.05,
            "solver": "newton",
            "newton_tol": 1e-5,
            "max_iter": 50,
            "h_init": 1e-4,
            "step": 10,  # Just 10 steps to verify step 0 works
            "T_max": 0.001,
            "T_int": (0.0, 0.001),
            "friction_model": "none",
            "mu_s": 0.0,
            "mu_k": 0.0,
            "sigma_0": 0.0,
            "sigma_1": 0.0,
            "sigma_2": 0.0,
        })

        # This should NOT raise NonConvergenceError
        df = run_simulation(params)

        # Verify simulation completed
        assert len(df) > 1, "Simulation should produce multiple time steps"
        assert df.attrs.get("converged_all_steps", False), "All steps should converge"

    def test_newton_residual_ref_not_pathologically_small(self):
        """Verify residual reference is not pathologically small (>> 1.0).

        The fix ensures that the reference includes the current RHS forces,
        so even when R_total_old = 0, the reference should be reasonable.
        """
        params = get_default_simulation_params()

        params.update({
            "n_masses": 3,
            "masses": [1000.0, 1000.0, 1000.0],
            "x_init": [0.0, 10.0, 20.0],
            "y_init": [0.0, 0.0, 0.0],
            "fy": [1e6, 1e6],
            "uy": [0.01, 0.01],
            "v0_init": -10.0,
            "d0": 5.0,
            "k_wall": 1e8,
            "cr_wall": 0.8,
            "contact_model": "hooke",
            "damping_model": "rayleigh",  # Rayleigh damping has mass term
            "damping_zeta": 0.05,
            "solver": "newton",
            "newton_tol": 1e-5,
            "max_iter": 50,
            "h_init": 1e-4,
            "step": 10,
            "T_max": 0.001,
            "T_int": (0.0, 0.001),
            "friction_model": "none",
            "mu_s": 0.0,
            "mu_k": 0.0,
            "sigma_0": 0.0,
            "sigma_1": 0.0,
            "sigma_2": 0.0,
        })

        df = run_simulation(params)

        # The max_residual_seen should be reasonable (< 1.0 for a well-converged case)
        # Before the fix, it was >> 1.0 due to division by small reference
        max_residual = df.attrs.get("max_residual_seen", float("inf"))
        assert max_residual < 1.0, f"Max residual {max_residual} should be < 1.0"

    def test_newton_linear_case_converges_quickly(self):
        """Newton should converge in 1-2 iterations for a linear case without contact.

        When there's no contact and no hysteresis (linear springs), Newton
        should converge very quickly since the Jacobian is exact.
        """
        params = get_default_simulation_params()

        params.update({
            "n_masses": 3,
            "masses": [1000.0, 1000.0, 1000.0],
            "x_init": [0.0, 10.0, 20.0],
            "y_init": [0.0, 0.0, 0.0],
            "fy": [1e6, 1e6],
            "uy": [0.01, 0.01],
            "v0_init": -1.0,  # Slow velocity
            "d0": 50.0,  # Large gap - no contact in simulation
            "k_wall": 1e8,
            "cr_wall": 0.8,
            "contact_model": "hooke",
            "damping_model": "rayleigh",  # Use Rayleigh damping
            "damping_zeta": 0.05,
            "solver": "newton",
            "newton_tol": 1e-6,
            "max_iter": 50,
            "h_init": 1e-4,
            "step": 100,
            "T_max": 0.01,
            "T_int": (0.0, 0.01),
            "friction_model": "none",
            "mu_s": 0.0,
            "mu_k": 0.0,
            "sigma_0": 0.0,
            "sigma_1": 0.0,
            "sigma_2": 0.0,
            # Linear Bouc-Wen (a=1 means purely elastic)
            "bw_a": 1.0,
            "bw_A": 1.0,
            "bw_beta": 0.5,
            "bw_gamma": 0.5,
            "bw_n": 2,
        })

        df = run_simulation(params)

        # Should complete without error
        assert len(df) > 1, "Simulation should produce results"

        # Max iterations per step should be low (1-3 for linear case)
        max_iters_step = df.attrs.get("max_iters_step", 100)
        assert max_iters_step <= 5, f"Linear case should converge quickly (max_iters={max_iters_step})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
