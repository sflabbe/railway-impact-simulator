"""
Tests for Euler-Lagrange energy balance.

Verifies that the energy tracking correctly implements:
    E_num = E0 + W_ext - (E_mech + E_diss)

And that |E_num| / E0 < 1% for typical cases.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from railway_simulator.core.engine import SimulationEngine, SimulationParams


def test_energy_balance_conservative_case():
    """
    Test energy balance for conservative case (no damping).

    With no dissipation, numerical residual should be very small (<0.2%).
    """
    params = SimulationParams.from_presets()

    # Create conservative case: no damping, fully elastic springs
    params.v0_kmh = 50  # Moderate speed
    params.sim_time = 0.2  # Short simulation
    params.dt = 0.0001  # Fine timestep
    params.bw_a = 1.0  # Fully elastic (no hysteresis)
    params.bw_beta = 0.0  # No hysteresis
    params.bw_gamma = 0.0  # No hysteresis
    params.alpha_damp = 0.0  # No Rayleigh damping
    params.beta_damp = 0.0  # No Rayleigh damping
    params.contact_model = "hertz"  # Elastic contact (no damping)
    params.mu_coulomb = 0.0  # No friction

    engine = SimulationEngine(params)
    df = engine.run()

    # Check energy balance
    E_num_ratio_max = df["E_num_ratio"].max()

    print(f"Conservative case: max |E_num|/E0 = {E_num_ratio_max * 100:.4f}%")

    # For conservative case, numerical error should be very small
    assert E_num_ratio_max < 0.002, (
        f"Conservative case failed: |E_num|/E0 = {E_num_ratio_max*100:.2f}% > 0.2%"
    )


def test_energy_balance_dissipative_case():
    """
    Test energy balance for dissipative case (with damping).

    With proper dissipation tracking, residual should be <1%.
    """
    params = SimulationParams.from_presets()

    # Create dissipative case: with damping
    params.v0_kmh = 100  # Higher speed
    params.sim_time = 0.3  # Longer simulation
    params.dt = 0.0001  # Fine timestep
    params.bw_a = 0.5  # Partially hysteretic
    params.contact_model = "lankarani-nikravesh"  # Dissipative contact
    params.alpha_damp = 0.01  # Small Rayleigh damping
    params.beta_damp = 0.0001  # Small Rayleigh damping

    engine = SimulationEngine(params)
    df = engine.run()

    # Check energy balance
    E_num_ratio_max = df["E_num_ratio"].max()
    E_diss_final = df["E_diss_total_J"].iloc[-1]
    E0 = df["E0_J"].iloc[0]

    print(f"Dissipative case: max |E_num|/E0 = {E_num_ratio_max * 100:.4f}%")
    print(f"  E0 = {E0/1e6:.3f} MJ")
    print(f"  E_diss_final = {E_diss_final/1e6:.3f} MJ ({E_diss_final/E0*100:.1f}%)")

    # Dissipation should be significant (>5% of initial energy)
    assert E_diss_final > 0.05 * E0, "Dissipation should be significant"

    # Numerical residual should be <1%
    assert E_num_ratio_max < 0.01, (
        f"Dissipative case failed: |E_num|/E0 = {E_num_ratio_max*100:.2f}% > 1%"
    )


def test_energy_components_non_negative():
    """
    Test that energy components have physically meaningful signs.

    - Kinetic energy T >= 0
    - Potential energy V >= 0 (assuming reference at initial state)
    - Dissipation E_diss >= 0 (energy cannot be "un-dissipated")
    """
    params = SimulationParams.from_presets()
    params.v0_kmh = 80
    params.sim_time = 0.2
    params.dt = 0.0001

    engine = SimulationEngine(params)
    df = engine.run()

    # Check non-negativity
    assert (df["E_kin_J"] >= -1e-6).all(), "Kinetic energy should be non-negative"
    assert (df["E_pot_J"] >= -1e-6).all(), "Potential energy should be non-negative"

    # Dissipation components should be non-negative (monotonically increasing)
    assert (df["E_diss_rayleigh_J"] >= -1e-6).all(), "Rayleigh dissipation should be non-negative"
    assert (df["E_diss_bw_J"] >= -1e-6).all(), "Bouc-Wen dissipation should be non-negative"
    assert (df["E_diss_contact_damp_J"] >= -1e-6).all(), "Contact damping dissipation should be non-negative"
    assert (df["E_diss_friction_J"] >= -1e-6).all(), "Friction dissipation should be non-negative"
    assert (df["E_diss_mass_contact_J"] >= -1e-6).all(), "Mass contact dissipation should be non-negative"

    # Total dissipation should be monotonically increasing (within numerical tolerance)
    dE_diss = np.diff(df["E_diss_total_J"])
    assert (dE_diss >= -1e-6).all(), "Total dissipation should not decrease"


def test_euler_lagrange_identity():
    """
    Test Euler-Lagrange energy identity:
        E_num = E0 + W_ext - (E_mech + E_diss)

    This should hold exactly (within floating point precision).
    """
    params = SimulationParams.from_presets()
    params.v0_kmh = 75
    params.sim_time = 0.25
    params.dt = 0.0001

    engine = SimulationEngine(params)
    df = engine.run()

    # Compute identity manually
    E0 = df["E0_J"].iloc[0]
    E_num_computed = E0 + df["W_ext_J"] - (df["E_mech_J"] + df["E_diss_total_J"])
    E_num_stored = df["E_num_J"]

    # Should match exactly
    relative_error = np.abs(E_num_computed - E_num_stored) / (abs(E0) + 1e-12)
    max_relative_error = relative_error.max()

    print(f"Identity check: max relative error = {max_relative_error:.2e}")

    assert max_relative_error < 1e-10, (
        f"Euler-Lagrange identity violated: max error = {max_relative_error:.2e}"
    )


if __name__ == "__main__":
    print("Running energy balance tests...")
    print("\n" + "="*60)
    print("Test 1: Conservative case")
    print("="*60)
    test_energy_balance_conservative_case()

    print("\n" + "="*60)
    print("Test 2: Dissipative case")
    print("="*60)
    test_energy_balance_dissipative_case()

    print("\n" + "="*60)
    print("Test 3: Non-negativity")
    print("="*60)
    test_energy_components_non_negative()

    print("\n" + "="*60)
    print("Test 4: Euler-Lagrange identity")
    print("="*60)
    test_euler_lagrange_identity()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
