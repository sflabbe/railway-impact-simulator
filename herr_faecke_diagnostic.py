"""
Herr Fäcke Diagnostic Tool
==========================
Quantitative analysis of energy balance residual.

NO poetry, NO assumptions. Only metrics.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from railway_simulator.core.engine import SimulationEngine, SimulationParams


def diagnose_peak_residual(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Find peak residual and analyze all energy components at that instant.

    Returns:
        Dictionary with complete diagnostic data.
    """
    # Find peak residual
    idx_peak = df["E_num_ratio"].idxmax()
    t_peak = df.loc[idx_peak, "Time_s"]

    # Extract state at peak
    E_num_ratio_peak = df.loc[idx_peak, "E_num_ratio"]
    E0 = df.loc[0, "E0_J"]

    # Energy components at peak
    E_kin_peak = df.loc[idx_peak, "E_kin_J"]
    E_pot_peak = df.loc[idx_peak, "E_pot_J"]
    E_mech_peak = df.loc[idx_peak, "E_mech_J"]
    W_ext_peak = df.loc[idx_peak, "W_ext_J"]
    E_diss_total_peak = df.loc[idx_peak, "E_diss_total_J"]
    E_num_peak = df.loc[idx_peak, "E_num_J"]

    # Energy increments at peak
    if idx_peak > 0:
        dt = df.loc[idx_peak, "Time_s"] - df.loc[idx_peak-1, "Time_s"]
        dE_kin = E_kin_peak - df.loc[idx_peak-1, "E_kin_J"]
        dE_pot = E_pot_peak - df.loc[idx_peak-1, "E_pot_J"]
        dE_mech = E_mech_peak - df.loc[idx_peak-1, "E_mech_J"]
        dW_ext = df.loc[idx_peak, "W_ext_J"] - df.loc[idx_peak-1, "W_ext_J"]
        dE_diss = E_diss_total_peak - df.loc[idx_peak-1, "E_diss_total_J"]
        dE_num = E_num_peak - df.loc[idx_peak-1, "E_num_J"]
    else:
        dt = 0
        dE_kin = dE_pot = dE_mech = dW_ext = dE_diss = dE_num = 0

    # Dissipation breakdown at peak
    E_diss_rayleigh = df.loc[idx_peak, "E_diss_rayleigh_J"]
    E_diss_bw = df.loc[idx_peak, "E_diss_bw_J"]
    E_diss_contact_damp = df.loc[idx_peak, "E_diss_contact_damp_J"]
    E_diss_friction = df.loc[idx_peak, "E_diss_friction_J"]
    E_diss_mass_contact = df.loc[idx_peak, "E_diss_mass_contact_J"]

    # Dissipation increments
    if idx_peak > 0:
        dE_diss_rayleigh = E_diss_rayleigh - df.loc[idx_peak-1, "E_diss_rayleigh_J"]
        dE_diss_bw = E_diss_bw - df.loc[idx_peak-1, "E_diss_bw_J"]
        dE_diss_contact_damp = E_diss_contact_damp - df.loc[idx_peak-1, "E_diss_contact_damp_J"]
        dE_diss_friction = E_diss_friction - df.loc[idx_peak-1, "E_diss_friction_J"]
        dE_diss_mass_contact = E_diss_mass_contact - df.loc[idx_peak-1, "E_diss_mass_contact_J"]
    else:
        dE_diss_rayleigh = dE_diss_bw = dE_diss_contact_damp = dE_diss_friction = dE_diss_mass_contact = 0

    # Potential breakdown
    E_pot_spring = df.loc[idx_peak, "E_pot_spring_J"]
    E_pot_contact = df.loc[idx_peak, "E_pot_contact_J"]

    if idx_peak > 0:
        dE_pot_spring = E_pot_spring - df.loc[idx_peak-1, "E_pot_spring_J"]
        dE_pot_contact = E_pot_contact - df.loc[idx_peak-1, "E_pot_contact_J"]
    else:
        dE_pot_spring = dE_pot_contact = 0

    # Ranking of culprits (contributors to dE_num)
    # dE_num should equal: dW_ext - dE_mech - dE_diss
    # So error comes from mismatch in increments

    # Expected: dE_num = dW_ext - (dE_kin + dE_pot) - dE_diss
    dE_num_expected = dW_ext - (dE_kin + dE_pot) - dE_diss
    consistency_error = abs(dE_num - dE_num_expected)

    culprits = []

    # Check each component contribution to mismatch
    if abs(dE_num) > 1e-6:
        # Potential energy mismatch
        share_pot = abs(dE_pot) / (abs(dE_num) + 1e-12)
        culprits.append(("Potential energy (dE_pot)", share_pot, dE_pot))

        # Dissipation mismatch
        share_diss = abs(dE_diss) / (abs(dE_num) + 1e-12)
        culprits.append(("Total dissipation (dE_diss)", share_diss, dE_diss))

        # External work mismatch
        share_wext = abs(dW_ext) / (abs(dE_num) + 1e-12)
        culprits.append(("External work (dW_ext)", share_wext, dW_ext))

        # Contact potential specific
        share_pot_contact = abs(dE_pot_contact) / (abs(dE_num) + 1e-12)
        culprits.append(("Contact potential (dE_pot_contact)", share_pot_contact, dE_pot_contact))

        # Contact damping specific
        share_contact_damp = abs(dE_diss_contact_damp) / (abs(dE_num) + 1e-12)
        culprits.append(("Contact damping (dE_diss_contact_damp)", share_contact_damp, dE_diss_contact_damp))

    culprits.sort(key=lambda x: x[1], reverse=True)

    diagnostic = {
        "idx_peak": idx_peak,
        "t_peak_s": t_peak,
        "dt_s": dt,
        "E_num_ratio_peak_pct": E_num_ratio_peak * 100,
        "E0_J": E0,
        "E_num_peak_J": E_num_peak,
        # State at peak
        "E_kin_J": E_kin_peak,
        "E_pot_J": E_pot_peak,
        "E_mech_J": E_mech_peak,
        "W_ext_J": W_ext_peak,
        "E_diss_total_J": E_diss_total_peak,
        # Increments
        "dE_kin_J": dE_kin,
        "dE_pot_J": dE_pot,
        "dE_mech_J": dE_mech,
        "dW_ext_J": dW_ext,
        "dE_diss_J": dE_diss,
        "dE_num_J": dE_num,
        # Dissipation breakdown
        "E_diss_rayleigh_J": E_diss_rayleigh,
        "E_diss_bw_J": E_diss_bw,
        "E_diss_contact_damp_J": E_diss_contact_damp,
        "E_diss_friction_J": E_diss_friction,
        "E_diss_mass_contact_J": E_diss_mass_contact,
        # Dissipation increments
        "dE_diss_rayleigh_J": dE_diss_rayleigh,
        "dE_diss_bw_J": dE_diss_bw,
        "dE_diss_contact_damp_J": dE_diss_contact_damp,
        "dE_diss_friction_J": dE_diss_friction,
        "dE_diss_mass_contact_J": dE_diss_mass_contact,
        # Potential breakdown
        "E_pot_spring_J": E_pot_spring,
        "E_pot_contact_J": E_pot_contact,
        "dE_pot_spring_J": dE_pot_spring,
        "dE_pot_contact_J": dE_pot_contact,
        # Consistency check
        "dE_num_expected_J": dE_num_expected,
        "consistency_error_J": consistency_error,
        # Culprits
        "culprits": culprits[:5],  # Top 5
    }

    if verbose:
        print("="*80)
        print("HERR FÄCKE DIAGNOSTIC REPORT - PEAK RESIDUAL")
        print("="*80)
        print(f"Peak at: t = {t_peak:.6f} s (step {idx_peak})")
        print(f"Timestep: dt = {dt:.6e} s")
        print(f"Peak residual: {E_num_ratio_peak*100:.2f}%")
        print(f"E_num = {E_num_peak/1e6:.6f} MJ (E0 = {E0/1e6:.3f} MJ)")
        print()
        print("-"*80)
        print("ENERGY STATE AT PEAK")
        print("-"*80)
        print(f"  E_kin     = {E_kin_peak/1e6:12.6f} MJ")
        print(f"  E_pot     = {E_pot_peak/1e6:12.6f} MJ")
        print(f"    Spring  = {E_pot_spring/1e6:12.6f} MJ")
        print(f"    Contact = {E_pot_contact/1e6:12.6f} MJ")
        print(f"  E_mech    = {E_mech_peak/1e6:12.6f} MJ")
        print(f"  W_ext     = {W_ext_peak/1e6:12.6f} MJ")
        print(f"  E_diss    = {E_diss_total_peak/1e6:12.6f} MJ")
        print()
        print("-"*80)
        print("INCREMENTS AT PEAK (dE/dt)")
        print("-"*80)
        print(f"  dE_kin            = {dE_kin/1e3:10.3f} kJ")
        print(f"  dE_pot            = {dE_pot/1e3:10.3f} kJ")
        print(f"    dE_pot_spring   = {dE_pot_spring/1e3:10.3f} kJ")
        print(f"    dE_pot_contact  = {dE_pot_contact/1e3:10.3f} kJ")
        print(f"  dW_ext            = {dW_ext/1e3:10.3f} kJ")
        print(f"  dE_diss           = {dE_diss/1e3:10.3f} kJ")
        print(f"    Rayleigh        = {dE_diss_rayleigh/1e3:10.3f} kJ")
        print(f"    Bouc-Wen        = {dE_diss_bw/1e3:10.3f} kJ")
        print(f"    Contact damp    = {dE_diss_contact_damp/1e3:10.3f} kJ")
        print(f"    Friction        = {dE_diss_friction/1e3:10.3f} kJ")
        print(f"    Mass contact    = {dE_diss_mass_contact/1e3:10.3f} kJ")
        print(f"  dE_num            = {dE_num/1e3:10.3f} kJ")
        print()
        print("-"*80)
        print("BALANCE CHECK")
        print("-"*80)
        print(f"  dE_num (computed)  = {dE_num/1e3:10.3f} kJ")
        print(f"  dE_num (expected)  = {dE_num_expected/1e3:10.3f} kJ")
        print(f"  Consistency error  = {consistency_error/1e3:10.3f} kJ")
        print()
        print("-"*80)
        print("RANKING - TOP CULPRITS")
        print("-"*80)
        for i, (name, share, value) in enumerate(culprits[:5], 1):
            print(f"  {i}. {name:40s}: share={share:6.2f}, value={value/1e3:10.3f} kJ")
        print("="*80)

    return diagnostic


def analyze_final_residual(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Analyze final residual at end of simulation.
    """
    idx_final = len(df) - 1
    t_final = df.loc[idx_final, "Time_s"]

    E_num_ratio_final = df.loc[idx_final, "E_num_ratio"]
    E0 = df.loc[0, "E0_J"]
    E_num_final = df.loc[idx_final, "E_num_J"]

    E_mech_final = df.loc[idx_final, "E_mech_J"]
    W_ext_final = df.loc[idx_final, "W_ext_J"]
    E_diss_final = df.loc[idx_final, "E_diss_total_J"]

    # Check balance
    E_total_computed = E_mech_final + E_diss_final
    E_total_expected = E0 + W_ext_final
    balance_error = E_total_computed - E_total_expected

    diagnostic = {
        "t_final_s": t_final,
        "E_num_ratio_final_pct": E_num_ratio_final * 100,
        "E_num_final_J": E_num_final,
        "E0_J": E0,
        "E_mech_final_J": E_mech_final,
        "W_ext_final_J": W_ext_final,
        "E_diss_final_J": E_diss_final,
        "E_total_computed_J": E_total_computed,
        "E_total_expected_J": E_total_expected,
        "balance_error_J": balance_error,
    }

    if verbose:
        print()
        print("="*80)
        print("FINAL STATE ANALYSIS")
        print("="*80)
        print(f"Final time: t = {t_final:.6f} s")
        print(f"Final residual: {E_num_ratio_final*100:.2f}%")
        print()
        print(f"  E0            = {E0/1e6:10.6f} MJ")
        print(f"  E_mech_final  = {E_mech_final/1e6:10.6f} MJ")
        print(f"  W_ext_final   = {W_ext_final/1e6:10.6f} MJ")
        print(f"  E_diss_final  = {E_diss_final/1e6:10.6f} MJ")
        print()
        print(f"  E_total (E_mech + E_diss) = {E_total_computed/1e6:10.6f} MJ")
        print(f"  E_total (E0 + W_ext)      = {E_total_expected/1e6:10.6f} MJ")
        print(f"  Balance error             = {balance_error/1e6:10.6f} MJ ({balance_error/E0*100:+.2f}%)")
        print("="*80)

    return diagnostic


def run_diagnostic(v0_kmh: float = 56, dt: float = 0.0001, alpha: float = -0.1):
    """
    Run simulation and full diagnostic.
    """
    print(f"Running simulation: v0={v0_kmh} km/h, dt={dt*1000} ms, alpha={alpha}")
    print()

    params = SimulationParams.from_presets()
    params.v0_kmh = v0_kmh
    params.dt = dt
    params.sim_time = 0.3
    params.alpha_hht = alpha

    engine = SimulationEngine(params)
    df = engine.run()

    # Peak analysis
    diag_peak = diagnose_peak_residual(df, verbose=True)

    # Final analysis
    diag_final = analyze_final_residual(df, verbose=True)

    # Save diagnostic to CSV
    diag_df = pd.DataFrame([{**diag_peak, **diag_final}])
    diag_df.to_csv("herr_faecke_diagnostic.csv", index=False)
    print()
    print("Diagnostic saved to: herr_faecke_diagnostic.csv")

    return df, diag_peak, diag_final


if __name__ == "__main__":
    print()
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "HERR FÄCKE DIAGNOSTIC TOOL" + " "*32 + "║")
    print("║" + " "*15 + "Energy Balance Residual Analysis" + " "*31 + "║")
    print("╚" + "="*78 + "╝")
    print()

    df, diag_peak, diag_final = run_diagnostic(v0_kmh=56, dt=0.0001, alpha=-0.1)

    print()
    print("BEFUND (Findings):")
    print(f"  - Peak residual: {diag_peak['E_num_ratio_peak_pct']:.2f}% at t={diag_peak['t_peak_s']:.6f}s")
    print(f"  - Final residual: {diag_final['E_num_ratio_final_pct']:.2f}%")
    print()

    if diag_peak['E_num_ratio_peak_pct'] > 5:
        print("NICHT PLAUSIBEL: Peak residual >5%. Further investigation required.")
    elif diag_final['E_num_ratio_final_pct'] > 1:
        print("TEILWEISE PLAUSIBEL: Final residual >1%. Improvement needed.")
    else:
        print("PLAUSIBEL: Energy balance within tolerance.")
