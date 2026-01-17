"""
Energy balance regression tests for the simulator.

These tests target the current public API (run_simulation + config dicts).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from railway_simulator.core.engine import get_default_simulation_params, run_simulation


def _base_params() -> dict:
    params = get_default_simulation_params()
    T_max = 0.02
    h_init = 1.0e-4
    params.update(
        {
            "T_max": T_max,
            "h_init": h_init,
            "step": int(np.ceil(T_max / h_init)),
            "T_int": (0.0, T_max),
        }
    )
    return params


def test_energy_residual_is_reasonable() -> None:
    params = _base_params()

    df = run_simulation(params)

    assert "E_num_ratio" in df.columns
    max_ratio = float(np.nanmax(np.abs(df["E_num_ratio"].to_numpy())))
    assert max_ratio < 0.05, f"Energy residual too large: {max_ratio:.4f}"


def test_dissipation_is_monotone_non_negative() -> None:
    params = _base_params()

    df = run_simulation(params)

    diss = df["E_diss_total_J"].to_numpy(dtype=float)
    assert diss.min() >= -1e-6
    assert np.diff(diss).min() >= -1e-6
