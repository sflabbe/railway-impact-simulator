"""Regression tests for wall-contact damping state.

These tests protect the bug where a single global contact_active flag allowed later
masses to keep the default v0_contact=1 m/s.  The dissipative laws then saw a
spurious du/v0 ratio and produced non-physical force spikes.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from railway_simulator.core.contact import ContactModels
from railway_simulator.core.engine import run_simulation


def _compression_magnitude(force: np.ndarray) -> float:
    # ContactModels returns negative compression and zero for no contact.
    return float(-force[0])


def test_signed_penetration_rate_increases_then_reduces_dissipative_force() -> None:
    """Approach should increase force; restitution should reduce it.

    Engine convention: u_contact=q_x<0 in contact, du_contact=qdot_x.
    Therefore delta_dot=-du_contact is positive on approach and negative on
    restitution.
    """
    k = 45.0e6
    cr = 0.76
    u = np.array([-0.10])
    v0 = np.array([6.0])

    f_backbone = _compression_magnitude(
        ContactModels.compute_force(u, np.array([0.0]), v0, k, cr, "lankarani-nikravesh")
    )
    f_approach = _compression_magnitude(
        ContactModels.compute_force(u, np.array([-1.0]), v0, k, cr, "lankarani-nikravesh")
    )
    f_restitution = _compression_magnitude(
        ContactModels.compute_force(u, np.array([+1.0]), v0, k, cr, "lankarani-nikravesh")
    )

    assert f_approach > f_backbone > f_restitution >= 0.0


def test_dissipative_contact_clamps_tension_to_zero() -> None:
    """Large restitution velocity may make raw force tensile; output must be zero."""
    k = 45.0e6
    cr = 0.76
    u = np.array([-0.10])
    v0 = np.array([1.0])
    f = ContactModels.compute_force(
        u, np.array([+50.0]), v0, k, cr, "lankarani-nikravesh"
    )
    assert f[0] == pytest.approx(0.0)


def _run_traxx_lankarani_fpeak(vn_ms: float) -> float:
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "configs" / "traxx_freight.yml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(
        v0_init=-float(vn_ms),
        contact_model="lankarani-nikravesh",
        T_max=0.10,
        T_int=(0.0, 0.10),
        step=1000,
        h_init=1.0e-4,
        solver="picard",
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        df = run_simulation(cfg, emit_peak_diagnostics=False)
    return float(df["Impact_Force_MN"].max())


@pytest.mark.slow
def test_traxx_lankarani_has_no_isolated_force_spike_around_6ms() -> None:
    """The former v0_contact bug produced a local spike at vn≈6 m/s."""
    f575 = _run_traxx_lankarani_fpeak(5.75)
    f600 = _run_traxx_lankarani_fpeak(6.00)
    f625 = _run_traxx_lankarani_fpeak(6.25)

    assert f575 < f600 < f625
    local_linear_mid = 0.5 * (f575 + f625)
    assert f600 <= 1.05 * local_linear_mid
