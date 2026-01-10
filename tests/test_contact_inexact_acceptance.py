import numpy as np

from railway_simulator.core.engine import should_accept_contact_inexact


def test_contact_inexact_acceptance() -> None:
    contact_tol = 5.0e-2
    inc_tol = 1.0e-6
    contact_only = True
    soft_reject_err = 2.0e-1

    assert should_accept_contact_inexact(
        err=0.037,
        dq_rel=1.0e-7,
        contact_active=True,
        contact_only=contact_only,
        contact_tol=contact_tol,
        dq_rel_tol=inc_tol,
        soft_reject_err=soft_reject_err,
        mech_energy=1.0,
    )

    assert not should_accept_contact_inexact(
        err=0.037,
        dq_rel=1.0e-4,
        contact_active=True,
        contact_only=contact_only,
        contact_tol=contact_tol,
        dq_rel_tol=inc_tol,
        soft_reject_err=soft_reject_err,
        mech_energy=1.0,
    )

    assert not should_accept_contact_inexact(
        err=0.037,
        dq_rel=1.0e-7,
        contact_active=False,
        contact_only=contact_only,
        contact_tol=contact_tol,
        dq_rel_tol=inc_tol,
        soft_reject_err=soft_reject_err,
        mech_energy=1.0,
    )

    assert not should_accept_contact_inexact(
        err=0.3,
        dq_rel=1.0e-7,
        contact_active=True,
        contact_only=contact_only,
        contact_tol=contact_tol,
        dq_rel_tol=inc_tol,
        soft_reject_err=soft_reject_err,
        mech_energy=1.0,
    )

    assert not should_accept_contact_inexact(
        err=0.037,
        dq_rel=np.nan,
        contact_active=True,
        contact_only=contact_only,
        contact_tol=contact_tol,
        dq_rel_tol=inc_tol,
        soft_reject_err=soft_reject_err,
        mech_energy=1.0,
    )
