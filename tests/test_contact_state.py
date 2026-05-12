from __future__ import annotations

import numpy as np
import pytest

from railway_simulator.core.contact import ContactModels
from railway_simulator.core.contact_state import ContactState


def test_contact_state_tracks_v0_per_mass() -> None:
    state = ContactState.initial(3)
    q = np.array([-0.01, 0.02, -0.03, 0.0, 0.0, 0.0])
    qp = np.array([-2.0, -99.0, -5.0, 0.0, 0.0, 0.0])

    updated = state.update(q, qp, n_masses=3)

    assert updated.active.tolist() == [True, False, True]
    assert updated.v0_contact[:3].tolist() == pytest.approx([2.0, 1.0, 5.0])


def test_contact_state_resets_lost_contact_only_for_that_mass() -> None:
    state = ContactState.initial(2).update(
        np.array([-0.01, -0.02, 0.0, 0.0]),
        np.array([-3.0, -4.0, 0.0, 0.0]),
        n_masses=2,
    )
    updated = state.update(
        np.array([0.01, -0.02, 0.0, 0.0]),
        np.array([+1.0, -4.0, 0.0, 0.0]),
        n_masses=2,
    )

    assert updated.active.tolist() == [False, True]
    assert updated.v0_contact[:2].tolist() == pytest.approx([1.0, 4.0])


def test_contact_kinematics_uses_engine_sign_convention() -> None:
    kin = ContactState.kinematics(
        np.array([-0.01, 0.02, 0.0, 0.0]),
        np.array([-2.0, +3.0, 0.0, 0.0]),
        n_masses=2,
    )
    assert kin.u_contact[:2].tolist() == pytest.approx([-0.01, 0.0])
    assert kin.du_contact[:2].tolist() == pytest.approx([-2.0, 0.0])


def test_unknown_contact_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown contact model"):
        ContactModels.compute_force(
            np.array([-0.01]),
            np.array([-1.0]),
            np.array([1.0]),
            1.0e6,
            0.7,
            "typo-model",
        )
