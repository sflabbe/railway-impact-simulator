"""Per-mass wall contact state tracking.

This module isolates the bug-prone part of dissipative contact models: the
initial impact speed must be tracked per x-mass, not with a global flag.  The
engine convention is preserved:

    u_contact = q_x < 0 in wall penetration
    du_contact = qdot_x

ContactModels then converts du_contact to signed penetration rate
``delta_dot = -du_contact``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactKinematics:
    """Computed wall-contact kinematic arrays for one solver state."""

    u_contact: np.ndarray
    du_contact: np.ndarray


@dataclass(frozen=True)
class ContactState:
    """Per-mass contact-active flags and initial approach speeds."""

    active: np.ndarray
    v0_contact: np.ndarray

    @classmethod
    def initial(cls, n_masses: int) -> "ContactState":
        n = int(n_masses)
        if n <= 0:
            raise ValueError("n_masses must be > 0")
        return cls(active=np.zeros(n, dtype=bool), v0_contact=np.ones(2 * n, dtype=float))

    @classmethod
    def coerce(
        cls,
        active: np.ndarray | bool,
        v0_contact: np.ndarray,
        *,
        n_masses: int,
    ) -> "ContactState":
        """Create a ContactState from legacy engine arrays or scalar flags."""
        n = int(n_masses)
        if np.isscalar(active):
            active_arr = np.full(n, bool(active), dtype=bool)
        else:
            active_arr = np.asarray(active, dtype=bool).copy()
            if active_arr.size != n:
                active_arr = np.resize(active_arr, n).astype(bool)

        v0 = np.asarray(v0_contact, dtype=float).copy()
        if v0.size != 2 * n:
            v0_new = np.ones(2 * n, dtype=float)
            take = min(v0_new.size, v0.size)
            if take:
                v0_new[:take] = v0[:take]
            v0 = v0_new
        v0[:n] = np.where(np.isfinite(v0[:n]) & (np.abs(v0[:n]) > 1.0e-8), np.abs(v0[:n]), 1.0)
        return cls(active=active_arr, v0_contact=v0)

    @staticmethod
    def kinematics(q: np.ndarray, qp: np.ndarray, *, n_masses: int) -> ContactKinematics:
        """Return u_contact and du_contact arrays from displacement/velocity."""
        n = int(n_masses)
        q_arr = np.asarray(q, dtype=float)
        qp_arr = np.asarray(qp, dtype=float)
        dof = 2 * n
        u_contact = np.zeros(dof, dtype=float)
        du_contact = np.zeros(dof, dtype=float)
        in_contact = q_arr[:n] < 0.0
        u_contact[:n] = np.where(in_contact, q_arr[:n], 0.0)
        du_contact[:n] = np.where(in_contact, qp_arr[:n], 0.0)
        return ContactKinematics(u_contact=u_contact, du_contact=du_contact)

    def update(self, q: np.ndarray, qp: np.ndarray, *, n_masses: int | None = None) -> "ContactState":
        """Update active flags and v0_contact from the current state.

        New contact is registered only for masses that are both penetrating and
        approaching the wall.  Lost contact resets the corresponding v0 to 1.0,
        preserving historical engine behavior outside contact.
        """
        n = int(n_masses or self.active.size)
        q_arr = np.asarray(q, dtype=float)
        qp_arr = np.asarray(qp, dtype=float)
        active = np.asarray(self.active, dtype=bool).copy()
        v0 = np.asarray(self.v0_contact, dtype=float).copy()
        if active.size != n:
            active = np.resize(active, n).astype(bool)
        if v0.size != 2 * n:
            v0_fixed = np.ones(2 * n, dtype=float)
            take = min(v0_fixed.size, v0.size)
            if take:
                v0_fixed[:take] = v0[:take]
            v0 = v0_fixed

        in_contact = q_arr[:n] < 0.0
        if not np.any(in_contact):
            active[:] = False
            v0[:n] = 1.0
            return ContactState(active=active, v0_contact=v0)

        approaching = in_contact & (qp_arr[:n] < 0.0)
        new_contact = approaching & (~active)
        if np.any(new_contact):
            active[new_contact] = True
            v0[:n][new_contact] = np.maximum(np.abs(qp_arr[:n][new_contact]), 1.0e-8)

        lost_contact = (~in_contact) & active
        if np.any(lost_contact):
            active[lost_contact] = False
            v0[:n][lost_contact] = 1.0

        return ContactState(active=active, v0_contact=v0)
