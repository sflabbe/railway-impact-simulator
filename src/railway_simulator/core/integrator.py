"""HHT-α implicit time integration for structural dynamics.

This module implements the Hilber-Hughes-Taylor (HHT-α) method for
implicit time integration of structural dynamics problems, providing
numerical damping control and unconditional stability.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


class HHTAlphaIntegrator:
    """HHT-α implicit time integration method.

    The Hilber-Hughes-Taylor alpha (HHT-α) method is an implicit time
    integration scheme for structural dynamics that extends the Newmark
    method with controllable numerical damping in the high-frequency range.

    Mathematical Formulation
    ------------------------
    The HHT-α method integrates the equation of motion:

        M * a_{n+1} = (1-α)*F_{n+1} + α*F_n - (1-α)*C*v_{n+1} - α*C*v_n

    with displacement and velocity updates:

        q_{n+1} = q_n + h*v_n + h²*[(1/2 - β)*a_n + β*a_{n+1}]
        v_{n+1} = v_n + h*[(1 - γ)*a_n + γ*a_{n+1}]

    where:
        - α ∈ [-1/3, 0]: HHT parameter (α=0 reduces to Newmark)
        - β = (1 + α)²/4: Newmark parameter (unconditional stability)
        - γ = 1/2 + α: Newmark parameter

    Attributes
    ----------
    alpha : float
        HHT parameter. Typical values: -0.3 to 0.0
        - α = 0: No numerical damping (Newmark method)
        - α < 0: Numerical damping increases with |α|
        - α = -1/3: Maximum recommended damping
    beta : float
        Newmark beta parameter, computed as (1 + α)²/4
    gamma : float
        Newmark gamma parameter, computed as 0.5 + α
    n_lu : int
        Counter for number of linear solves (LU factorizations) performed

    Examples
    --------
    >>> # Initialize integrator with moderate damping
    >>> integrator = HHTAlphaIntegrator(alpha=-0.3)
    >>>
    >>> # In a time-stepping loop:
    >>> q_new, v_new = integrator.predict(
    ...     q=q_current,
    ...     qp=v_current,
    ...     qpp=a_current,
    ...     qpp_new=a_new,
    ...     h=dt
    ... )
    >>>
    >>> # Compute new acceleration
    >>> a_new = integrator.compute_acceleration(
    ...     M=mass_matrix,
    ...     R_internal=internal_forces_new,
    ...     R_internal_old=internal_forces_old,
    ...     R_contact=contact_forces_new,
    ...     R_contact_old=contact_forces_old,
    ...     R_friction=friction_forces_new,
    ...     R_friction_old=friction_forces_old,
    ...     R_mass_contact=mass_contact_forces_new,
    ...     R_mass_contact_old=mass_contact_forces_old,
    ...     C=damping_matrix,
    ...     qp=v_new,
    ...     qp_old=v_current
    ... )

    References
    ----------
    .. [1] Hilber, H. M., Hughes, T. J., and Taylor, R. L. "Improved
           numerical dissipation for time integration algorithms in
           structural dynamics." Earthquake Engineering & Structural
           Dynamics 5.3 (1977): 283-292.
    .. [2] Hughes, T. J. "The finite element method: linear static and
           dynamic finite element analysis." Courier Corporation, 2012.

    Notes
    -----
    - The method is unconditionally stable for α ∈ [-1/3, 0]
    - Second-order accurate for all values of α in the stable range
    - Numerical damping is proportional to |α| and affects high frequencies
    - For α = 0, reduces to the trapezoidal rule (average acceleration method)
    """

    def __init__(self, alpha: float):
        """Initialize HHT-α integrator.

        Parameters
        ----------
        alpha : float
            HHT parameter, typically in range [-0.3, 0.0]
            - α = 0: No numerical damping (Newmark trapezoidal)
            - α = -0.3: Moderate damping (recommended for impact problems)
            - α = -1/3: Maximum stable damping

        Raises
        ------
        ValueError
            If alpha is outside the stable range [-1/3, 0]
        """
        if not (-1.0 / 3.0 <= alpha <= 0.0):
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"HHT alpha={alpha} is outside stable range [-1/3, 0]. "
                "Unconditional stability may be compromised."
            )

        self.alpha = alpha
        self.beta = 0.25 * (1.0 + alpha) ** 2
        self.gamma = 0.5 + alpha

        # Count of linear solves (LU) performed in this run
        self.n_lu: int = 0

    def predict(
        self,
        q: np.ndarray,
        qp: np.ndarray,
        qpp: np.ndarray,
        qpp_new: np.ndarray,
        h: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Newmark predictor-corrector step for displacement and velocity.

        This method updates displacement and velocity using the Newmark
        formulas with the current and new acceleration.

        Parameters
        ----------
        q : np.ndarray
            Current displacement vector, shape (n_dof,)
        qp : np.ndarray
            Current velocity vector, shape (n_dof,)
        qpp : np.ndarray
            Current acceleration vector, shape (n_dof,)
        qpp_new : np.ndarray
            New (predicted or converged) acceleration vector, shape (n_dof,)
        h : float
            Time step size

        Returns
        -------
        q_new : np.ndarray
            Updated displacement vector, shape (n_dof,)
        qp_new : np.ndarray
            Updated velocity vector, shape (n_dof,)

        Notes
        -----
        The Newmark update formulas are:

            q_{n+1} = q_n + h*v_n + h²*[(1/2 - β)*a_n + β*a_{n+1}]
            v_{n+1} = v_n + h*[(1 - γ)*a_n + γ*a_{n+1}]

        where β and γ are the Newmark parameters determined by the HHT
        alpha parameter.
        """
        q_new = (
            q + h * qp + (0.5 - self.beta) * h**2 * qpp + self.beta * h**2 * qpp_new
        )

        qp_new = qp + (1.0 - self.gamma) * h * qpp + self.gamma * h * qpp_new

        return q_new, qp_new

    def compute_acceleration(
        self,
        M: np.ndarray,
        R_internal: np.ndarray,
        R_internal_old: np.ndarray,
        R_contact: np.ndarray,
        R_contact_old: np.ndarray,
        R_friction: np.ndarray,
        R_friction_old: np.ndarray,
        R_mass_contact: np.ndarray,
        R_mass_contact_old: np.ndarray,
        C: np.ndarray,
        qp: np.ndarray,
        qp_old: np.ndarray,
    ) -> np.ndarray:
        """Compute acceleration using HHT-α equilibrium equation.

        Solves the HHT-α form of the equation of motion:

            M * a_{n+1} = (1-α)*F_{n+1} + α*F_n - (1-α)*C*v_{n+1} - α*C*v_n

        where F includes all force contributions (internal, contact, friction,
        and mass-contact forces).

        Parameters
        ----------
        M : np.ndarray
            Mass matrix, shape (n_dof, n_dof)
        R_internal : np.ndarray
            Internal (spring) forces at new time step, shape (n_dof,)
        R_internal_old : np.ndarray
            Internal forces at previous time step, shape (n_dof,)
        R_contact : np.ndarray
            Contact forces at new time step, shape (n_dof,)
        R_contact_old : np.ndarray
            Contact forces at previous time step, shape (n_dof,)
        R_friction : np.ndarray
            Friction forces at new time step, shape (n_dof,)
        R_friction_old : np.ndarray
            Friction forces at previous time step, shape (n_dof,)
        R_mass_contact : np.ndarray
            Mass-to-mass contact forces at new time step, shape (n_dof,)
        R_mass_contact_old : np.ndarray
            Mass-to-mass contact forces at previous time step, shape (n_dof,)
        C : np.ndarray
            Damping matrix, shape (n_dof, n_dof)
        qp : np.ndarray
            Velocity at new time step, shape (n_dof,)
        qp_old : np.ndarray
            Velocity at previous time step, shape (n_dof,)

        Returns
        -------
        np.ndarray
            Acceleration vector at new time step, shape (n_dof,)

        Notes
        -----
        - This method increments the LU solve counter (self.n_lu)
        - Uses dense LU factorization via numpy.linalg.solve
        - All force vectors are assumed to follow the sign convention:
          positive = tension/expansion, negative = compression
        """
        # Assemble total forces
        R_total_new = R_internal + R_contact + R_friction + R_mass_contact
        R_total_old = (
            R_internal_old + R_contact_old + R_friction_old + R_mass_contact_old
        )

        # HHT-α weighted force term
        force = (
            (1.0 - self.alpha) * R_total_new
            + self.alpha * R_total_old
            - (1.0 - self.alpha) * (C @ qp)
            - self.alpha * (C @ qp_old)
        )

        # Count this linear solve (dense LU)
        self.n_lu += 1

        # Solve M * a = force for acceleration
        return np.linalg.solve(M, force)

    def reset_counters(self):
        """Reset performance counters.

        Useful for benchmarking individual simulation runs.
        """
        self.n_lu = 0

    def get_stability_info(self) -> dict:
        """Get information about integrator stability and properties.

        Returns
        -------
        dict
            Dictionary containing:
            - 'alpha': HHT parameter
            - 'beta': Newmark beta parameter
            - 'gamma': Newmark gamma parameter
            - 'is_stable': Whether parameters are in stable range
            - 'spectral_radius_inf': Spectral radius at infinite frequency
            - 'order': Accuracy order (always 2 for HHT-α)
        """
        # Spectral radius at infinite frequency
        rho_inf = (1.0 + self.alpha) / (1.0 - self.alpha)

        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "is_stable": -1.0 / 3.0 <= self.alpha <= 0.0,
            "spectral_radius_inf": rho_inf,
            "order": 2,
        }
