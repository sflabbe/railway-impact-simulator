"""Contact force models for railway impact simulation.

This module implements various normal contact force models used in
impact and collision dynamics, including linear (Hooke), Hertzian,
and dissipative models from the literature.
"""

from __future__ import annotations
from typing import Dict, Callable, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from railway_simulator.config.laws import ForceDisplacementLaw
import numpy as np


class ContactModels:
    """Normal contact force model implementations.

    This class provides a unified interface for computing normal contact
    forces during impact using various models from the literature. All
    models support unilateral contact (compression only, no tension).

    Available Models
    ----------------
    - **hooke**: Linear elastic contact (k*δ)
    - **hertz**: Nonlinear Hertzian contact (k*δ^1.5)
    - **hunt-crossley**: Hertz with velocity-dependent damping
    - **lankarani-nikravesh**: Modified Hunt-Crossley for better energy dissipation
    - **flores**: Alternative dissipative model
    - **gonthier**: Contact model with modified damping
    - **ye**: Linear stiffness with velocity damping
    - **pant-wijeyewickrema**: Linear contact with modified damping
    - **anagnostopoulos**: Linear contact (default)

    Model Equations
    ---------------
    All models follow the general form:

        F = -k * f(δ) * [1 + g(cr, δ̇, v₀)]

    where:
        - k: Contact stiffness
        - δ: Penetration depth (positive when in contact)
        - δ̇: Penetration velocity
        - v₀: Initial impact velocity
        - cr: Coefficient of restitution
        - f(δ): Force-penetration relationship (linear or nonlinear)
        - g(...): Damping term

    Examples
    --------
    >>> # Compute contact forces using Hunt-Crossley model
    >>> u_contact = np.array([0.0, -0.01, -0.02])  # penetration (negative = contact)
    >>> du_contact = np.array([0.0, -1.0, -0.5])    # penetration velocity
    >>> v0 = np.array([0.0, 2.0, 2.0])              # initial impact velocity
    >>> F = ContactModels.compute_force(
    ...     u_contact=u_contact,
    ...     du_contact=du_contact,
    ...     v0=v0,
    ...     k_wall=1e8,           # N/m^1.5 for Hertzian
    ...     cr_wall=0.5,          # coefficient of restitution
    ...     model="hunt-crossley"
    ... )

    References
    ----------
    .. [1] Hunt, K. H., and Crossley, F. R. E. "Coefficient of restitution
           interpreted as damping in vibroimpact." Journal of applied
           mechanics 42.2 (1975): 440-445.
    .. [2] Lankarani, H. M., and Nikravesh, P. E. "A contact force model
           with hysteresis damping for impact analysis of multibody systems."
           Journal of mechanical design 112.3 (1990): 369-376.
    .. [3] Flores, P., et al. "On the continuous contact force models for
           soft materials in multibody dynamics." Multibody System Dynamics
           25.3 (2011): 357-375.
    """

    # Dictionary of contact force models
    # All models return negative force for compression
    # Parameters: k, d (penetration), cr, dv (penetration velocity), v0 (initial velocity)
    MODELS: Dict[str, Callable[[float, float, float, float, float], float]] = {
        "hooke": lambda k, d, cr, dv, v0: -k * d,
        "hertz": lambda k, d, cr, dv, v0: -k * d**1.5,
        "hunt-crossley": lambda k, d, cr, dv, v0: (
            -k * d**1.5 * (1.0 + 3.0 * (1.0 - cr) / 2.0 * (dv / v0))
        ),
        "lankarani-nikravesh": lambda k, d, cr, dv, v0: (
            -k * d**1.5 * (1.0 + 3.0 * (1.0 - cr**2) / 4.0 * (dv / v0))
        ),
        "flores": lambda k, d, cr, dv, v0: (
            -k * d**1.5 * (1.0 + 8.0 * (1.0 - cr) / (5.0 * cr) * (dv / v0))
        ),
        "gonthier": lambda k, d, cr, dv, v0: (
            -k * d**1.5 * (1.0 + (1.0 - cr**2) / cr * (dv / v0))
        ),
        "ye": lambda k, d, cr, dv, v0: (
            -k * d * (1.0 + 3.0 * (1.0 - cr) / (2.0 * cr) * (dv / v0))
        ),
        "pant-wijeyewickrema": lambda k, d, cr, dv, v0: (
            -k * d * (1.0 + 3.0 * (1.0 - cr**2) / (2.0 * cr**2) * (dv / v0))
        ),
        "anagnostopoulos": lambda k, d, cr, dv, v0: (
            -k * d * (1.0 + 3.0 * (1.0 - cr) / (2.0 * cr) * (dv / v0))
        ),
    }

    @staticmethod
    def compute_force(
        u_contact: np.ndarray,
        du_contact: np.ndarray,
        v0: np.ndarray,
        k_wall: float,
        cr_wall: float,
        model: str,
        contact_law: Optional["ForceDisplacementLaw"] = None,
    ) -> np.ndarray:
        """Compute normal contact forces with unilateral constraint.

        This method computes contact forces for all degrees of freedom,
        enforcing the unilateral contact constraint (compression only,
        no tension).

        Parameters
        ----------
        u_contact : np.ndarray
            Penetration displacement for each DOF. Negative values indicate
            contact (penetration into the wall). Shape: (n_dof,)
        du_contact : np.ndarray
            Penetration velocity for each DOF (time derivative of u_contact).
            Shape: (n_dof,)
        v0 : np.ndarray
            Initial impact velocity for each DOF. Used to normalize the
            damping term in velocity-dependent models. Shape: (n_dof,)
        k_wall : float
            Wall stiffness parameter. Units depend on the model:
            - Linear models (hooke, ye, etc.): N/m
            - Hertzian models (hertz, hunt-crossley, etc.): N/m^1.5
        cr_wall : float
            Coefficient of restitution (0 = perfectly plastic, 1 = perfectly elastic).
            Typical values: 0.3-0.8 for structural impacts.
        model : str
            Name of the contact model to use. Must be one of the keys in
            ContactModels.MODELS. Default fallback is "anagnostopoulos".

        Returns
        -------
        np.ndarray
            Contact force for each DOF. Negative values indicate compression
            forces. Zero when not in contact. Shape: (n_dof,)

        Notes
        -----
        - Contact is detected when u_contact < 0 (penetration)
        - Forces are only computed where penetration occurs (δ = -u > 0)
        - The method prevents division by zero in v0 (uses 1e-8 minimum)
        - Unilateral constraint enforced: forces clamped to non-positive values

        Examples
        --------
        >>> # Single mass impacting a wall
        >>> u = np.array([-0.005])  # 5mm penetration
        >>> du = np.array([-0.5])    # approaching at 0.5 m/s
        >>> v0 = np.array([1.0])     # initial impact at 1 m/s
        >>> F = ContactModels.compute_force(
        ...     u_contact=u,
        ...     du_contact=du,
        ...     v0=v0,
        ...     k_wall=1e8,
        ...     cr_wall=0.6,
        ...     model="hunt-crossley"
        ... )
        """
        u = np.asarray(u_contact, dtype=float)
        du = np.asarray(du_contact, dtype=float)
        v0_arr = np.asarray(v0, dtype=float)

        # Penetration magnitude (δ = -u for u < 0)
        delta = -u
        R = np.zeros_like(u)

        # Only compute forces where there's penetration
        mask = delta > 0.0
        if not np.any(mask):
            return R

        d = delta[mask]
        dv = du[mask]
        v0m = v0_arr[mask]

        # Prevent division by zero in damping terms
        v0m = np.where(
            np.abs(v0m) < 1e-8,
            np.sign(v0m) * 1e-8 + (v0m == 0) * 1e-8,
            v0m,
        )

        model_lower = model.lower()

        if contact_law is not None or model_lower == "tabulated":
            if contact_law is None:
                raise ValueError("contact_law must be provided when using model 'tabulated'")
            R_tab = np.zeros_like(u)
            for idx, dval in zip(np.where(mask)[0], d):
                R_tab[idx] = -contact_law.evaluate(float(dval))
            return R_tab

        # Get model function (default to anagnostopoulos if not found)
        model_func = ContactModels.MODELS.get(
            model_lower,
            ContactModels.MODELS["anagnostopoulos"],
        )

        # Raw model force: negative = compression, positive = tension
        R_raw = model_func(k_wall, d, cr_wall, dv, v0m)

        # Enforce unilateral contact: no tension allowed
        R_comp = np.minimum(R_raw, 0.0)

        R[mask] = R_comp
        return R

    @staticmethod
    def list_models() -> list[str]:
        """Return list of available contact model names.

        Returns
        -------
        list[str]
            Sorted list of contact model names that can be passed to
            the 'model' parameter of compute_force().
        """
        return sorted(list(ContactModels.MODELS.keys()) + ["tabulated"])

    @staticmethod
    def get_model_info(model: str) -> str:
        """Get information about a specific contact model.

        Parameters
        ----------
        model : str
            Name of the contact model

        Returns
        -------
        str
            Description of the model including its mathematical form
        """
        model_lower = model.lower()
        if model_lower == "tabulated":
            return "Tabulated force-displacement law (requires contact_law input)"
        if model_lower not in ContactModels.MODELS:
            return f"Unknown model: {model}"

        descriptions = {
            "hooke": "Linear elastic contact: F = -k*δ",
            "hertz": "Hertzian contact: F = -k*δ^1.5",
            "hunt-crossley": "Hunt-Crossley: F = -k*δ^1.5*[1 + (3/2)(1-cr)(δ̇/v₀)]",
            "lankarani-nikravesh": "Lankarani-Nikravesh: F = -k*δ^1.5*[1 + (3/4)(1-cr²)(δ̇/v₀)]",
            "flores": "Flores et al.: F = -k*δ^1.5*[1 + (8/5cr)(1-cr)(δ̇/v₀)]",
            "gonthier": "Gonthier et al.: F = -k*δ^1.5*[1 + (1-cr²)/cr*(δ̇/v₀)]",
            "ye": "Ye & Zhu: F = -k*δ*[1 + (3/2cr)(1-cr)(δ̇/v₀)]",
            "pant-wijeyewickrema": "Pant & Wijeyewickrema: F = -k*δ*[1 + (3/2cr²)(1-cr²)(δ̇/v₀)]",
            "anagnostopoulos": "Anagnostopoulos: F = -k*δ*[1 + (3/2cr)(1-cr)(δ̇/v₀)]",
        }

        return descriptions.get(model_lower, f"No description available for {model}")
