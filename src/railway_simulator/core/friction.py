"""Friction models for railway impact simulation.

This module contains implementations of various friction models used in
the railway impact simulator, including LuGre, Dahl, Coulomb-Stribeck,
and Brown-McPhee models.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


class SimulationConstants:
    """Physical and numerical constants for the railway impact simulator.

    This is a minimal copy for friction module dependencies.
    The canonical version is in engine.py.
    """
    ZERO_TOL = 1e-12  # General zero tolerance for numerical comparisons


class FrictionModels:
    """Collection of friction model implementations.

    This class provides static methods for computing friction forces using
    various friction models commonly used in structural dynamics and impact
    simulations.

    Available Models
    ----------------
    - **LuGre**: Dynamic friction model with internal bristle state
    - **Dahl**: Simplified dynamic friction model
    - **Coulomb-Stribeck**: Classical friction with Stribeck effect
    - **Brown-McPhee**: Smooth approximation of Coulomb friction

    Examples
    --------
    >>> # Coulomb-Stribeck friction
    >>> F = FrictionModels.coulomb_stribeck(
    ...     v=0.5,           # velocity (m/s)
    ...     Fc=100.0,        # Coulomb force (N)
    ...     Fs=120.0,        # Static force (N)
    ...     vs=0.01,         # Stribeck velocity (m/s)
    ...     Fv=10.0          # Viscous coefficient (N·s/m)
    ... )

    >>> # LuGre friction (requires internal state)
    >>> F, z_new = FrictionModels.lugre(
    ...     z_prev=0.0,      # previous bristle state
    ...     v=0.5,           # velocity (m/s)
    ...     F_coulomb=100.0, # Coulomb force (N)
    ...     F_static=120.0,  # Static force (N)
    ...     v_stribeck=0.01, # Stribeck velocity (m/s)
    ...     sigma_0=1e5,     # Bristle stiffness (N/m)
    ...     sigma_1=1e3,     # Bristle damping (N·s/m)
    ...     sigma_2=10.0,    # Viscous damping (N·s/m)
    ...     h=1e-4           # time step (s)
    ... )
    """

    @staticmethod
    def lugre(
        z_prev: float,
        v: float,
        F_coulomb: float,
        F_static: float,
        v_stribeck: float,
        sigma_0: float,
        sigma_1: float,
        sigma_2: float,
        h: float,
    ) -> Tuple[float, float]:
        """LuGre friction model with stable time integration.

        The LuGre model includes an internal bristle state z that evolves
        according to:

            z_dot = v - |v| * z / g(v)

        where g(v) is the steady-state bristle deflection that depends on
        the Stribeck velocity and static/Coulomb friction forces.

        Notes
        -----
        The internal bristle state ODE can become stiff when g(v) is small
        and/or when the time step is large. A naive explicit Euler update
        makes friction sweeps artificially sensitive to Δt and may create
        non-physical outliers.

        This implementation uses the closed-form solution for constant (v, g)
        over one step, which is unconditionally stable:

            z_{n+1} = z_n * exp(-k*h) + g*sign(v) * (1 - exp(-k*h))

        where k = |v| / g, then evaluates z_dot at the end of the step for
        the damping term.

        Parameters
        ----------
        z_prev : float
            Previous bristle state (dimensionless displacement)
        v : float
            Current velocity (m/s)
        F_coulomb : float
            Coulomb friction force (N)
        F_static : float
            Static friction force (N)
        v_stribeck : float
            Stribeck velocity parameter (m/s)
        sigma_0 : float
            Bristle stiffness (N/m)
        sigma_1 : float
            Bristle damping coefficient (N·s/m)
        sigma_2 : float
            Viscous friction coefficient (N·s/m)
        h : float
            Time step size (s)

        Returns
        -------
        F : float
            Friction force (N)
        z : float
            Updated bristle state (dimensionless)

        References
        ----------
        .. [1] Canudas de Wit, C., et al. "A new model for control of systems
               with friction." IEEE TAC 40.3 (1995): 419-425.
        """
        v_stribeck = max(abs(v_stribeck), 1e-10)

        # g(v) is the steady-state bristle deflection (units of length)
        g = (F_coulomb + (F_static - F_coulomb) * np.exp(-(v / v_stribeck) ** 2)) / max(
            abs(sigma_0), 1e-12
        )
        g = max(abs(g), 1e-12)

        v_abs = float(np.abs(v))
        if v_abs < 1e-12:
            # No slip: keep bristle state, no velocity-dependent terms
            z = z_prev
            z_dot = 0.0
        else:
            k = v_abs / g
            # Stable closed-form update for the linear ODE
            exp_kh = float(np.exp(-k * h))
            z = float(z_prev) * exp_kh + float(np.sign(v)) * g * (1.0 - exp_kh)
            # Evaluate z_dot at the end of the step for sigma_1 term
            z_dot = float(v) - v_abs * z / g

        F = float(sigma_0) * z + float(sigma_1) * z_dot + float(sigma_2) * float(v)
        return F, z

    @staticmethod
    def dahl(
        z_prev: float, v: float, F_coulomb: float, sigma_0: float, h: float
    ) -> Tuple[float, float]:
        """Dahl friction model with stable time integration.

        The Dahl model is a simplified dynamic friction model with internal
        state z that evolves according to:

            z_dot = v - (sigma_0 * |v| / F_c) * z

        This implementation uses a stable closed-form update for constant
        velocity sign over one time step.

        Parameters
        ----------
        z_prev : float
            Previous internal state
        v : float
            Current velocity (m/s)
        F_coulomb : float
            Coulomb friction force (N)
        sigma_0 : float
            Stiffness parameter (N/m)
        h : float
            Time step size (s)

        Returns
        -------
        F : float
            Friction force (N)
        z : float
            Updated internal state

        References
        ----------
        .. [1] Dahl, P. R. "Solid friction damping of mechanical vibrations."
               AIAA journal 14.12 (1976): 1675-1682.
        """
        Fc = max(abs(F_coulomb), SimulationConstants.ZERO_TOL)
        v_abs = float(np.abs(v))

        if v_abs < 1e-12:
            z = z_prev
        else:
            # z_dot = v - (sigma_0 * |v| / Fc) * z
            k = float(sigma_0) * v_abs / float(Fc)
            exp_kh = float(np.exp(-k * h))
            z_ss = float(Fc) / max(float(sigma_0), 1e-12) * float(np.sign(v))
            z = float(z_prev) * exp_kh + z_ss * (1.0 - exp_kh)

        F = float(sigma_0) * float(z)
        return F, float(z)

    @staticmethod
    def coulomb_stribeck(v: float, Fc: float, Fs: float, vs: float, Fv: float) -> float:
        """Coulomb + Stribeck + viscous friction model.

        Combines Coulomb friction, Stribeck effect (velocity-dependent
        transition between static and kinetic friction), and viscous damping.

        Parameters
        ----------
        v : float
            Velocity (m/s)
        Fc : float
            Coulomb (kinetic) friction force (N)
        Fs : float
            Static friction force (N)
        vs : float
            Stribeck velocity parameter (m/s)
        Fv : float
            Viscous friction coefficient (N·s/m)

        Returns
        -------
        float
            Friction force (N)

        Notes
        -----
        The friction force is computed as:

            F = [Fc + (Fs - Fc) * exp(-(v/vs)^2)] * sign(v) + Fv * v
        """
        vs = max(vs, 1e-6)
        v_abs = np.abs(v)
        return (Fc + (Fs - Fc) * np.exp(-((v_abs / vs) ** 2))) * np.sign(v) + Fv * v

    @staticmethod
    def brown_mcphee(v: float, Fc: float, Fs: float, vs: float) -> float:
        """Brown & McPhee friction model.

        A smooth approximation of Coulomb-Stribeck friction using hyperbolic
        tangent and rational functions, designed to avoid discontinuities at
        zero velocity.

        Parameters
        ----------
        v : float
            Velocity (m/s)
        Fc : float
            Coulomb friction force (N)
        Fs : float
            Static friction force (N)
        vs : float
            Stribeck velocity parameter (m/s)

        Returns
        -------
        float
            Friction force (N)

        References
        ----------
        .. [1] Brown, P., and McPhee, J. "A continuous velocity-based friction
               model for dynamics and control with physically meaningful
               parameters." Journal of Computational and Nonlinear Dynamics,
               11.5 (2016): 054502.
        """
        vs = max(vs, 1e-6)
        v_abs = np.abs(v)
        x = v_abs / vs

        term1 = Fc * np.tanh(4.0 * x)
        term2 = (Fs - Fc) * x / ((0.25 * x**2 + 0.75) ** 2)

        return (term1 + term2) * np.sign(v)
