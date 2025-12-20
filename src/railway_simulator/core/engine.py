"""Engine for the Railway Impact Simulator.

This module is UI-agnostic: it contains the HHT-α time integration,
Bouc–Wen hysteresis, friction and contact laws, and the main
ImpactSimulator class.

Nonlinear solver
----------------
The engine supports two implicit nonlinear solvers on top of HHT-α:

- ``solver='newton'``: Newton–Raphson on the displacement unknown ``q_{n+1}``
  (residual assembled consistently with the current HHT-α equilibrium form;
  Jacobian by finite differences for robustness across Bouc–Wen, contact,
  friction and mass-contact).
- ``solver='picard'``: legacy fixed-point iteration in acceleration.

Use from CLI, Streamlit app or tests as:

    from railway_simulator.core.engine import SimulationParams, run_simulation

    sim_params = SimulationParams(**params_dict)
    df = run_simulation(sim_params)

Set ``SimulationParams.solver`` (or the YAML key ``solver``) to switch.
"""


from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Tuple, Dict, Any
import logging
import numpy as np
import pandas as pd
from scipy.constants import g as GRAVITY

logger = logging.getLogger(__name__)


# ====================================================================
# SIMULATION CONSTANTS
# ====================================================================

class SimulationConstants:
    """Physical and numerical constants for the railway impact simulator.

    Centralizes magic numbers to improve maintainability and make
    parameter tuning more explicit.
    """

    # Contact/collision parameters
    MASS_CONTACT_STIFFNESS = 1e8  # N/m - Stiffness for mass-to-mass contact
    MASS_CONTACT_DAMPING = 1e5    # N·s/m - Damping for mass-to-mass contact
    MIN_SPRING_LENGTH_FRACTION = 0.05  # Minimum spring length as fraction of initial

    # Friction parameters
    STRIBECK_VELOCITY = 1.0e-3  # m/s - Reference velocity for Stribeck friction
    MIN_VELOCITY_THRESHOLD = 1e-8  # m/s - Minimum velocity for numerical stability

    # Numerical tolerances
    ZERO_TOL = 1e-12  # General zero tolerance for numerical comparisons
    BW_Z_CLIP = 10.0  # Dimensionless z(t) clip for Bouc–Wen stability (only affects unstable parameter sets)


    # Newton-Raphson parameters
    FD_EPSILON = 1e-6  # Finite difference epsilon for Jacobian
    MAX_LINE_SEARCH_ITERS = 8  # Maximum backtracking iterations
    ARMIJO_COEFF = 1e-4  # Armijo condition coefficient

    # Default building damping ratio
    DEFAULT_BUILDING_ZETA = 0.05  # 5% critical damping for building SDOF


# ====================================================================
# STRAIN-RATE METRICS
# ====================================================================

def strain_rate_metrics(
    df: pd.DataFrame,
    t_col: str = "Time_s",
    penetration_col: str = "Penetration_mm",
    penetration_units: str = "mm",
    L_ref_m: float = 1.0,
    contact_force_col: str | None = "Impact_Force_MN",
    force_threshold: float = 0.001,  # MN
    pen_threshold: float = 1e-9,  # m
    smooth_window: int = 5,
) -> Dict[str, float]:
    """
    Compute strain-rate proxy metrics from penetration time history.

    Strain rate proxy: ε̇(t) ≈ δ̇(t) / L_ref
    where δ(t) is penetration and L_ref is a characteristic length.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe with time and penetration columns.
    t_col : str
        Column name for time (default: "Time_s").
    penetration_col : str
        Column name for penetration (default: "Penetration_mm").
    penetration_units : str
        Units of penetration: "m" or "mm" (default: "mm").
    L_ref_m : float
        Characteristic length in meters (default: 1.0 m).
        Physical choices: wall thickness, crush zone length, buffer stroke.
    contact_force_col : str | None
        Optional force column to define contact window (default: "Impact_Force_MN").
    force_threshold : float
        Force threshold for contact detection in MN (default: 0.001 MN = 1 kN).
    pen_threshold : float
        Penetration threshold for contact detection in meters (default: 1e-9 m).
    smooth_window : int
        Smoothing window size for penetration (odd int, default: 5).
        Set to 1 to disable smoothing.

    Returns
    -------
    dict
        Dictionary with keys:
        - "strain_rate_peak_1_s": Peak strain rate (1/s)
        - "strain_rate_rms_1_s": RMS strain rate (1/s)
        - "strain_rate_p95_1_s": 95th percentile strain rate (1/s)
    """
    t = df[t_col].to_numpy(dtype=float)
    pen = df[penetration_col].to_numpy(dtype=float)

    # Convert to meters
    if penetration_units == "mm":
        pen = pen * 1e-3

    # Clamp to avoid tiny negative noise
    pen = np.maximum(pen, 0.0)

    # Light smoothing (helps derivative noise)
    if smooth_window and smooth_window > 1:
        w = int(smooth_window)
        if w % 2 == 0:
            w += 1
        k = np.ones(w) / w
        pen_s = np.convolve(pen, k, mode="same")
    else:
        pen_s = pen

    # Derivative (central diff via gradient)
    pen_dot = np.gradient(pen_s, t)  # m/s
    eps_dot = pen_dot / float(L_ref_m)  # 1/s

    # Contact window: penetration > threshold OR force > threshold
    mask = pen_s > float(pen_threshold)
    if contact_force_col is not None and contact_force_col in df.columns:
        f = df[contact_force_col].to_numpy(dtype=float)
        mask = mask | (np.abs(f) > float(force_threshold))

    if not np.any(mask):
        return {
            "strain_rate_peak_1_s": 0.0,
            "strain_rate_rms_1_s": 0.0,
            "strain_rate_p95_1_s": 0.0,
        }

    x = np.abs(eps_dot[mask])
    return {
        "strain_rate_peak_1_s": float(np.max(x)),
        "strain_rate_rms_1_s": float(np.sqrt(np.mean(x**2))),
        "strain_rate_p95_1_s": float(np.percentile(x, 95)),
    }


# ====================================================================
# CONFIGURATION & DATA CLASSES
# ====================================================================

@dataclass
class SimulationParams:
    """Container for all simulation parameters."""
    # Geometry
    n_masses: int
    masses: np.ndarray
    x_init: np.ndarray
    y_init: np.ndarray

    # Kinematics
    v0_init: float
    angle_rad: float
    d0: float  # Initial distance to wall

    # Material properties
    fy: np.ndarray
    uy: np.ndarray

    # Contact
    k_wall: float
    cr_wall: float
    contact_model: str

    # Building SDOF (still stored here, but *used* in app/postprocessing)
    building_enable: bool
    building_mass: float
    building_zeta: float
    building_height: float
    building_model: str
    building_uy: float
    building_uy_mm: float
    building_alpha: float
    building_gamma: float

    # Friction
    mu_s: float
    mu_k: float
    sigma_0: float
    sigma_1: float
    sigma_2: float
    friction_model: str

    # Bouc-Wen
    bw_a: float
    bw_A: float
    bw_beta: float
    bw_gamma: float
    bw_n: int

    # Integration
    alpha_hht: float
    newton_tol: float
    max_iter: int
    h_init: float  # Time step
    T_max: float   # Maximum simulation time
    step: int      # Number of steps (T_max / h_init)
    T_int: Tuple[float, float]  # Time interval (0, T_max)

    # Optional metadata / consistency check: linear stiffness between masses
    k_train: np.ndarray | None = None

    # Strain-rate analysis
    L_ref_m: float = 1.0  # Characteristic length for strain-rate computation (m)

    # Nonlinear solver selection
    solver: str = "newton"  # "newton" (Newton–Raphson) or "picard" (legacy)
    newton_jacobian_mode: str = "per_step"  # "per_step" (fast) or "each_iter" (pure/slow)

@dataclass
class TrainConfig:
    """Train configuration parameters (helper for building mass/x_init)."""
    n_wagons: int
    mass_lok_t: float
    mass_wagon_t: float
    L_lok: float
    L_wagon: float
    mass_points_lok: int
    mass_points_wagon: int
    gap: float


# ====================================================================
# BOUC-WEN HYSTERESIS MODEL
# ====================================================================

class BoucWenModel:
    """Bouc-Wen hysteretic model implementation."""

    @staticmethod
    def evolution_rate(x: float, u: float, A: float, beta: float,
                       gamma: float, n: int, uy: float) -> float:
        """
        Calculate Bouc-Wen evolution rate.

        Args:
            x: Current hysteretic state (dimensionless)
            u: Velocity input
            A, beta, gamma, n: Bouc-Wen parameters
            uy: Yield deformation
        """
        if abs(uy) < SimulationConstants.ZERO_TOL:
            return 0.0
        return (A - np.abs(x) ** n * (beta + np.sign(u * x) * gamma)) * u / uy

    @staticmethod
    def integrate_rk4(x0: float, u: float, h: float, A: float,
                      beta: float, gamma: float, n: int, uy: float) -> float:
        """4th-order Runge-Kutta integration for Bouc-Wen model."""
        evolve = lambda x_val: BoucWenModel.evolution_rate(
            x_val, u, A, beta, gamma, n, uy
        )

        k1 = evolve(x0)
        k2 = evolve(x0 + 0.5 * h * k1)
        k3 = evolve(x0 + 0.5 * h * k2)
        k4 = evolve(x0 + h * k3)

        x1 = x0 + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Numerical safeguard: allow exploration of "wild" parameter sets (incl. negative β/γ)
        # without NaNs/Inf breaking the solver. Stable calibrated cases are unaffected.
        if not np.isfinite(x1):
            x1 = x0
        x1 = float(np.clip(x1, -SimulationConstants.BW_Z_CLIP, SimulationConstants.BW_Z_CLIP))
        return x1

    @staticmethod
    def compute_forces(u: np.ndarray, du: np.ndarray, x: np.ndarray,
                       uy: np.ndarray, fy: np.ndarray, h: float,
                       a: float, A: float, beta: float, gamma: float,
                       n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute nodal forces using Bouc-Wen model.

        Returns:
            R: Nodal force vector (length n_springs+1)
            xfunc: Updated hysteretic state array
        """
        u = np.asarray(u, dtype=float)
        du = np.asarray(du, dtype=float)
        x = np.asarray(x, dtype=float)
        uy = np.asarray(uy, dtype=float)
        fy = np.asarray(fy, dtype=float)

        n_springs = len(u)
        xfunc = np.zeros_like(x)
        R = np.zeros(n_springs + 1)

        for i in range(n_springs):
            # Update hysteretic state
            xfunc[i] = BoucWenModel.integrate_rk4(
                x[i], du[i], h, A, beta, gamma, n, uy[i]
            )

            # Compute spring force (elastic + hysteretic)
            f_spring = (a * (fy[i] / uy[i]) * u[i] +
                        (1.0 - a) * fy[i] * xfunc[i])

            # Distribute to nodes
            if i == 0:
                R[0] = -f_spring
            else:
                R[i] = R[i] - f_spring

            if i < n_springs:
                R[i + 1] = R[i + 1] + f_spring

        return R, xfunc


# ====================================================================
# FRICTION MODELS
# ====================================================================

class FrictionModels:
    """Collection of friction model implementations."""

    @staticmethod
    def lugre(z_prev: float, v: float, F_coulomb: float, F_static: float,
              v_stribeck: float, sigma_0: float, sigma_1: float,
              sigma_2: float, h: float) -> Tuple[float, float]:
        """LuGre friction model.

        Notes
        -----
        The internal bristle state ODE

            z_dot = v - |v| * z / g(v)

        can become stiff when g(v) is small and/or when the time step is large.
        A naive explicit Euler update makes friction sweeps artificially sensitive
        to Δt and may create non-physical outliers (e.g. one μ case behaving very
        differently).

        Here we use the closed-form solution for constant (v, g) over one step,
        which is unconditionally stable:

            z_{n+1} = z_n e^{-k h} + g sign(v) (1 - e^{-k h}),   k = |v| / g

        and then evaluate z_dot at the end of the step for the damping term.
        """
        v_stribeck = max(abs(v_stribeck), 1e-10)
        # g(v) is the steady-state bristle deflection (units of length)
        g = (F_coulomb + (F_static - F_coulomb) * np.exp(-(v / v_stribeck) ** 2)) / max(abs(sigma_0), 1e-12)
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
    def dahl(z_prev: float, v: float, F_coulomb: float,
             sigma_0: float, h: float) -> Tuple[float, float]:
        """Dahl friction model.

        Uses a stable closed-form update of the internal state for constant
        velocity sign over one time step (same motivation as LuGre).
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
    def coulomb_stribeck(v: float, Fc: float, Fs: float,
                         vs: float, Fv: float) -> float:
        """Coulomb + Stribeck + viscous friction."""
        vs = max(vs, 1e-6)
        v_abs = np.abs(v)
        return (Fc + (Fs - Fc) * np.exp(-(v_abs / vs) ** 2)) * np.sign(v) + Fv * v

    @staticmethod
    def brown_mcphee(v: float, Fc: float, Fs: float, vs: float) -> float:
        """Brown & McPhee friction model."""
        vs = max(vs, 1e-6)
        v_abs = np.abs(v)
        x = v_abs / vs

        term1 = Fc * np.tanh(4.0 * x)
        term2 = (Fs - Fc) * x / ((0.25 * x ** 2 + 0.75) ** 2)

        return (term1 + term2) * np.sign(v)


# ====================================================================
# CONTACT MODELS
# ====================================================================

class ContactModels:
    """Normal contact force model implementations."""

    MODELS = {
        "hooke": lambda k, d, cr, dv, v0: -k * d,
        "hertz": lambda k, d, cr, dv, v0: -k * d ** 1.5,
        "hunt-crossley": lambda k, d, cr, dv, v0: (
            -k * d ** 1.5 * (1.0 + 3.0 * (1.0 - cr) / 2.0 * (dv / v0))
        ),
        "lankarani-nikravesh": lambda k, d, cr, dv, v0: (
            -k * d ** 1.5 * (1.0 + 3.0 * (1.0 - cr ** 2) / 4.0 * (dv / v0))
        ),
        "flores": lambda k, d, cr, dv, v0: (
            -k * d ** 1.5 * (1.0 + 8.0 * (1.0 - cr) / (5.0 * cr) * (dv / v0))
        ),
        "gonthier": lambda k, d, cr, dv, v0: (
            -k * d ** 1.5 * (1.0 + (1.0 - cr ** 2) / cr * (dv / v0))
        ),
        "ye": lambda k, d, cr, dv, v0: (
            -k * d * (1.0 + 3.0 * (1.0 - cr) / (2.0 * cr) * (dv / v0))
        ),
        "pant-wijeyewickrema": lambda k, d, cr, dv, v0: (
            -k * d * (1.0 + 3.0 * (1.0 - cr ** 2) / (2.0 * cr ** 2) * (dv / v0))
        ),
        "anagnostopoulos": lambda k, d, cr, dv, v0: (
            -k * d * (1.0 + 3.0 * (1.0 - cr) / (2.0 * cr) * (dv / v0))
        ),
    }

    @staticmethod
    def compute_force(u_contact: np.ndarray, du_contact: np.ndarray,
                      v0: np.ndarray, k_wall: float, cr_wall: float,
                      model: str) -> np.ndarray:
        """
        Compute normal contact forces (unilateral: compression only).

        Args:
            u_contact: Penetration (negative when in contact)
            du_contact: Penetration velocity
            v0: Initial contact velocity
            k_wall: Wall stiffness
            cr_wall: Coefficient of restitution
            model: Contact model name
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

        # Prevent division by zero
        v0m = np.where(
            np.abs(v0m) < 1e-8,
            np.sign(v0m) * 1e-8 + (v0m == 0) * 1e-8,
            v0m,
        )

        # Get model function (default to anagnostopoulos)
        model_func = ContactModels.MODELS.get(
            model.lower(),
            ContactModels.MODELS["anagnostopoulos"],
        )

        # Raw model force: negative = compression, positive = tension
        R_raw = model_func(k_wall, d, cr_wall, dv, v0m)

        # Enforce unilateral contact: no tension allowed
        R_comp = np.minimum(R_raw, 0.0)

        R[mask] = R_comp
        return R


# ====================================================================
# TRAIN GEOMETRY BUILDERS
# ====================================================================

class TrainBuilder:
    """Utilities for building train geometries."""

    @staticmethod
    def distribute_masses(total_mass: float, n_points: int) -> np.ndarray:
        """
        Distribute mass across points.

        - 2 points: 50/50
        - 3 points: 25/50/25
        - else: uniform
        """
        if n_points == 3:
            fractions = np.array([0.25, 0.5, 0.25])
        elif n_points == 2:
            fractions = np.array([0.5, 0.5])
        else:
            fractions = np.ones(n_points) / n_points

        return total_mass * fractions

    @staticmethod
    def build_train(config: TrainConfig) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a locomotive + wagons train configuration.

        Returns:
            n_masses: Total number of mass points
            masses: Mass array [kg]
            x_init: Initial x-coordinates [m]
            y_init: Initial y-coordinates [m]
        """
        mass_lok = config.mass_lok_t * 1000.0
        mass_wagon = config.mass_wagon_t * 1000.0

        n_masses = (
            config.mass_points_lok +
            config.n_wagons * config.mass_points_wagon
        )

        masses = np.zeros(n_masses)
        x_init = np.zeros(n_masses)
        y_init = np.zeros(n_masses)

        x_front = 0.02  # 2 cm in front of wall
        idx = 0

        # Locomotive
        m_lok = TrainBuilder.distribute_masses(mass_lok, config.mass_points_lok)
        for j in range(config.mass_points_lok):
            masses[idx] = m_lok[j]
            if config.mass_points_lok == 1:
                x_init[idx] = x_front
            else:
                x_init[idx] = x_front + config.L_lok * (j / (config.mass_points_lok - 1))
            idx += 1

        x_front += config.L_lok + config.gap

        # Wagons
        for _ in range(config.n_wagons):
            m_wag = TrainBuilder.distribute_masses(mass_wagon, config.mass_points_wagon)
            for j in range(config.mass_points_wagon):
                masses[idx] = m_wag[j]
                if config.mass_points_wagon == 1:
                    x_init[idx] = x_front
                else:
                    x_init[idx] = x_front + config.L_wagon * (j / (config.mass_points_wagon - 1))
                idx += 1
            x_front += config.L_wagon + config.gap

        return n_masses, masses, x_init, y_init


# ====================================================================
# STRUCTURAL DYNAMICS
# ====================================================================

class StructuralDynamics:
    """Structural analysis utilities."""

    @staticmethod
    def build_stiffness_matrix_2d(
        n: int, x: np.ndarray, y: np.ndarray, k: np.ndarray
    ) -> np.ndarray:
        """
        Build 2D bar-element stiffness matrix.

        Args:
            n: Number of nodes
            x, y: Node coordinates
            k: Element stiffness values (length n-1)
        """
        dof = 2 * n
        K = np.zeros((dof, dof))

        for i in range(n - 1):
            j = i + 1

            dx = x[j] - x[i]
            dy = y[j] - y[i]
            L0 = np.hypot(dx, dy)

            if L0 < SimulationConstants.ZERO_TOL:
                continue

            cx = dx / L0
            cy = dy / L0
            k_elem = k[i]

            # Element stiffness matrix
            ke = k_elem * np.array([
                [cx * cx,  cx * cy, -cx * cx, -cx * cy],
                [cx * cy,  cy * cy, -cx * cy, -cy * cy],
                [-cx * cx, -cx * cy,  cx * cx,  cx * cy],
                [-cx * cy, -cy * cy,  cx * cy,  cy * cy]
            ])

            # Assemble into global matrix
            dofs = [i, n + i, j, n + j]
            for a in range(4):
                for b in range(4):
                    K[dofs[a], dofs[b]] += ke[a, b]

        return K

    @staticmethod
    def compute_rayleigh_damping(M: np.ndarray, K: np.ndarray,
                                 zeta: float = 0.05) -> np.ndarray:
        """
        Compute Rayleigh damping matrix using highest two frequencies.

        C = α*M + β*K
        """
        eigenvalues, _ = np.linalg.eig(np.linalg.solve(M, K))
        positive = np.real(eigenvalues[np.real(eigenvalues) > SimulationConstants.ZERO_TOL])
        freqs = np.sqrt(np.abs(positive))
        freqs = np.real(freqs)
        freqs.sort()

        if len(freqs) >= 2:
            wn1, wn2 = freqs[-2], freqs[-1]
            alpha_r = 2 * zeta * wn1 * wn2 / (wn1 + wn2)
            beta_r = 2 * zeta / (wn1 + wn2)
        elif len(freqs) == 1:
            wn1 = freqs[0]
            alpha_r = 2 * zeta * wn1
            beta_r = 0.0
        else:
            alpha_r = 0.0
            beta_r = 0.0

        return M * alpha_r + K * beta_r


# ====================================================================
# TIME INTEGRATION (HHT-α)
# ====================================================================

class HHTAlphaIntegrator:
    """HHT-α implicit time integration."""

    def __init__(self, alpha: float):
        """
        Initialize HHT-α integrator.

        Args:
            alpha: HHT parameter (typically -0.3 to 0.0)
        """
        self.alpha = alpha
        self.beta = 0.25 * (1.0 + alpha) ** 2
        self.gamma = 0.5 + alpha

        # Count of linear solves (LU) performed in this run
        self.n_lu: int = 0


    def predict(self, q: np.ndarray, qp: np.ndarray, qpp: np.ndarray,
                qpp_new: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Newmark-like predictor / corrector step.

        Returns:
            q_new: Predicted or corrected displacement
            qp_new: Predicted or corrected velocity
        """
        q_new = (
            q + h * qp +
            (0.5 - self.beta) * h ** 2 * qpp +
            self.beta * h ** 2 * qpp_new
        )

        qp_new = (
            qp +
            (1.0 - self.gamma) * h * qpp +
            self.gamma * h * qpp_new
        )

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
        qp_old: np.ndarray
    ) -> np.ndarray:
        """
        Compute acceleration using HHT-α method.

        M * a_new = (1-α)*R_new + α*R_old - (1-α)*C*v_new - α*C*v_old
        """
        R_total_new = R_internal + R_contact + R_friction + R_mass_contact
        R_total_old = R_internal_old + R_contact_old + R_friction_old + R_mass_contact_old

        force = (
            (1.0 - self.alpha) * R_total_new +
            self.alpha * R_total_old -
            (1.0 - self.alpha) * (C @ qp) -
            self.alpha * (C @ qp_old)
        )

        # Count this linear solve (dense LU)
        self.n_lu += 1
        return np.linalg.solve(M, force)


# ====================================================================
# MAIN SIMULATION ENGINE
# ====================================================================

class ImpactSimulator:
    """Main HHT-α simulation engine.

    Solver selection is controlled by ``SimulationParams.solver``:

    - ``"newton"``: Newton–Raphson on the displacement unknown ``q_{n+1}``
      using a finite-difference Jacobian of the *discrete* HHT-α residual.
      This is robust across Bouc–Wen, contact, friction, and mass-contact.
    - ``"picard"``: legacy fixed-point iteration in acceleration.
    """

    def __init__(self, params: SimulationParams):
        self.params = params

        # Performance counters
        self.linear_solves: int = 0
        self.total_iters: int = 0  # renamed (no "newton" in the name)
        self.max_iters_per_step: int = 0  # max iterations needed for any timestep
        self.max_residual: float = 0.0  # max residual seen across all iterations

        self.setup()

    # ----------------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------------
    def setup(self):
        """Initialize simulation matrices and state."""
        p = self.params

        # ------------------------------------------------------------------
        # Normalise core arrays and scalar types
        # ------------------------------------------------------------------
        p.masses = np.asarray(p.masses, dtype=float)
        p.x_init = np.asarray(p.x_init, dtype=float)
        p.y_init = np.asarray(p.y_init, dtype=float)
        p.fy = np.asarray(p.fy, dtype=float)
        p.uy = np.asarray(p.uy, dtype=float)

        p.n_masses = int(p.n_masses)
        p.step = int(p.step)
        p.h_init = float(p.h_init)
        p.T_max = float(p.T_max)

        # Time discretization
        self.h = p.h_init
        self.t = p.T_int[0] + self.h * np.arange(p.step + 1)

        # Rotation matrix for impact angle
        c = np.cos(p.angle_rad)
        s = np.sin(p.angle_rad)
        self.rot = np.array([[c, -s], [s, c]])

        # Apply initial distance to wall (shift all x-coordinates)
        x_init_shifted = p.x_init + p.d0

        # Transform initial conditions with shifted positions
        xy_init = self.rot @ np.vstack([x_init_shifted, p.y_init])
        self.x_init = xy_init[0, :]
        self.y_init = xy_init[1, :]

        xp_init = np.full(p.n_masses, p.v0_init, dtype=float)
        yp_init = np.zeros(p.n_masses, dtype=float)
        xpyp_init = self.rot @ np.vstack([xp_init, yp_init])
        self.xp_init = xpyp_init[0, :]
        self.yp_init = xpyp_init[1, :]

        # Mass matrix (2D DOFs, x and z/y)
        self.M = np.diag(np.concatenate([p.masses, p.masses]))

        # Stiffness matrix (use original unshifted geometry)
        k_init = p.fy / p.uy
        self.k_lin = k_init.copy()
        self.K = StructuralDynamics.build_stiffness_matrix_2d(
            p.n_masses, p.x_init, p.y_init, k_init
        )

        # Damping matrix (Rayleigh)
        self.C = StructuralDynamics.compute_rayleigh_damping(self.M, self.K)

        # Initial spring lengths (original unshifted geometry)
        self.u10 = np.zeros(p.n_masses - 1)
        for i in range(p.n_masses - 1):
            dx = p.x_init[i + 1] - p.x_init[i]
            dy = p.y_init[i + 1] - p.y_init[i]
            self.u10[i] = np.hypot(dx, dy)

        # Mass-to-mass contact tracking
        # Minimum allowed spring lengths (5% of initial length to prevent mass overlap)
        self.L_min = SimulationConstants.MIN_SPRING_LENGTH_FRACTION * self.u10
        self.mass_contact_active = np.zeros(p.n_masses - 1, dtype=bool)

        # HHT integrator
        self.integrator = HHTAlphaIntegrator(p.alpha_hht)

        # Pre-compute friction enablement (optimization)
        #
        # NOTE: In earlier versions we disabled friction when (sigma_0,sigma_1,sigma_2)==0.
        # That unintentionally disables *Coulomb/Stribeck* friction sweeps, because those
        # models do not require LuGre bristle parameters.
        #
        # Here we enable friction whenever:
        # - friction_model is not off/none
        # - and either mu is non-zero OR any sigma term is non-zero
        fm = (p.friction_model or "none").strip().lower()
        mu_zero = (abs(p.mu_s) < SimulationConstants.ZERO_TOL and abs(p.mu_k) < SimulationConstants.ZERO_TOL)
        sigma_all_zero = (
            abs(p.sigma_0) < SimulationConstants.ZERO_TOL
            and abs(p.sigma_1) < SimulationConstants.ZERO_TOL
            and abs(p.sigma_2) < SimulationConstants.ZERO_TOL
        )
        self.friction_enabled = not (fm in ("none", "off", "") or (mu_zero and sigma_all_zero))

        # Guard LuGre/Dahl against sigma_0 == 0 (division by zero / extremely slow convergence).
        # If the user selected LuGre/Dahl with mu != 0 but forgot sigma_0, pick a sensible default
        # so parametric sweeps actually show an effect without extra UI knobs.
        if self.friction_enabled and fm in ("lugre", "dahl") and abs(p.sigma_0) < SimulationConstants.ZERO_TOL and not mu_zero:
            p.sigma_0 = 1.0e6
            if fm == "lugre" and abs(p.sigma_1) < SimulationConstants.ZERO_TOL:
                p.sigma_1 = 1.0e3



    # ----------------------------------------------------------------
    # RUN
    # ----------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        Execute time-stepping simulation with full HHT-α fixed-point / Newton iteration.
        """
        p = self.params
        n = p.n_masses
        dof = 2 * n

        # State arrays
        q = np.zeros((dof, p.step + 1))
        qp = np.zeros((dof, p.step + 1))
        qpp = np.zeros((dof, p.step + 1))

        # Initial conditions
        q[:, 0] = np.concatenate([self.x_init, self.y_init])
        qp[:, 0] = np.concatenate([self.xp_init, self.yp_init])
        qpp[:, 0] = np.zeros(dof)

        # Forces & deformations
        u_spring = np.zeros((n - 1, p.step + 1))
        u_contact = np.zeros((dof, p.step + 1))
        R_contact = np.zeros((dof, p.step + 1))
        R_friction = np.zeros((dof, p.step + 1))
        R_internal = np.zeros((dof, p.step + 1))
        R_mass_contact = np.zeros((dof, p.step + 1))

        # Hysteretic and friction internal states
        X_bw = np.zeros((n - 1, p.step + 1))
        z_friction = np.zeros((n, p.step + 1))

        # Contact tracking
        contact_active = False
        v0_contact = np.ones(dof)

        # Normal forces for friction (per node)
        FN_node = GRAVITY * p.masses
        vs = SimulationConstants.STRIBECK_VELOCITY

        # --------------------------------------------------------------
        # Initial forces & acceleration (t0 consistency)
        # --------------------------------------------------------------
        # Populate R_internal[:,0], R_contact[:,0] and qpp[:,0] so the
        # first HHT/Newmark predictor uses a physically consistent a0.
        # This avoids a spurious one-step energy residual at step=1.

        # Spring deformation at t0 (relative to initial lengths)
        for i in range(n - 1):
            r1 = q[[i, n + i], 0]
            r2 = q[[i + 1, n + i + 1], 0]
            u_spring[i, 0] = np.linalg.norm(r2 - r1) - self.u10[i]
        du0 = np.zeros(n - 1)

        R_spring0, X_bw[:, 0] = BoucWenModel.compute_forces(
            -u_spring[:, 0],
            -du0,
            X_bw[:, 0],
            self.params.uy,
            self.params.fy,
            self.h,
            self.params.bw_a,
            self.params.bw_A,
            self.params.bw_beta,
            self.params.bw_gamma,
            self.params.bw_n,
        )
        R_internal[:, 0] = np.concatenate([R_spring0, np.zeros(n)])

        # Wall contact at t0 (if any penetration exists)
        u_contact[:, 0] = 0.0
        R_contact[:, 0] = 0.0
        if np.any(q[:n, 0] < 0.0):
            u_contact[:n, 0] = np.where(q[:n, 0] < 0.0, q[:n, 0], 0.0)
            du_contact0 = np.zeros(dof)
            # Initialize v0_contact for approaching x-DOFs that are in contact
            mask = (q[:n, 0] < 0.0) & (qp[:n, 0] < 0.0)
            if np.any(mask):
                contact_active = True
                v0_contact[:n] = np.where(mask, qp[:n, 0], 1.0)
                v0_contact[:n][v0_contact[:n] == 0.0] = 1.0
            R0 = ContactModels.compute_force(
                u_contact[:, 0],
                du_contact0,
                v0_contact,
                p.k_wall,
                p.cr_wall,
                p.contact_model,
            )
            R_contact[:, 0] = -R0

        # Initial acceleration from EOM: M a0 = R_total0 - C v0
        R_total0 = R_internal[:, 0] + R_contact[:, 0] + R_friction[:, 0] + R_mass_contact[:, 0]
        qpp[:, 0] = np.linalg.solve(self.M, R_total0 - (self.C @ qp[:, 0]))

        # Energy bookkeeping (Euler-Lagrange formulation)
        # Mechanical energy
        E_kin = np.zeros(p.step + 1)  # Kinetic energy T
        E_pot_spring = np.zeros(p.step + 1)  # Spring potential (conservative part)
        E_pot_contact = np.zeros(p.step + 1)  # Contact potential (elastic part)
        E_pot = np.zeros(p.step + 1)  # Total potential V = E_pot_spring + E_pot_contact
        E_mech = np.zeros(p.step + 1)  # Total mechanical E_mech = T + V

        # Work and dissipation (integrated from generalized forces)
        W_ext = np.zeros(p.step + 1)  # External work ∫ qdot^T Q_ext dt
        E_diss_rayleigh = np.zeros(p.step + 1)  # Rayleigh damping dissipation
        E_diss_bw = np.zeros(p.step + 1)  # Bouc-Wen hysteretic dissipation
        E_diss_softening = np.zeros(p.step + 1)  # Post-yield softening (a<0) treated as non-conservative work
        E_diss_contact_damp = np.zeros(p.step + 1)  # Contact damping dissipation
        E_diss_friction = np.zeros(p.step + 1)  # Friction dissipation
        E_diss_mass_contact = np.zeros(p.step + 1)  # Mass-to-mass contact dissipation
        E_diss_total = np.zeros(p.step + 1)  # Total dissipation

        # Numerical residual
        E_num = np.zeros(p.step + 1)  # E_num = E0 + W_ext - (E_mech + E_diss)
        E_num_ratio = np.zeros(p.step + 1)  # |E_num| / E0

        # Diagnostics (peak residual step)
        iters_hist = np.zeros(p.step, dtype=int)
        P_nc_hist = np.zeros(p.step)   # qdot_mid @ Q_nc_mid
        P_ext_hist = np.zeros(p.step)  # qdot_mid @ Q_ext_mid
        delta_hist = np.zeros(p.step + 1)
        delta_dot_hist = np.zeros(p.step)  # between n and n+1
        f_el_hist = np.zeros(p.step + 1)
        f_damp_hist = np.zeros(p.step + 1)

        # Initial conditions (all energy is kinetic, V(0) = 0)
        v0 = qp[:, 0]
        E_kin[0] = 0.5 * float(v0.T @ self.M @ v0)
        E_pot[0] = 0.0
        E_mech[0] = E_kin[0]
        E0 = E_mech[0]  # Initial total energy

        # Time stepping
        solver = str(getattr(p, "solver", "newton")).lower()
        for step_idx in range(p.step):

            converged = False
            err = np.inf
            iters_this_step = 0

            if solver in ("newton", "nr", "newton-raphson", "newton_raphson"):
                # =========================================================
                # Newton–Raphson on displacement q_{n+1}
                # =========================================================
                beta = self.integrator.beta
                gamma = self.integrator.gamma
                alpha = self.integrator.alpha
                h = self.h

                a0 = 1.0 / (beta * h * h)

                q_n = q[:, step_idx]
                v_n = qp[:, step_idx]
                a_n = qpp[:, step_idx]

                # Constant-acceleration predictor for q_{n+1}
                q_guess = q_n + h * v_n + 0.5 * h * h * a_n

                # Cached previous-step state (treated as constant in this step)
                u_s_old = u_spring[:, step_idx].copy()
                u_c_old = u_contact[:, step_idx].copy()
                X_old = X_bw[:, step_idx].copy()
                z_old = z_friction[:, step_idx].copy()

                R_total_old = (
                    R_internal[:, step_idx]
                    + R_contact[:, step_idx]
                    + R_friction[:, step_idx]
                    + R_mass_contact[:, step_idx]
                )
                Cv_old = self.C @ v_n

                contact_active_prev = bool(contact_active)
                v0_contact_prev = v0_contact.copy()

                def _av_from_q(q_trial: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                    """Compute (v_{n+1}, a_{n+1}) from trial q_{n+1} (Newmark inversion)."""
                    a_trial = a0 * (
                        q_trial
                        - q_n
                        - h * v_n
                        - h * h * (0.5 - beta) * a_n
                    )
                    v_trial = v_n + h * ((1.0 - gamma) * a_n + gamma * a_trial)
                    return v_trial, a_trial

                def _eval_state(q_trial: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
                    """Return (residual, state-dict) for Newton iterations."""
                    v_trial, a_trial = _av_from_q(q_trial)

                    # --- Springs (Bouc–Wen in 2D) ---
                    u_s_new = np.zeros_like(u_s_old)
                    R_int_new = np.zeros(dof)
                    X_new = np.zeros_like(X_old)

                    # Spring deformations
                    for i in range(n - 1):
                        r1 = q_trial[[i, n + i]]
                        r2 = q_trial[[i + 1, n + i + 1]]
                        u_s_new[i] = np.linalg.norm(r2 - r1) - self.u10[i]

                    du = (u_s_new - u_s_old) / h
                    u_comp = -u_s_new
                    du_comp = -du

                    for i in range(n - 1):
                        X_new[i] = BoucWenModel.integrate_rk4(
                            X_old[i],
                            du_comp[i],
                            h,
                            self.params.bw_A,
                            self.params.bw_beta,
                            self.params.bw_gamma,
                            self.params.bw_n,
                            self.params.uy[i],
                        )

                        f_spring = (
                            p.bw_a * (self.k_lin[i] * u_comp[i])
                            + (1.0 - p.bw_a) * p.fy[i] * X_new[i]
                        )

                        r1 = q_trial[[i, n + i]]
                        r2 = q_trial[[i + 1, n + i + 1]]
                        dr = r2 - r1
                        L = np.linalg.norm(dr)
                        n_vec = (dr / L) if L > SimulationConstants.ZERO_TOL else np.array([1.0, 0.0])

                        R_int_new[i] += -f_spring * n_vec[0]
                        R_int_new[n + i] += -f_spring * n_vec[1]
                        R_int_new[i + 1] += +f_spring * n_vec[0]
                        R_int_new[n + i + 1] += +f_spring * n_vec[1]

                    # --- Friction (regularized) ---
                    R_fric_new = np.zeros(dof)
                    z_new = z_old.copy()
                    if self.friction_enabled:
                        Fc_node = p.mu_k * np.abs(FN_node)
                        Fs_node = p.mu_s * np.abs(FN_node)

                        for i in range(n):
                            vx = v_trial[i]
                            vz = v_trial[n + i]
                            v_t = np.hypot(vx, vz)

                            z_prev = z_old[i]
                            z_i = z_prev
                            fx = 0.0
                            fz = 0.0

                            if v_t > 1e-8:
                                if p.friction_model == "dahl":
                                    F_tmp, z_i = FrictionModels.dahl(
                                        z_prev, v_t, Fc_node[i], p.sigma_0, h
                                    )
                                elif p.friction_model == "lugre":
                                    F_tmp, z_i = FrictionModels.lugre(
                                        z_prev, v_t, Fc_node[i], Fs_node[i], vs,
                                        p.sigma_0, p.sigma_1, p.sigma_2, h
                                    )
                                elif p.friction_model == "coulomb":
                                    F_tmp = FrictionModels.coulomb_stribeck(
                                        v_t, Fc_node[i], Fs_node[i], vs, p.sigma_2
                                    )
                                elif p.friction_model == "brown-mcphee":
                                    F_tmp = FrictionModels.brown_mcphee(
                                        v_t, Fc_node[i], Fs_node[i], vs
                                    )
                                else:
                                    F_tmp, z_i = FrictionModels.lugre(
                                        z_prev, v_t, Fc_node[i], Fs_node[i], vs,
                                        p.sigma_0, p.sigma_1, p.sigma_2, h
                                    )

                                F_mag = abs(F_tmp)
                                fx = -F_mag * vx / v_t
                                fz = -F_mag * vz / v_t

                            R_fric_new[i] = fx
                            R_fric_new[n + i] = fz
                            z_new[i] = z_i

                    # --- Wall contact (unilateral) ---
                    u_c_new = np.zeros_like(u_c_old)
                    u_c_new[:n] = np.where(q_trial[:n] < 0.0, q_trial[:n], 0.0)
                    du_c = (u_c_new - u_c_old) / h

                    v0_eff = v0_contact_prev.copy()
                    contact_active_new = contact_active_prev
                    if np.any(u_c_new[:n] < 0.0):
                        if (not contact_active_prev) and np.any(du_c[:n] < 0.0):
                            contact_active_new = True
                            mask = du_c[:n] < 0.0
                            v0_eff[:n] = np.where(mask, du_c[:n], 1.0)
                            v0_eff[:n][v0_eff[:n] == 0.0] = 1.0

                        R_raw = ContactModels.compute_force(
                            u_c_new,
                            du_c,
                            v0_eff,
                            p.k_wall,
                            p.cr_wall,
                            p.contact_model,
                        )
                        R_cont_new = -R_raw
                    else:
                        # Loss of contact
                        if np.any(u_c_old[:n] < 0.0):
                            contact_active_new = False
                            v0_eff = np.ones_like(v0_eff)
                        R_cont_new = np.zeros(dof)

                    # --- Mass-to-mass contact (penalty) ---
                    R_mc_new = np.zeros(dof)
                    k_contact = SimulationConstants.MASS_CONTACT_STIFFNESS
                    c_contact = SimulationConstants.MASS_CONTACT_DAMPING

                    for i in range(n - 1):
                        L_current = self.u10[i] + u_s_new[i]
                        if L_current <= self.L_min[i]:
                            penetration = self.L_min[i] - L_current

                            r1 = q_trial[[i, n + i]]
                            r2 = q_trial[[i + 1, n + i + 1]]
                            v1 = v_trial[[i, n + i]]
                            v2 = v_trial[[i + 1, n + i + 1]]

                            dr = r2 - r1
                            dist = np.linalg.norm(dr)
                            n_vec = (dr / dist) if dist > SimulationConstants.ZERO_TOL else np.array([1.0, 0.0])

                            dv = v2 - v1
                            v_rel_normal = float(np.dot(dv, n_vec))

                            F_elastic = k_contact * penetration
                            F_damping = c_contact * abs(v_rel_normal) if v_rel_normal < 0.0 else 0.0
                            F_contact = F_elastic + F_damping

                            R_mc_new[i] -= F_contact * n_vec[0]
                            R_mc_new[n + i] -= F_contact * n_vec[1]
                            R_mc_new[i + 1] += F_contact * n_vec[0]
                            R_mc_new[n + i + 1] += F_contact * n_vec[1]

                    # Assemble total forces
                    R_total_new = R_int_new + R_cont_new + R_fric_new + R_mc_new
                    Cv_new = self.C @ v_trial

                    # Residual consistent with compute_acceleration():
                    #   M a_{n+1} - [(1-α)R_{n+1} + αR_n - (1-α)C v_{n+1} - α C v_n] = 0
                    force = (
                        (1.0 - alpha) * R_total_new
                        + alpha * R_total_old
                        - (1.0 - alpha) * Cv_new
                        - alpha * Cv_old
                    )
                    r = (self.M @ a_trial) - force

                    state = {
                        "q": q_trial,
                        "v": v_trial,
                        "a": a_trial,
                        "R_internal": R_int_new,
                        "R_contact": R_cont_new,
                        "R_friction": R_fric_new,
                        "R_mass_contact": R_mc_new,
                        "u_spring": u_s_new,
                        "u_contact": u_c_new,
                        "X_bw": X_new,
                        "z_friction": z_new,
                        "contact_active": contact_active_new,
                        "v0_contact": v0_eff,
                    }
                    return r, state

                # Newton loop
                state_last: Dict[str, Any] | None = None
                jac_mode = str(getattr(p, "newton_jacobian_mode", "per_step")).lower()
                J_cache: np.ndarray | None = None
                contact_cache: bool | None = None
                for it in range(p.max_iter):
                    r0, state0 = _eval_state(q_guess)
                    state_last = state0

                    iters_this_step = it + 1
                    self.total_iters += 1

                    rnorm = float(np.linalg.norm(r0))
                    ref = float(np.linalg.norm(R_total_old) + 1.0)
                    err = rnorm / ref
                    if err > self.max_residual:
                        self.max_residual = err

                    if err < self.params.newton_tol:
                        converged = True
                        break
                    # Finite-difference Jacobian J = ∂r/∂q (FD; expensive)
                    # newton_jacobian_mode:
                    #   - 'per_step': build once per time step (modified Newton; fast)
                    #   - 'each_iter': rebuild each Newton iteration (pure FD Newton; slow)
                    contact_now = bool(state0.get("contact_active", False))
                    if contact_cache is None:
                        contact_cache = contact_now
                    elif contact_now != contact_cache:
                        # Contact state changed -> refresh Jacobian for stability
                        J_cache = None
                        contact_cache = contact_now

                    rebuild_J = (J_cache is None) or (jac_mode in ("each_iter", "pure", "every_iter"))
                    if rebuild_J:
                        J = np.zeros((dof, dof), dtype=float)
                        eps0 = SimulationConstants.FD_EPSILON
                        for j in range(dof):
                            dq = eps0 * (1.0 + abs(q_guess[j]))
                            q_pert = q_guess.copy()
                            q_pert[j] += dq
                            r_pert, _ = _eval_state(q_pert)
                            J[:, j] = (r_pert - r0) / dq
                        J_cache = J
                    else:
                        J = J_cache

                    # Solve for update
                    try:
                        dq_vec = np.linalg.solve(J, -r0)
                    except np.linalg.LinAlgError:
                        dq_vec = np.linalg.lstsq(J, -r0, rcond=None)[0]

                    self.linear_solves += 1

                    # Backtracking line search (simple Armijo)
                    lam = 1.0
                    q_next = q_guess + dq_vec
                    for _ls in range(SimulationConstants.MAX_LINE_SEARCH_ITERS):
                        r_try, _ = _eval_state(q_next)
                        if np.linalg.norm(r_try) <= (1.0 - SimulationConstants.ARMIJO_COEFF * lam) * rnorm:
                            break
                        lam *= 0.5
                        q_next = q_guess + lam * dq_vec

                    q_guess = q_next

                # Commit last iterate (converged or best-effort)
                if state_last is None:
                    _, state_last = _eval_state(q_guess)

                q[:, step_idx + 1] = state_last["q"]
                qp[:, step_idx + 1] = state_last["v"]
                qpp[:, step_idx + 1] = state_last["a"]
                R_internal[:, step_idx + 1] = state_last["R_internal"]
                R_contact[:, step_idx + 1] = state_last["R_contact"]
                R_friction[:, step_idx + 1] = state_last["R_friction"]
                R_mass_contact[:, step_idx + 1] = state_last["R_mass_contact"]
                u_spring[:, step_idx + 1] = state_last["u_spring"]
                u_contact[:, step_idx + 1] = state_last["u_contact"]
                X_bw[:, step_idx + 1] = state_last["X_bw"]
                z_friction[:, step_idx + 1] = state_last["z_friction"]
                contact_active = bool(state_last["contact_active"])
                v0_contact = state_last["v0_contact"]

                if not converged:
                    logger.warning(
                        "Newton solver did not converge at step %d (||r||/ref = %.3e)",
                        step_idx,
                        err,
                    )

            else:
                # =========================================================
                # Legacy Picard iteration in acceleration (kept for reference)
                # =========================================================

                # Initial guess for a_{n+1}
                qpp[:, step_idx + 1] = qpp[:, step_idx]

                for it in range(p.max_iter):
                    # Reset forces for this iteration
                    R_internal[:, step_idx + 1] = 0.0
                    R_contact[:, step_idx + 1] = 0.0
                    R_friction[:, step_idx + 1] = 0.0
                    u_contact[:, step_idx + 1] = 0.0

                    # Predictor with current acceleration guess
                    q[:, step_idx + 1], qp[:, step_idx + 1] = self.integrator.predict(
                        q[:, step_idx],
                        qp[:, step_idx],
                        qpp[:, step_idx],
                        qpp[:, step_idx + 1],
                        self.h,
                    )

                    # --- Internal Bouc-Wen springs ---
                    for i in range(n - 1):
                        r1 = q[[i, n + i], step_idx + 1]
                        r2 = q[[i + 1, n + i + 1], step_idx + 1]
                        u_spring[i, step_idx + 1] = np.linalg.norm(r2 - r1) - self.u10[i]

                    if step_idx > 0:
                        du = (u_spring[:, step_idx + 1] - u_spring[:, step_idx]) / self.h
                    else:
                        du = np.zeros(n - 1)

                    # Bouc–Wen springs: compute scalar force and distribute in 2D (x/y) along current spring direction
                    u_comp = -u_spring[:, step_idx + 1]  # compression positive (legacy convention)
                    du_comp = -du

                    # Assemble internal forces directly in global DOFs
                    R_internal[:, step_idx + 1] = 0.0
                    for i in range(n - 1):
                        # Update hysteretic state for this spring
                        X_bw[i, step_idx + 1] = BoucWenModel.integrate_rk4(
                            X_bw[i, step_idx],
                            du_comp[i],
                            self.h,
                            self.params.bw_A,
                            self.params.bw_beta,
                            self.params.bw_gamma,
                            self.params.bw_n,
                            self.params.uy[i],
                        )

                        # Scalar spring force (acts along spring axis)
                        f_spring = (
                            self.params.bw_a * (self.k_lin[i] * u_comp[i])
                            + (1.0 - self.params.bw_a) * self.params.fy[i] * X_bw[i, step_idx + 1]
                        )

                        # Current spring direction
                        r1 = q[[i, n + i], step_idx + 1]
                        r2 = q[[i + 1, n + i + 1], step_idx + 1]
                        dr = r2 - r1
                        L = np.linalg.norm(dr)
                        if L > SimulationConstants.ZERO_TOL:
                            n_vec = dr / L
                        else:
                            n_vec = np.array([1.0, 0.0])

                        # Distribute to nodes (action = -reaction)
                        R_internal[i, step_idx + 1] += -f_spring * n_vec[0]
                        R_internal[n + i, step_idx + 1] += -f_spring * n_vec[1]
                        R_internal[i + 1, step_idx + 1] += +f_spring * n_vec[0]
                        R_internal[n + i + 1, step_idx + 1] += +f_spring * n_vec[1]

                    # --- Friction ---
                    self._compute_friction(
                        step_idx,
                        qp[:, step_idx + 1],
                        FN_node,
                        z_friction,
                        R_friction,
                        vs,
                    )

                    # --- Contact ---
                    contact_active, v0_contact = self._compute_contact(
                        step_idx,
                        q[:, step_idx + 1],
                        qp[:, step_idx + 1],
                        u_contact,
                        R_contact,
                        contact_active,
                        v0_contact,
                    )

                    # --- Mass-to-mass contact ---
                    self._compute_mass_contact(
                        step_idx,
                        u_spring,
                        q[:, step_idx + 1],
                        qp[:, step_idx + 1],
                        R_mass_contact,
                    )

                    # --- Corrector: update acceleration ---
                    qpp_old = qpp[:, step_idx + 1].copy()

                    qpp[:, step_idx + 1] = self.integrator.compute_acceleration(
                        self.M,
                        R_internal[:, step_idx + 1], R_internal[:, step_idx],
                        R_contact[:, step_idx + 1], R_contact[:, step_idx],
                        R_friction[:, step_idx + 1], R_friction[:, step_idx],
                        R_mass_contact[:, step_idx + 1], R_mass_contact[:, step_idx],
                        self.C,
                        qp[:, step_idx + 1],
                        qp[:, step_idx],
                    )

                    # Track iterations
                    self.total_iters += 1
                    iters_this_step += 1

                    # Convergence check
                    err = self._check_convergence(qpp[:, step_idx + 1], qpp_old)
                    if err > self.max_residual:
                        self.max_residual = err

                    if err < self.params.newton_tol:
                        converged = True
                        break

                iters_this_step = max(iters_this_step, 1)

                if not converged:
                    logger.warning(
                        "Picard solver did not converge at step %d (rel Δa = %.3e)",
                        step_idx,
                        err,
                    )

            # Track max iterations needed per step
            if iters_this_step > self.max_iters_per_step:
                self.max_iters_per_step = iters_this_step

            iters_hist[step_idx] = iters_this_step


            # --------------------------------------------------
            # Energy bookkeeping (Euler-Lagrange formulation)
            # --------------------------------------------------
            # IMPORTANT: energies are stored at *end* time levels t_{n+1}.
            # We still integrate power using the HHT-α evaluation point
            # (forces at t_{n+α}, velocities consistent with that).
            q_new = q[:, step_idx + 1]
            q_old = q[:, step_idx]
            v_new = qp[:, step_idx + 1]
            v_old = qp[:, step_idx]

            # NOTE on HHT-α and energy bookkeeping:
            # HHT-α evaluates equilibrium at an *extrapolated* time level when α < 0,
            # i.e. x_{n+α} = (1-α) x_{n+1} + α x_n with α negative.
            # That is great for high-frequency numerical damping, but it is NOT a
            # good quadrature point for physical work over [t_n, t_{n+1}].
            #
            # For energy/work accounting we therefore use a *consistent trapezoidal*
            # rule on the interval [t_n, t_{n+1}] (midpoint inside the step):
            v_pow = 0.5 * (v_new + v_old)

            # =========================================================
            # KINETIC ENERGY (end level): T_{n+1} = 0.5 * v_{n+1}^T M v_{n+1}
            # =========================================================
            E_kin[step_idx + 1] = 0.5 * float(v_new.T @ self.M @ v_new)

            # =========================================================
            # POTENTIAL ENERGY: V = V_spring + V_contact
            # =========================================================

            # Spring potential (conservative part of Bouc-Wen) at end level
            # For Bouc-Wen: f = a*k*u + (1-a)*fy*z
            # Conservative part: f_cons = a*k*u → V = 0.5*a*k*u^2
            if p.bw_a > SimulationConstants.ZERO_TOL:
                E_pot_spring[step_idx + 1] = 0.5 * p.bw_a * float(
                    np.sum(self.k_lin * u_spring[:, step_idx + 1] ** 2)
                )
            else:
                E_pot_spring[step_idx + 1] = 0.0

            # Contact potential (elastic part only) at end level
            # Wall at x=0 → penetration δ = max(-x, 0)
            delta = np.maximum(-u_contact[:n, step_idx + 1], 0.0)
            model = self.params.contact_model.lower()
            if model in ["hooke", "ye", "pant-wijeyewickrema", "anagnostopoulos"]:
                exp = 1.0
            else:
                exp = 1.5

            if np.any(delta > 0.0):
                if exp == 1.0:
                    E_pot_contact[step_idx + 1] = 0.5 * self.params.k_wall * float(np.sum(delta ** 2))
                else:
                    # For Hertzian: V = ∫ k*δ^n dδ = k*δ^(n+1)/(n+1)
                    E_pot_contact[step_idx + 1] = (
                        self.params.k_wall / (exp + 1.0) * float(np.sum(delta ** (exp + 1.0)))
                    )
            else:
                E_pot_contact[step_idx + 1] = 0.0

            # Total potential and mechanical energy
            E_pot[step_idx + 1] = E_pot_spring[step_idx + 1] + E_pot_contact[step_idx + 1]
            E_mech[step_idx + 1] = E_kin[step_idx + 1] + E_pot[step_idx + 1]

            # =========================================================
            # DISSIPATION: E_diss = -∫ qdot^T Q_nc dt
            # =========================================================
            # ---------------------------------------------------------
            # DISSIPATION: E_diss = -∫ qdot_{n+α}^T Q_nc_{n+α} dt
            # ---------------------------------------------------------
            # Non-conservative forces at n and n+1
            Q_ray_new = -self.C @ v_new
            Q_ray_old = -self.C @ v_old

            Q_fric_new = R_friction[:, step_idx + 1]
            Q_fric_old = R_friction[:, step_idx]

            Q_mc_new = R_mass_contact[:, step_idx + 1]
            Q_mc_old = R_mass_contact[:, step_idx]

            # Bouc–Wen non-conservative part (2D distribution)
            Q_bw_new = np.zeros(dof)
            Q_bw_old = np.zeros(dof)

            # If bw_a < 0, the linear term a*k*u represents post-yield softening (descending branch).
            # It is not physically recoverable "elastic" energy, so we treat its work as non-conservative
            # to keep the energy balance meaningful when experimenting with negative a.
            Q_soft_new = np.zeros(dof)
            Q_soft_old = np.zeros(dof)
            u_comp_new = -u_spring[:, step_idx + 1] if (n > 1) else np.zeros(0)
            u_comp_old = -u_spring[:, step_idx] if (n > 1) else np.zeros(0)

            for i in range(n - 1):
                f_nc_new = (1.0 - p.bw_a) * p.fy[i] * X_bw[i, step_idx + 1]
                f_nc_old = (1.0 - p.bw_a) * p.fy[i] * X_bw[i, step_idx]

                # Optional softening work (a < 0): treat linear term as non-conservative
                if p.bw_a < 0.0:
                    f_soft_new = p.bw_a * (self.k_lin[i] * u_comp_new[i])
                    f_soft_old = p.bw_a * (self.k_lin[i] * u_comp_old[i])
                else:
                    f_soft_new = 0.0
                    f_soft_old = 0.0

                r1n = q[[i, n + i], step_idx + 1]
                r2n = q[[i + 1, n + i + 1], step_idx + 1]
                drn = r2n - r1n
                Ln = np.linalg.norm(drn)
                n_vec_n = (drn / Ln) if Ln > SimulationConstants.ZERO_TOL else np.array([1.0, 0.0])

                r1o = q[[i, n + i], step_idx]
                r2o = q[[i + 1, n + i + 1], step_idx]
                dro = r2o - r1o
                Lo = np.linalg.norm(dro)
                n_vec_o = (dro / Lo) if Lo > SimulationConstants.ZERO_TOL else np.array([1.0, 0.0])

                # new
                Q_bw_new[i] -= f_nc_new * n_vec_n[0]
                Q_bw_new[n + i] -= f_nc_new * n_vec_n[1]
                Q_bw_new[i + 1] += f_nc_new * n_vec_n[0]
                Q_bw_new[n + i + 1] += f_nc_new * n_vec_n[1]

                # new (softening term)
                Q_soft_new[i] -= f_soft_new * n_vec_n[0]
                Q_soft_new[n + i] -= f_soft_new * n_vec_n[1]
                Q_soft_new[i + 1] += f_soft_new * n_vec_n[0]
                Q_soft_new[n + i + 1] += f_soft_new * n_vec_n[1]

                # old
                Q_bw_old[i] -= f_nc_old * n_vec_o[0]
                Q_bw_old[n + i] -= f_nc_old * n_vec_o[1]
                Q_bw_old[i + 1] += f_nc_old * n_vec_o[0]
                Q_bw_old[n + i + 1] += f_nc_old * n_vec_o[1]

                # old (softening term)
                Q_soft_old[i] -= f_soft_old * n_vec_o[0]
                Q_soft_old[n + i] -= f_soft_old * n_vec_o[1]
                Q_soft_old[i + 1] += f_soft_old * n_vec_o[0]
                Q_soft_old[n + i + 1] += f_soft_old * n_vec_o[1]

            # Contact damping split at n and n+1 (elastic potential handled in V_contact)
            delta_old = np.maximum(-u_contact[:n, step_idx], 0.0)

            def _contact_elastic_force(d: np.ndarray) -> np.ndarray:
                """Elastic contact force in the SAME sign convention as R_contact.

                Note: delta = -u_contact (penetration). With V(delta) = 0.5*k*delta^2,
                the generalized force on q is +k*delta (positive, pushes mass away from wall).
                """
                R_el = np.zeros(dof)
                if np.any(d > 0.0):
                    for j in range(n):
                        if d[j] > 0.0:
                            if model in ["hooke", "ye", "pant-wijeyewickrema", "anagnostopoulos"]:
                                R_el[j] = p.k_wall * d[j]
                            else:
                                R_el[j] = p.k_wall * d[j] ** 1.5
                return R_el

            R_contact_elastic_new = _contact_elastic_force(delta)
            R_contact_elastic_old = _contact_elastic_force(delta_old)

            R_contact_total_new = R_contact[:, step_idx + 1]
            R_contact_total_old = R_contact[:, step_idx]

            Q_cd_new = R_contact_total_new - R_contact_elastic_new
            Q_cd_old = R_contact_total_old - R_contact_elastic_old

            # Trapezoidal midpoint for energy/work (inside the step)
            Q_ray_mid = 0.5 * (Q_ray_new + Q_ray_old)
            Q_bw_mid = 0.5 * (Q_bw_new + Q_bw_old)
            Q_soft_mid = 0.5 * (Q_soft_new + Q_soft_old)
            Q_cd_mid = 0.5 * (Q_cd_new + Q_cd_old)
            Q_fric_mid = 0.5 * (Q_fric_new + Q_fric_old)
            Q_mc_mid = 0.5 * (Q_mc_new + Q_mc_old)

            Q_nc_mid = Q_ray_mid + Q_bw_mid + Q_soft_mid + Q_cd_mid + Q_fric_mid + Q_mc_mid

            # External forces (excluding rigid wall): currently none
            Q_ext_mid = np.zeros(dof)

            # Diagnostics: contact kinematics at end level (front mass)
            delta_hist[step_idx + 1] = float(delta[0]) if delta.size else 0.0
            delta_dot_hist[step_idx] = float(
                (delta_hist[step_idx + 1] - delta_hist[step_idx]) / self.h
            )
            f_el_hist[step_idx + 1] = float(R_contact_elastic_new[0])
            f_damp_hist[step_idx + 1] = float(R_contact_total_new[0] - R_contact_elastic_new[0])

            # Diagnostics: midpoint powers
            P_nc_hist[step_idx] = float(v_pow.T @ Q_nc_mid)
            P_ext_hist[step_idx] = float(v_pow.T @ Q_ext_mid)

            # Integrate dissipation (component-wise, consistent with t_{n+α})
            dE_rayleigh = -float(v_pow.T @ Q_ray_mid) * self.h
            dE_bw = -float(v_pow.T @ Q_bw_mid) * self.h
            dE_softening = -float(v_pow.T @ Q_soft_mid) * self.h
            dE_contact_damp = -float(v_pow.T @ Q_cd_mid) * self.h
            dE_friction = -float(v_pow.T @ Q_fric_mid) * self.h
            dE_mass_contact = -float(v_pow.T @ Q_mc_mid) * self.h

            E_diss_rayleigh[step_idx + 1] = E_diss_rayleigh[step_idx] + dE_rayleigh
            E_diss_bw[step_idx + 1] = E_diss_bw[step_idx] + dE_bw
            E_diss_softening[step_idx + 1] = E_diss_softening[step_idx] + dE_softening
            E_diss_contact_damp[step_idx + 1] = E_diss_contact_damp[step_idx] + dE_contact_damp
            E_diss_friction[step_idx + 1] = E_diss_friction[step_idx] + dE_friction
            E_diss_mass_contact[step_idx + 1] = E_diss_mass_contact[step_idx] + dE_mass_contact

            E_diss_total[step_idx + 1] = (
                E_diss_rayleigh[step_idx + 1]
                + E_diss_bw[step_idx + 1]
                + E_diss_softening[step_idx + 1]
                + E_diss_contact_damp[step_idx + 1]
                + E_diss_friction[step_idx + 1]
                + E_diss_mass_contact[step_idx + 1]
            )

            # =========================================================
            # EXTERNAL WORK: W_ext = ∫ qdot^T Q_ext dt
            # =========================================================
            # FIX 5: Do NOT reset W_ext. For a rigid wall, simply do not add wall work.
            # (Wall reaction is treated internally via V_contact and E_diss_contact_damp.)
            dW_ext_other = float(v_pow.T @ Q_ext_mid) * self.h
            W_ext[step_idx + 1] = W_ext[step_idx] + dW_ext_other

            # =========================================================
            # NUMERICAL RESIDUAL: E_num = E0 + W_ext - (E_mech + E_diss)
            # =========================================================
            E_num[step_idx + 1] = E0 + W_ext[step_idx + 1] - (E_mech[step_idx + 1] + E_diss_total[step_idx + 1])
            if abs(E0) > SimulationConstants.ZERO_TOL:
                E_num_ratio[step_idx + 1] = abs(E_num[step_idx + 1]) / abs(E0)
            else:
                E_num_ratio[step_idx + 1] = 0.0

        # --------------------------------------------------
        # Peak residual diagnostic print (exactly 10 lines)
        # --------------------------------------------------
        try:
            idx_peak = int(np.argmax(E_num_ratio))
            if idx_peak >= 1:
                n0 = idx_peak - 1
                # Energies are stored at end levels t_{n+1}
                t_peak = float(self.t[idx_peak])
                dt = float(self.h)
                iters_peak = int(iters_hist[n0])

                dE_mech = float(E_mech[idx_peak] - E_mech[n0])
                dE_diss = float(E_diss_total[idx_peak] - E_diss_total[n0])
                dW_ext = float(W_ext[idx_peak] - W_ext[n0])
                r_inc = float(dE_mech + dE_diss - dW_ext)

                P_nc = float(P_nc_hist[n0])
                P_ext = float(P_ext_hist[n0])

                delta0 = float(delta_hist[idx_peak])
                delta_dot0 = float(delta_dot_hist[n0])
                f_el0 = float(f_el_hist[idx_peak])
                f_damp0 = float(f_damp_hist[idx_peak])
                V_contact0 = float(E_pot_contact[idx_peak])

                print(f"[EB_PEAK 01/10] step={idx_peak} t={t_peak:.6e} dt={dt:.3e} iters={iters_peak}")
                print(f"[EB_PEAK 02/10] E_kin={E_kin[idx_peak]:.6e} E_pot={E_pot[idx_peak]:.6e} E_mech={E_mech[idx_peak]:.6e} E_diss={E_diss_total[idx_peak]:.6e}")
                print(f"[EB_PEAK 03/10] W_ext={W_ext[idx_peak]:.6e} E_num={E_num[idx_peak]:.6e} E_num_ratio={E_num_ratio[idx_peak]:.6e}")
                print(f"[EB_PEAK 04/10] dE_mech={dE_mech:.6e} dE_diss={dE_diss:.6e} dW_ext={dW_ext:.6e}")
                print(f"[EB_PEAK 05/10] r_inc=dE_mech+dE_diss-dW_ext={r_inc:.6e}")
                print(f"[EB_PEAK 06/10] P_nc=qdot_mid@Q_nc_mid={P_nc:.6e}")
                print(f"[EB_PEAK 07/10] P_ext=qdot_mid@Q_ext_mid={P_ext:.6e}")
                print(f"[EB_PEAK 08/10] contact delta={delta0:.6e} delta_dot={delta_dot0:.6e}")
                print(f"[EB_PEAK 09/10] contact f_el={f_el0:.6e} f_damp={f_damp0:.6e}")
                print(f"[EB_PEAK 10/10] contact V_contact={V_contact0:.6e}")
        except Exception:
            # Never fail a run because of optional diagnostics
            pass

        # Sync linear solves count from integrator
        self.linear_solves = self.integrator.n_lu

        energies = {
            # Mechanical energy components
            "E_kin": E_kin,  # Kinetic energy T
            "E_pot_spring": E_pot_spring,  # Spring potential (conservative part)
            "E_pot_contact": E_pot_contact,  # Contact potential (elastic part)
            "E_pot": E_pot,  # Total potential V
            "E_mech": E_mech,  # Total mechanical E = T + V
            # Work and dissipation
            "W_ext": W_ext,  # External work
            "E_diss_rayleigh": E_diss_rayleigh,  # Rayleigh damping dissipation
            "E_diss_bw": E_diss_bw,  # Bouc-Wen hysteretic dissipation
            "E_diss_softening": E_diss_softening,  # Softening work (bw_a<0) treated as dissipation
            "E_diss_contact_damp": E_diss_contact_damp,  # Contact damping dissipation
            "E_diss_friction": E_diss_friction,  # Friction dissipation
            "E_diss_mass_contact": E_diss_mass_contact,  # Mass contact dissipation
            "E_diss_total": E_diss_total,  # Total dissipation
            # Numerical residual
            "E_num": E_num,  # Numerical residual
            "E_num_ratio": E_num_ratio,  # |E_num| / E0
            "E0": E0,  # Initial energy
        }

        return self._build_results_dataframe(
            q=q,
            qp=qp,
            qpp=qpp,
            R_internal=R_internal,
            R_contact=R_contact,
            R_friction=R_friction,
            R_mass_contact=R_mass_contact,
            u_contact=u_contact,
            u_spring=u_spring,
            X_bw=X_bw,
            energies=energies,
        )

    # ----------------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------------

    def _compute_friction(
        self,
        step_idx: int,
        qp: np.ndarray,
        FN_node: np.ndarray,
        z_friction: np.ndarray,
        R_friction: np.ndarray,
        vs: float,
    ) -> None:
        """
        Compute friction forces per mass node and distribute to x/z DOFs.

        If friction is disabled (model 'none' or zero coefficients),
        we simply copy the internal state and return zeros.
        """
        p = self.params
        n = p.n_masses
        dof = 2 * n

        assert FN_node.shape[0] == n
        assert qp.shape[0] == dof

        # --------------------------------------------------------------
        # Early exit: friction disabled or coefficients zero (pre-computed)
        # --------------------------------------------------------------
        if not self.friction_enabled:
            # No new friction forces, just carry forward the internal state
            R_friction[:, step_idx + 1] = 0.0
            z_friction[:, step_idx + 1] = z_friction[:, step_idx]
            return

        # --------------------------------------------------------------
        # Full friction computation (LuGre/Dahl/etc.)
        # --------------------------------------------------------------
        R_friction[:, step_idx + 1] = 0.0

        Fc_node = p.mu_k * np.abs(FN_node)
        Fs_node = p.mu_s * np.abs(FN_node)

        for i in range(n):
            vx = qp[i]
            vz = qp[n + i]
            v_t = np.hypot(vx, vz)

            z_prev = z_friction[i, step_idx]
            z_new = z_prev
            fx = 0.0
            fz = 0.0

            if v_t > 1e-8:
                if p.friction_model == "dahl":
                    F_tmp, z_new = FrictionModels.dahl(
                        z_prev, v_t, Fc_node[i], p.sigma_0, self.h
                    )
                elif p.friction_model == "lugre":
                    F_tmp, z_new = FrictionModels.lugre(
                        z_prev, v_t, Fc_node[i], Fs_node[i], vs,
                        p.sigma_0, p.sigma_1, p.sigma_2, self.h
                    )
                elif p.friction_model == "coulomb":
                    F_tmp = FrictionModels.coulomb_stribeck(
                        v_t, Fc_node[i], Fs_node[i], vs, p.sigma_2
                    )
                elif p.friction_model == "brown-mcphee":
                    F_tmp = FrictionModels.brown_mcphee(
                        v_t, Fc_node[i], Fs_node[i], vs
                    )
                else:
                    # Fallback: LuGre
                    F_tmp, z_new = FrictionModels.lugre(
                        z_prev, v_t, Fc_node[i], Fs_node[i], vs,
                        p.sigma_0, p.sigma_1, p.sigma_2, self.h
                    )

                F_mag = abs(F_tmp)
                fx = -F_mag * vx / v_t
                fz = -F_mag * vz / v_t

            R_friction[i, step_idx + 1] = fx
            R_friction[n + i, step_idx + 1] = fz

            z_friction[i, step_idx + 1] = z_new

    def _compute_contact(
        self,
        step_idx: int,
        q: np.ndarray,
        qp: np.ndarray,
        u_contact: np.ndarray,
        R_contact: np.ndarray,
        contact_active: bool,
        v0_contact: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """Compute contact forces at wall and update contact state."""
        p = self.params
        n = p.n_masses
        dof = 2 * n

        u_contact[:, step_idx + 1] = 0.0

        # Check for contact (x < 0)
        if np.any(q[:n] < 0.0):
            for i in range(n):
                if q[i] < 0.0:
                    u_contact[i, step_idx + 1] = q[i]

            if step_idx > 0:
                du_contact = (u_contact[:, step_idx + 1] -
                              u_contact[:, step_idx]) / self.h
            else:
                du_contact = np.zeros(dof)

            # First contact
            if (not contact_active) and np.any(du_contact[:n] < 0.0):
                contact_active = True
                # Only set v0 for x-DOFs that are actually in contact
                mask_contact_x = du_contact[:n] < 0.0
                v0_contact[:n] = np.where(mask_contact_x, du_contact[:n], 1.0)
                v0_contact[:n][v0_contact[:n] == 0.0] = 1.0

            R = ContactModels.compute_force(
                u_contact[:, step_idx + 1],
                du_contact,
                v0_contact,
                p.k_wall,
                p.cr_wall,
                p.contact_model,
            )

            R_contact[:, step_idx + 1] = -R

        # Loss of contact
        elif step_idx > 0 and np.any(u_contact[:n, step_idx] < 0.0):
            contact_active = False
            v0_contact = np.ones_like(v0_contact)

        return contact_active, v0_contact

    def _compute_mass_contact(
        self,
        step_idx: int,
        u_spring: np.ndarray,
        q: np.ndarray,
        qp: np.ndarray,
        R_mass_contact: np.ndarray,
    ) -> None:
        """
        Detect and enforce contact between adjacent masses.

        When spring compression causes adjacent masses to touch
        (spring length ≤ L_min), add contact force to prevent
        further compression.

        Args:
            step_idx: Current timestep index
            u_spring: Spring deformations (negative = compression)
            q: Current positions
            qp: Current velocities
            R_mass_contact: Mass contact force vector (output)
        """
        p = self.params
        n = p.n_masses

        # Contact parameters
        k_contact = SimulationConstants.MASS_CONTACT_STIFFNESS  # Contact stiffness [N/m]
        c_contact = SimulationConstants.MASS_CONTACT_DAMPING  # Contact damping [N·s/m]

        R_mass_contact[:, step_idx + 1] = 0.0

        for i in range(n - 1):
            # Current spring length: L = L0 + u_spring
            # (u_spring is negative for compression)
            L_current = self.u10[i] + u_spring[i, step_idx + 1]

            # Check if masses are in contact (spring fully compressed)
            if L_current <= self.L_min[i]:
                penetration = self.L_min[i] - L_current
                self.mass_contact_active[i] = True

                # Get positions and velocities of masses i and i+1
                r1 = q[[i, n + i], step_idx + 1]
                r2 = q[[i + 1, n + i + 1], step_idx + 1]
                v1 = qp[[i, n + i], step_idx + 1]
                v2 = qp[[i + 1, n + i + 1], step_idx + 1]

                # Contact normal (from mass i to mass i+1)
                dr = r2 - r1
                dist = np.linalg.norm(dr)
                if dist > SimulationConstants.ZERO_TOL:
                    n_vec = dr / dist
                else:
                    # Masses at same location, use x-direction
                    n_vec = np.array([1.0, 0.0])

                # Relative velocity along normal
                dv = v2 - v1
                v_rel_normal = np.dot(dv, n_vec)

                # Contact force (penalty + damping)
                # Elastic component
                F_elastic = k_contact * penetration

                # Damping component (only if approaching)
                F_damping = 0.0
                if v_rel_normal < 0:  # masses approaching
                    F_damping = c_contact * abs(v_rel_normal)

                F_contact = F_elastic + F_damping

                # Apply force along normal direction
                # Force on mass i: push away from mass i+1 (direction -n_vec)
                R_mass_contact[i, step_idx + 1] -= F_contact * n_vec[0]
                R_mass_contact[n + i, step_idx + 1] -= F_contact * n_vec[1]

                # Force on mass i+1: equal and opposite (direction +n_vec)
                R_mass_contact[i + 1, step_idx + 1] += F_contact * n_vec[0]
                R_mass_contact[n + i + 1, step_idx + 1] += F_contact * n_vec[1]

            elif L_current > self.L_min[i] * 1.1:  # Hysteresis: 10% buffer
                # Release contact if spring extends beyond 110% of minimum
                self.mass_contact_active[i] = False

    @staticmethod
    def _check_convergence(qpp_new: np.ndarray, qpp_old: np.ndarray) -> float:
        """Symmetric relative change in acceleration."""
        delta = qpp_new - qpp_old
        norm_new = np.linalg.norm(qpp_new)
        norm_old = np.linalg.norm(qpp_old)
        denom = norm_new + norm_old + 1e-16
        return 2.0 * np.linalg.norm(delta) / denom

    def _build_results_dataframe(
        self,
        q: np.ndarray,
        qp: np.ndarray,
        qpp: np.ndarray,
        R_internal: np.ndarray,
        R_contact: np.ndarray,
        R_friction: np.ndarray,
        R_mass_contact: np.ndarray,
        u_contact: np.ndarray,
        u_spring: np.ndarray,
        X_bw: np.ndarray,
        energies: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Build results DataFrame for export (including energy bookkeeping)."""
        n_masses = self.params.n_masses
        F_total = R_contact[0, :]          # [N], positive in compression
        F_total_clamped = np.maximum(F_total, 0.0)

        u_pen_mm = -u_contact[0, :] * 1000.0
        a_front = qpp[0, :] / GRAVITY

        delta = np.maximum(-u_contact[0, :], 0.0)

        model = self.params.contact_model.lower()
        if model in ["hooke", "ye", "pant-wijeyewickrema", "anagnostopoulos"]:
            exponent = 1.0
        else:
            exponent = 1.5

        F_backbone_MN = (self.params.k_wall * delta ** exponent) / 1e6

        if X_bw.size > 0:
            bw_state = X_bw[0, :]
        else:
            bw_state = np.zeros_like(self.t)

        # Extract energy components from Euler-Lagrange formulation
        E_kin = energies["E_kin"]
        E_pot_spring = energies["E_pot_spring"]
        E_pot_contact = energies["E_pot_contact"]
        E_pot = energies["E_pot"]
        E_mech = energies["E_mech"]
        W_ext = energies["W_ext"]
        E_diss_rayleigh = energies["E_diss_rayleigh"]
        E_diss_bw = energies["E_diss_bw"]
        E_diss_softening = energies.get("E_diss_softening", np.zeros_like(E_diss_bw))
        E_diss_contact_damp = energies["E_diss_contact_damp"]
        E_diss_friction = energies["E_diss_friction"]
        E_diss_mass_contact = energies["E_diss_mass_contact"]
        E_diss_total = energies["E_diss_total"]
        E_num = energies["E_num"]
        E_num_ratio = energies["E_num_ratio"]
        E0 = energies["E0"]

        df = pd.DataFrame(
            {
                "Time_s": self.t,
                "Time_ms": self.t * 1000.0,
                "Impact_Force_MN": F_total_clamped / 1e6,
                "Penetration_mm": u_pen_mm,
                "Acceleration_g": a_front,
                "Velocity_m_s": qp[0, :],
                "Position_x_m": q[0, :],
                "BoucWen_State_1": bw_state,
                "Backbone_Force_MN": F_backbone_MN,
                # Mechanical energy components [J]
                "E_kin_J": E_kin,
                "E_pot_spring_J": E_pot_spring,
                "E_pot_contact_J": E_pot_contact,
                "E_pot_J": E_pot,
                "E_mech_J": E_mech,
                # Work and dissipation [J]
                "W_ext_J": W_ext,
                "E_diss_rayleigh_J": E_diss_rayleigh,
                "E_diss_bw_J": E_diss_bw,
                "E_diss_softening_J": E_diss_softening,
                "E_diss_contact_damp_J": E_diss_contact_damp,
                "E_diss_friction_J": E_diss_friction,
                "E_diss_mass_contact_J": E_diss_mass_contact,
                "E_diss_total_J": E_diss_total,
                # Numerical residual [J]
                "E_num_J": E_num,
                "E_num_ratio": E_num_ratio,
                "E0_J": np.full_like(self.t, E0, dtype=float),
            }
        )


        # Add per-mass/spring columns in one batch to avoid DataFrame fragmentation
        extra_cols: dict[str, object] = {}

        # Per-mass kinematics
        for i in range(n_masses):
            idx = i + 1
            extra_cols[f"Mass{idx}_Position_x_m"] = q[i, :]
            extra_cols[f"Mass{idx}_Position_y_m"] = q[n_masses + i, :]
            extra_cols[f"Mass{idx}_Velocity_x_m_s"] = qp[i, :]
            extra_cols[f"Mass{idx}_Velocity_y_m_s"] = qp[n_masses + i, :]
            extra_cols[f"Mass{idx}_Acceleration_x_m_s2"] = qpp[i, :]
            extra_cols[f"Mass{idx}_Acceleration_y_m_s2"] = qpp[n_masses + i, :]

        # Per-mass nodal forces (exported for debugging and F–u loops)
        # These are the *applied* generalized forces in global DOFs (x/y) for each mass.
        for i in range(n_masses):
            idx = i + 1
            fx_int = R_internal[i, :]
            fy_int = R_internal[n_masses + i, :]
            fx_wall = R_contact[i, :]
            fy_wall = R_contact[n_masses + i, :]
            fx_fric = R_friction[i, :]
            fy_fric = R_friction[n_masses + i, :]
            fx_mc = R_mass_contact[i, :]
            fy_mc = R_mass_contact[n_masses + i, :]

            extra_cols[f"Mass{idx}_Force_internal_x_N"] = fx_int
            extra_cols[f"Mass{idx}_Force_internal_y_N"] = fy_int
            extra_cols[f"Mass{idx}_Force_wall_x_N"] = fx_wall
            extra_cols[f"Mass{idx}_Force_wall_y_N"] = fy_wall
            extra_cols[f"Mass{idx}_Force_friction_x_N"] = fx_fric
            extra_cols[f"Mass{idx}_Force_friction_y_N"] = fy_fric
            extra_cols[f"Mass{idx}_Force_mass_contact_x_N"] = fx_mc
            extra_cols[f"Mass{idx}_Force_mass_contact_y_N"] = fy_mc

            extra_cols[f"Mass{idx}_Force_total_x_N"] = fx_int + fx_wall + fx_fric + fx_mc
            extra_cols[f"Mass{idx}_Force_total_y_N"] = fy_int + fy_wall + fy_fric + fy_mc

        # Per-spring response (disp/force)
        if n_masses > 1:
            u_comp = -u_spring
            f_spring = (
                self.params.bw_a * (self.k_lin[:, None] * u_comp)
                + (1.0 - self.params.bw_a) * (self.params.fy[:, None] * X_bw)
            )
            for i in range(n_masses - 1):
                idx = i + 1
                extra_cols[f"Spring{idx}_Disp_m"] = u_spring[i, :]
                extra_cols[f"Spring{idx}_Force_N"] = f_spring[i, :]

        if extra_cols:
            df = pd.concat([df, pd.DataFrame(extra_cols)], axis=1)

        # Attach some solver statistics as DataFrame metadata
        try:
            df.attrs["n_dof"] = self.M.shape[0]
            df.attrs["n_masses"] = self.params.n_masses
            df.attrs["n_lu"] = getattr(self.integrator, "n_lu", 0)
            df.attrs["n_nonlinear_iters"] = self.total_iters
            df.attrs["max_iters_per_step"] = self.max_iters_per_step
            df.attrs["max_residual"] = self.max_residual
            # Store actual timestep used (not requested h_init, but effective dt)
            df.attrs["dt_eff"] = self.h
            df.attrs["h_requested"] = self.params.h_init
            df.attrs["newton_tol"] = self.params.newton_tol
            df.attrs["alpha_hht"] = self.params.alpha_hht

            # Compute strain-rate metrics
            # Use L_ref from params if available, otherwise default to 1.0 m
            L_ref = getattr(self.params, "L_ref_m", 1.0)
            strain_metrics = strain_rate_metrics(df, L_ref_m=L_ref)
            df.attrs.update(strain_metrics)
        except Exception:
            # Metadata is best-effort; never break the simulation because of it
            logger.debug(
                "Could not attach solver statistics to results DataFrame.",
                exc_info=True,
            )

        return df

# ====================================================================
# PUBLIC ENTRY POINT
# ====================================================================

def get_default_simulation_params() -> dict:
    """
    Baseline parameter set corresponding roughly to a 40 t passenger car
    (Pioneer / ICE-1 type) impacting a rigid wall at about 80 km/h.

    Returned as a plain dict so it can be updated from YAML/JSON configs
    and then passed into SimulationParams(**params).
    """
    # Time integration baseline
    T_max = 0.4      # total simulation time [s]
    h_init = 1e-4    # time step [s]
    n_steps = int(T_max / h_init)

    return {
        # ------------------------------------------------------------------
        # Geometry and kinematics
        # ------------------------------------------------------------------
        "n_masses": 7,
        # masses in kg (approx. 40 t total)
        "masses": [4_000.0, 10_000.0, 4_000.0, 4_000.0, 4_000.0, 10_000.0, 4_000.0],
        # mass positions along train [m]
        "x_init": [1.5, 4.5, 8.0, 11.5, 15.0, 18.5, 21.5],
        # vertical coordinates (flat track)
        "y_init": [0.0] * 7,

        # Initial velocity and angle
        "v0_init": -22.22,   # [m/s] ≈ -80 km/h towards the wall
        "angle_rad": 0.0,    # [rad]
        "d0": 0.0,           # initial distance / pre-penetration [m]

        # ------------------------------------------------------------------
        # Vehicle crushing springs (between masses) – 6 springs for 7 masses
        # ------------------------------------------------------------------
        # yield force [N] and yield deformation [m]
        "fy": [15e6] * 6,    # 15 MN
        "uy": [0.20] * 6,    # 200 mm

        # ------------------------------------------------------------------
        # Contact with rigid wall
        # ------------------------------------------------------------------
        "k_wall": 45e6,                       # [N/m]
        "cr_wall": 0.8,                       # restitution / damping parameter
        "contact_model": "lankarani-nikravesh",

        # ------------------------------------------------------------------
        # Optional building / pier SDOF (disabled by default)
        # ------------------------------------------------------------------
        "building_enable": False,
        "building_mass": 5.0e6,       # [kg] if enabled
        "building_zeta": SimulationConstants.DEFAULT_BUILDING_ZETA,  # 5% critical
        "building_height": 10.0,      # [m]
        "building_model": "linear",   # or "takeda"
        "building_uy": 0.05,          # [m]
        "building_uy_mm": 50.0,       # [mm], for UI convenience
        "building_alpha": 0.0,        # Takeda post-yield stiffness ratio
        "building_gamma": 0.0,        # Takeda pinching parameter

        # ------------------------------------------------------------------
        # Friction (disabled by default)
        # ------------------------------------------------------------------
        "mu_s": 0.0,
        "mu_k": 0.0,
        "sigma_0": 0.0,
        "sigma_1": 0.0,
        "sigma_2": 0.0,
        "friction_model": "none",     # "lugre", "dahl", "coulomb", "brown-mcphee", ...

        # ------------------------------------------------------------------
        # Bouc–Wen hysteresis (dimensionless backbone)
        # ------------------------------------------------------------------
        "bw_a": 1.0,
        "bw_A": 1.0,
        "bw_beta": 0.5,
        "bw_gamma": 0.5,
        "bw_n": 2,

        # ------------------------------------------------------------------
        # Nonlinear solver / time integration controls
        # ------------------------------------------------------------------
        "alpha_hht": -0.15,      # HHT-α parameter
        "newton_tol": 1e-6,
        "max_iter": 25,
        "solver": "newton",     # "newton" (NR) or "picard" (legacy)
        "newton_jacobian_mode": "per_step",  # "per_step" (fast) or "each_iter" (pure/slow)
        "h_init": h_init,        # [s]
        "T_max": T_max,          # [s]
        "step": n_steps,         # number of steps
        "T_int": (0.0, T_max),   # time interval [s]
    }


def run_simulation(params: SimulationParams | Dict[str, Any]) -> pd.DataFrame:
    """
    High-level convenience wrapper.

    - If a dict is passed, it may contain only overrides; missing fields are
      filled from get_default_simulation_params(), and all types are normalised
      (floats/ints/arrays).
    - If a SimulationParams instance is passed, we still run it through the
      same normalisation pipeline to be robust against objects constructed
      directly from raw YAML/JSON dicts.
    """
    # Track which time/grid keys were explicitly set by the user when passing a dict.
    user_overrides: Dict[str, Any] = {}
    user_provided_step = False
    user_provided_T_int = False
    user_provided_T_max = False
    user_provided_h_init = False

    # Get defaults to compare against
    defaults = get_default_simulation_params()

    if isinstance(params, SimulationParams):
        # Dataclass -> plain dict
        raw = {f.name: getattr(params, f.name) for f in fields(SimulationParams)}
        # Don't assume dataclass means explicit time grid - compare to defaults instead
        user_provided_step = raw.get("step") != defaults.get("step")
        user_provided_T_int = raw.get("T_int") != defaults.get("T_int")
        user_provided_T_max = raw.get("T_max") != defaults.get("T_max")
        user_provided_h_init = raw.get("h_init") != defaults.get("h_init")
    else:
        # Dict of overrides coming from CLI / YAML
        user_overrides = params or {}
        user_provided_step = ("step" in user_overrides) and (user_overrides.get("step") is not None)
        user_provided_T_int = ("T_int" in user_overrides) and (user_overrides.get("T_int") is not None)
        user_provided_T_max = ("T_max" in user_overrides) and (user_overrides.get("T_max") is not None)
        user_provided_h_init = ("h_init" in user_overrides) and (user_overrides.get("h_init") is not None)

        base = defaults
        base.update(user_overrides)
        raw = base

    # Be forgiving with YAML/CLI configs: allow extra *metadata* keys
    # (e.g., case_name, notes) without breaking the simulation.
    allowed = {f.name for f in fields(SimulationParams)}
    extra_ok = {
        # purely descriptive / output-naming fields commonly present in YAML
        "case_name",
        "notes",
        "description",
        "title",
        "tags",
    }
    unknown = sorted(set(raw.keys()) - allowed)
    unknown_nonmeta = [k for k in unknown if k not in extra_ok]
    if unknown_nonmeta:
        logger.warning(
            "Ignoring %d unknown SimulationParams key(s): %s",
            len(unknown_nonmeta),
            ", ".join(unknown_nonmeta),
        )
    if unknown:
        raw = {k: raw[k] for k in allowed if k in raw}

    coerced = _coerce_scalar_types_for_simulation(raw)

    # Keep the time grid consistent with YAML overrides:
    # If the user set T_max and/or h_init (or T_int) but did not explicitly set `step`,
    # derive `step = ceil(T_max / h_init)` and normalise T_int accordingly.
    if (not user_provided_step) and (user_provided_T_max or user_provided_h_init or user_provided_T_int):
        try:
            if coerced.get("T_int") is not None:
                # Convention: T_int = (0.0, T_max)
                t0, t1 = coerced["T_int"]
                T_max_eff = float(t1) - float(t0)
                if T_max_eff <= 0.0:
                    T_max_eff = float(coerced.get("T_max", 0.4))
            else:
                T_max_eff = float(coerced.get("T_max", 0.4))
                coerced["T_int"] = (0.0, T_max_eff)

            h = float(coerced.get("h_init", 1e-4))
            if h > 0.0 and T_max_eff > 0.0:
                coerced["T_max"] = float(T_max_eff)
                coerced["T_int"] = (0.0, float(T_max_eff))
                coerced["step"] = int(np.ceil(T_max_eff / h))
        except Exception:
            # Never fail a run because of an auxiliary consistency correction
            pass

    # Optional consistency check: if k_train is provided in YAML,
    # verify it matches the implied elastic stiffness fy/uy.
    if coerced.get("k_train") is not None:
        try:
            k_train = np.asarray(coerced["k_train"], dtype=float)
            fy = np.asarray(coerced.get("fy"), dtype=float)
            uy = np.asarray(coerced.get("uy"), dtype=float)
            if fy.size and uy.size and np.all(uy != 0.0):
                k_eff = fy / uy
                # broadcast scalar k_train to vector if needed
                if k_train.size == 1 and k_eff.size > 1:
                    k_train = np.full_like(k_eff, float(k_train.ravel()[0]))
                if k_train.shape == k_eff.shape:
                    rel = np.max(np.abs(k_train - k_eff) / (np.abs(k_eff) + SimulationConstants.ZERO_TOL))
                    if rel > 0.02:  # >2% mismatch
                        logger.warning(
                            "k_train provided (YAML) differs from implied fy/uy by up to %.1f%%. "
                            "Simulation uses fy & uy; consider removing k_train or adjusting values.",
                            100.0 * rel,
                        )
        except Exception:
            # Never fail a run because of an auxiliary check
            pass
    sim_params = SimulationParams(**coerced)

    simulator = ImpactSimulator(sim_params)
    return simulator.run()

def _coerce_scalar_types_for_simulation(base: dict) -> dict:
    """
    Normalize types coming from YAML/JSON before constructing SimulationParams.

    - Convert scalar fields that should be floats (incl. things that may come
      as strings like '1.0e5', '6.0e7') to float.
    - Convert integer fields to int.
    - Convert list-like fields for masses/positions/stiffnesses to float arrays.
    - Normalise T_int to a 2-tuple of floats (default: (0.0, T_max)).
    """
    data: dict = dict(base)  # shallow copy

    def _to_float(val, name: str):
        if val is None:
            return None
        if isinstance(val, (float, int)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError as exc:
                raise ValueError(
                    f"Parameter '{name}' expects a float-compatible value, "
                    f"got {val!r} (type {type(val).__name__})."
                ) from exc
        # Para cualquier cosa exótica, fallar fuerte y claro
        raise TypeError(
            f"Parameter '{name}' expects a scalar float, got {val!r} "
            f"(type {type(val).__name__})."
        )

    def _to_int(val, name: str):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError as exc:
                raise ValueError(
                    f"Parameter '{name}' expects an int-compatible value, "
                    f"got {val!r} (type {type(val).__name__})."
                ) from exc
        raise TypeError(
            f"Parameter '{name}' expects an int, got {val!r} "
            f"(type {type(val).__name__})."
        )

    # --- Scalars that should be floats ---------------------------------
    scalar_float_keys = [
        # geometry / kinematics
        "v0_init",
        "angle_rad",
        "d0",
        # wall / contact
        "k_wall",
        "cr_wall",
        # building
        "building_mass",
        "building_zeta",
        "building_height",
        "building_uy",
        "building_uy_mm",
        "building_alpha",
        "building_gamma",
        # friction
        "mu_s",
        "mu_k",
        "sigma_0",
        "sigma_1",
        "sigma_2",
        # Bouc-Wen
        "bw_a",
        "bw_A",
        "bw_beta",
        "bw_gamma",
        # HHT and solver
        "alpha_hht",
        "newton_tol",
        "h_init",
        "T_max",
    ]

    for key in scalar_float_keys:
        if key in data and data[key] is not None:
            data[key] = _to_float(data[key], key)

    # --- Scalars that should be ints ------------------------------------
    int_keys = [
        "n_masses",
        "max_iter",
        "step",
        "bw_n",
    ]

    for key in int_keys:
        if key in data and data[key] is not None:
            data[key] = _to_int(data[key], key)

    # --- Arrays: coerce entries to float (handles '8.0e6' strings) ------
    array_float_keys = [
        "masses",
        "x_init",
        "y_init",
        "fy",
        "uy",
        "k_train",
    ]

    for key in array_float_keys:
        if key in data and data[key] is not None:
            data[key] = np.asarray(data[key], dtype=float)

    # --- Time interval T_int --------------------------------------------
    if "T_int" in data and data["T_int"] is not None:
        T_int = data["T_int"]
        if isinstance(T_int, (list, tuple)) and len(T_int) == 2:
            t0, t1 = T_int
            data["T_int"] = (float(t0), float(t1))
        else:
            raise ValueError(
                f"Parameter 'T_int' must be a 2-tuple/list, got {T_int!r}"
            )
    else:
        # If not provided, derive from T_max when available
        if "T_max" in data and data["T_max"] is not None:
            data["T_int"] = (0.0, float(data["T_max"]))

    return data
