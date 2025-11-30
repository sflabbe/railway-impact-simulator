"""
Numerical engine for the Railway Impact Simulator.

This module is UI-agnostic: it contains the HHT-α time integration,
Bouc–Wen hysteresis, friction and contact laws, and the main
ImpactSimulator class.

Use from your Streamlit app (or tests) as:

    from core.engine import SimulationParams, run_simulation

    sim_params = SimulationParams(**params_dict)
    df = run_simulation(sim_params)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.constants import g as GRAVITY


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
        if abs(uy) < 1e-12:
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

        return x0 + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

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
        """LuGre friction model."""
        v_stribeck = max(abs(v_stribeck), 1e-10)
        g = (F_coulomb + (F_static - F_coulomb) *
             np.exp(-(v / v_stribeck) ** 2)) / sigma_0

        z_dot = v - np.abs(v) * z_prev / g
        z = z_prev + z_dot * h
        F = sigma_0 * z + sigma_1 * z_dot + sigma_2 * v

        return F, z

    @staticmethod
    def dahl(z_prev: float, v: float, F_coulomb: float,
             sigma_0: float, h: float) -> Tuple[float, float]:
        """Dahl friction model."""
        Fc = max(abs(F_coulomb), 1e-12)
        z_dot = (1.0 - sigma_0 / Fc * z_prev * np.sign(v)) * v
        z = z_prev + z_dot * h
        F = sigma_0 * z

        return F, z

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

            if L0 == 0:
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
        positive = np.real(eigenvalues[np.real(eigenvalues) > 1e-12])
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
        C: np.ndarray,
        qp: np.ndarray,
        qp_old: np.ndarray
    ) -> np.ndarray:
        """
        Compute acceleration using HHT-α method.

        M * a_new = (1-α)*R_new + α*R_old - (1-α)*C*v_new - α*C*v_old
        """
        R_total_new = R_internal + R_contact + R_friction
        R_total_old = R_internal_old + R_contact_old + R_friction_old

        force = (
            (1.0 - self.alpha) * R_total_new +
            self.alpha * R_total_old -
            (1.0 - self.alpha) * (C @ qp) -
            self.alpha * (C @ qp_old)
        )

        return np.linalg.solve(M, force)


# ====================================================================
# MAIN SIMULATION ENGINE
# ====================================================================

class ImpactSimulator:
    """Main HHT-α simulation engine with Newton iterations."""

    def __init__(self, params: SimulationParams):
        self.params = params
        self.setup()

    # ----------------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------------
    def setup(self):
        """Initialize simulation matrices and state."""
        p = self.params

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

        xp_init = np.full(p.n_masses, p.v0_init)
        yp_init = np.zeros(p.n_masses)
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

        # HHT integrator
        self.integrator = HHTAlphaIntegrator(p.alpha_hht)

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

        # Hysteretic and friction internal states
        X_bw = np.zeros((n - 1, p.step + 1))
        z_friction = np.zeros((n, p.step + 1))

        # Contact tracking
        contact_active = False
        v0_contact = np.ones(dof)

        # Normal forces for friction (per node)
        FN_node = GRAVITY * p.masses
        vs = 1.0e-3  # Stribeck reference velocity

        # Energy bookkeeping
        E_kin = np.zeros(p.step + 1)
        E_spring = np.zeros(p.step + 1)
        E_contact = np.zeros(p.step + 1)
        E_damp_rayleigh = np.zeros(p.step + 1)
        E_fric = np.zeros(p.step + 1)

        # Initial kinetic energy
        v0 = qp[:, 0]
        E_kin[0] = 0.5 * float(v0.T @ self.M @ v0)

        # Time stepping
        for step_idx in range(p.step):

            # Initial guess for a_{n+1}
            qpp[:, step_idx + 1] = qpp[:, step_idx]

            converged = False
            err = np.inf

            for _ in range(p.max_iter):
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

                R_spring_nodal, X_bw[:, step_idx + 1] = BoucWenModel.compute_forces(
                    -u_spring[:, step_idx + 1],  # compression positive
                    -du,
                    X_bw[:, step_idx],
                    self.params.uy,
                    self.params.fy,
                    self.h,
                    self.params.bw_a,
                    self.params.bw_A,
                    self.params.bw_beta,
                    self.params.bw_gamma,
                    self.params.bw_n,
                )

                R_internal[:, step_idx + 1] = np.concatenate(
                    [R_spring_nodal, np.zeros(n)]
                )

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

                # --- Corrector: update acceleration ---
                qpp_old = qpp[:, step_idx + 1].copy()

                qpp[:, step_idx + 1] = self.integrator.compute_acceleration(
                    self.M,
                    R_internal[:, step_idx + 1], R_internal[:, step_idx],
                    R_contact[:, step_idx + 1], R_contact[:, step_idx],
                    R_friction[:, step_idx + 1], R_friction[:, step_idx],
                    self.C,
                    qp[:, step_idx + 1],
                    qp[:, step_idx],
                )

                # Convergence check
                err = self._check_convergence(qpp[:, step_idx + 1], qpp_old)
                if err < self.params.newton_tol:
                    converged = True
                    break

            if not converged:
                # No Streamlit here; just print to console/log
                print(
                    f"[ImpactSimulator] Newton did not converge at step {step_idx} "
                    f"(rel Δa = {err:.3e})"
                )

            # --------------------------------------------------
            # Energy bookkeeping (after convergence)
            # --------------------------------------------------
            v_new = qp[:, step_idx + 1]
            v_old = qp[:, step_idx]
            v_mid = 0.5 * (v_old + v_new)

            # 1) Kinetic energy
            E_kin[step_idx + 1] = 0.5 * float(v_new.T @ self.M @ v_new)

            # 2) Elastic energy in train springs
            E_spring[step_idx + 1] = 0.5 * float(
                np.sum(self.k_lin * u_spring[:, step_idx + 1] ** 2)
            )

            # 3) Elastic energy in wall contact
            delta = np.maximum(-u_contact[:n, step_idx + 1], 0.0)
            model = self.params.contact_model.lower()
            if model in ["hooke", "ye", "pant-wijeyewickrema", "anagnostopoulos"]:
                exp = 1.0
            else:
                exp = 1.5

            if np.any(delta > 0.0):
                if exp == 1.0:
                    E_contact[step_idx + 1] = 0.5 * self.params.k_wall * float(np.sum(delta ** 2))
                else:
                    E_contact[step_idx + 1] = (
                        self.params.k_wall / (exp + 1.0) * float(np.sum(delta ** (exp + 1.0)))
                    )
            else:
                E_contact[step_idx + 1] = 0.0

            # 4) Rayleigh damping loss
            p_damp = float(v_mid.T @ self.C @ v_mid)
            dE_damp = max(p_damp, 0.0) * self.h
            E_damp_rayleigh[step_idx + 1] = E_damp_rayleigh[step_idx] + dE_damp

            # 5) Friction loss
            F_fric = R_friction[:, step_idx + 1]
            p_fric = -float(np.dot(F_fric, v_mid))
            dE_fric = max(p_fric, 0.0) * self.h
            E_fric[step_idx + 1] = E_fric[step_idx] + dE_fric

        energies = {
            "E_kin": E_kin,
            "E_spring": E_spring,
            "E_contact": E_contact,
            "E_damp_rayleigh": E_damp_rayleigh,
            "E_fric": E_fric,
        }

        return self._build_results_dataframe(
            q, qp, qpp, R_contact, u_contact, X_bw, energies
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
    ):
        """
        Compute friction forces per mass node and distribute to x/z DOFs.
        """
        p = self.params
        n = p.n_masses
        dof = 2 * n

        assert FN_node.shape[0] == n
        assert qp.shape[0] == dof

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
                    # Default: LuGre
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
                v0_contact = np.where(du_contact < 0.0, du_contact, 1.0)
                v0_contact[v0_contact == 0.0] = 1.0

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

    @staticmethod
    def _check_convergence(qpp_new: np.ndarray, qpp_old: np.ndarray) -> float:
        """Relative change in acceleration."""
        delta = qpp_new - qpp_old
        norm_new = np.linalg.norm(qpp_new) + 1e-16
        return np.linalg.norm(delta) / norm_new

    def _build_results_dataframe(
        self,
        q: np.ndarray,
        qp: np.ndarray,
        qpp: np.ndarray,
        R_contact: np.ndarray,
        u_contact: np.ndarray,
        X_bw: np.ndarray,
        energies: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Build results DataFrame for export (including energy bookkeeping)."""
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

        E_kin = energies["E_kin"]
        E_spring = energies["E_spring"]
        E_contact = energies["E_contact"]
        E_damp_rayleigh = energies["E_damp_rayleigh"]
        E_fric = energies["E_fric"]

        E_mech = E_kin + E_spring + E_contact
        E_diss_tracked = E_damp_rayleigh + E_fric
        E_total_tracked = E_mech + E_diss_tracked
        E_initial = float(E_total_tracked[0]) if len(E_total_tracked) > 0 else 0.0
        E_balance_error = E_total_tracked - E_initial

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
                # Energies in Joules
                "E_kin_J": E_kin,
                "E_spring_J": E_spring,
                "E_contact_J": E_contact,
                "E_damp_rayleigh_J": E_damp_rayleigh,
                "E_friction_J": E_fric,
                "E_mech_J": E_mech,
                "E_diss_tracked_J": E_diss_tracked,
                "E_total_tracked_J": E_total_tracked,
                "E_total_initial_J": np.full_like(self.t, E_initial, dtype=float),
                "E_balance_error_J": E_balance_error,
            }
        )

        return df


# ====================================================================
# PUBLIC ENTRY POINT
# ====================================================================

def run_simulation(params: SimulationParams | Dict[str, Any]) -> pd.DataFrame:
    """
    High-level convenience wrapper.

    Parameters
    ----------
    params : SimulationParams or dict
        If a dict is passed, it must contain the same keys as SimulationParams.

    Returns
    -------
    DataFrame with time history, contact force, penetration, acceleration
    and energy bookkeeping.
    """
    if isinstance(params, SimulationParams):
        sim_params = params
    else:
        sim_params = SimulationParams(**params)

    simulator = ImpactSimulator(sim_params)
    return simulator.run()
