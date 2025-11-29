"""
Railway Impact Simulator
========================
HHT-α Implicit Integration with Bouc-Wen Hysteresis

Copyright (c) 2025 Sebastián Labbé
Licensed under the MIT License. See LICENSE file in the project root for full license text.

This module simulates train impacts on rigid barriers using advanced
numerical integration and material models.

Code Implementation by:
- Sebastián Labbé, Dipl.-Ing. (Karlsruher Institut für Technologie - KIT)

Based on the research report:
"Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr"
(Review and Adjustment of Impact Loads from Railway Traffic)

Original Research Authors:
- Univ.-Prof. Dr.-Ing. Lothar Stempniewski (Karlsruhe Institute of Technology - KIT)
- Dipl.-Ing. Sebastián Labbé (Karlsruhe Institute of Technology - KIT)
- Dr.-Ing. Steffen Siegel (Siegel und Wünschel beratende Ingenieure PartG mbB)
- Robin Bosch, M.Sc. (Siegel und Wünschel beratende Ingenieure PartG mbB)

Research Institutions:
- Karlsruher Institut für Technologie (KIT)
  Institut für Massivbau und Baustofftechnologie, Abteilung Massivbau
  Gotthard-Franz-Straße 3, 76131 Karlsruhe
  
- Siegel und Wünschel beratende Ingenieure Partnerschaftsgesellschaft mbB
  Zehntwiesenstraße 35a, 76275 Ettlingen

Commissioned by:
- Eisenbahn-Bundesamt (EBA) - Federal Railway Authority

Published by:
- Deutsches Zentrum für Schienenverkehrsforschung (DZSF)
  German Centre for Rail Transport Research

Reference:
DZSF Bericht 53 (2024)
Project Number: 2018-08-U-1217
Study Completion: June 2021
Publication: June 2024

DOI: 10.48755/dzsf.240006.01
ISSN: 2629-7973
License: CC BY 4.0

Download:
https://www.dzsf.bund.de/SharedDocs/Downloads/DZSF/Veroeffentlichungen/Forschungsberichte/2024/ForBe_53_2024_Anpralllasten.pdf
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.constants import g as GRAVITY
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import io
import base64
from pathlib import Path


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
    
    # Material properties
    fy: np.ndarray
    uy: np.ndarray
    
    # Contact
    k_wall: float
    cr_wall: float
    contact_model: str
    
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
    step: int
    T_int: Tuple[float, float]


@dataclass
class TrainConfig:
    """Train configuration parameters."""
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
        return (A - np.abs(x)**n * (beta + np.sign(u * x) * gamma)) * u / uy
    
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
        
        return x0 + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
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
             np.exp(-(v / v_stribeck)**2)) / sigma_0
        
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
        return (Fc + (Fs - Fc) * np.exp(-(v_abs / vs)**2)) * np.sign(v) + Fv * v
    
    @staticmethod
    def brown_mcphee(v: float, Fc: float, Fs: float, vs: float) -> float:
        """Brown & McPhee friction model."""
        vs = max(vs, 1e-6)
        v_abs = np.abs(v)
        x = v_abs / vs
        
        term1 = Fc * np.tanh(4.0 * x)
        term2 = (Fs - Fc) * x / ((0.25 * x**2 + 0.75)**2)
        
        return (term1 + term2) * np.sign(v)


# ====================================================================
# CONTACT MODELS
# ====================================================================

class ContactModels:
    """Normal contact force model implementations.
    
    References:
    - Anagnostopoulos (1988, 2004): Linear viscoelastic (Kelvin-Voigt) model
    - Hunt & Crossley (1975): Hertz contact with damping
    - Lankarani & Nikravesh (1990): Modified Hunt-Crossley
    - Flores et al. (2006): Energy-based dissipation
    - Gonthier et al. (2004): Compliant contact model
    - Ye et al. (2009): Linear spring with damping
    - Pant & Wijeyewickrema (2012): Enhanced linear model
    """
    
    MODELS = {
        'hooke': lambda k, d, cr, dv, v0: -k * d,
        'hertz': lambda k, d, cr, dv, v0: -k * d**1.5,
        'hunt-crossley': lambda k, d, cr, dv, v0: (
            -k * d**1.5 * (1.0 + 3.0*(1.0 - cr)/2.0 * (dv / v0))
        ),
        'lankarani-nikravesh': lambda k, d, cr, dv, v0: (
            -k * d**1.5 * (1.0 + 3.0*(1.0 - cr**2)/4.0 * (dv / v0))
        ),
        'flores': lambda k, d, cr, dv, v0: (
            -k * d**1.5 * (1.0 + 8.0*(1.0 - cr)/(5.0*cr) * (dv / v0))
        ),
        'gonthier': lambda k, d, cr, dv, v0: (
            -k * d**1.5 * (1.0 + (1.0 - cr**2)/cr * (dv / v0))
        ),
        'ye': lambda k, d, cr, dv, v0: (
            -k * d * (1.0 + 3.0*(1.0 - cr)/(2.0*cr) * (dv / v0))
        ),
        'pant-wijeyewickrema': lambda k, d, cr, dv, v0: (
            -k * d * (1.0 + 3.0*(1.0 - cr**2)/(2.0*cr**2) * (dv / v0))
        ),
        'anagnostopoulos': lambda k, d, cr, dv, v0: (
            -k * d * (1.0 + 3.0*(1.0 - cr)/(2.0*cr) * (dv / v0))
        ),
    }
    
    @staticmethod
    def compute_force(u_contact: np.ndarray, du_contact: np.ndarray,
                     v0: np.ndarray, k_wall: float, cr_wall: float,
                     model: str) -> np.ndarray:
        """
        Compute normal contact forces.
        
        Args:
            u_contact: Penetration (negative when in contact)
            du_contact: Penetration velocity
            v0: Initial contact velocity
            k_wall: Wall stiffness
            cr_wall: Coefficient of restitution
            model: Contact model name
        
        Available models:
            - hooke: Linear elastic (F = k·δ)
            - hertz: Hertzian contact (F = k·δ^1.5)
            - hunt-crossley: Hertz with Hunt-Crossley damping
            - lankarani-nikravesh: Modified Hunt-Crossley
            - flores: Energy-based dissipation
            - gonthier: Compliant contact with restitution
            - ye: Linear with Ye damping
            - pant-wijeyewickrema: Enhanced linear damping
            - anagnostopoulos: Linear viscoelastic (Kelvin-Voigt)
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
        v0m = np.where(np.abs(v0m) < 1e-8, 
                       np.sign(v0m) * 1e-8 + (v0m == 0) * 1e-8, 
                       v0m)
        
        # Get model function (default to anagnostopoulos)
        model_func = ContactModels.MODELS.get(
            model.lower(), 
            ContactModels.MODELS['anagnostopoulos']
        )
        
        R[mask] = model_func(k_wall, d, cr_wall, dv, v0m)
        
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
        
        n_masses = (config.mass_points_lok + 
                   config.n_wagons * config.mass_points_wagon)
        
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
    def build_stiffness_matrix_2d(n: int, x: np.ndarray, y: np.ndarray, 
                                   k: np.ndarray) -> np.ndarray:
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
                [ cx*cx,  cx*cy, -cx*cx, -cx*cy],
                [ cx*cy,  cy*cy, -cx*cy, -cy*cy],
                [-cx*cx, -cx*cy,  cx*cx,  cx*cy],
                [-cx*cy, -cy*cy,  cx*cy,  cy*cy]
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
# TIME INTEGRATION
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
        self.beta = 0.25 * (1.0 + alpha)**2
        self.gamma = 0.5 + alpha
    
    def predict(self, q: np.ndarray, qp: np.ndarray, qpp: np.ndarray, 
                qpp_new: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Newmark predictor step.
        
        Returns:
            q_new: Predicted displacement
            qp_new: Predicted velocity
        """
        q_new = (q + h * qp + 
                (0.5 - self.beta) * h**2 * qpp + 
                self.beta * h**2 * qpp_new)
        
        qp_new = (qp + 
                 (1.0 - self.gamma) * h * qpp + 
                 self.gamma * h * qpp_new)
        
        return q_new, qp_new
    
    def compute_acceleration(self, M: np.ndarray, R_internal: np.ndarray,
                            R_internal_old: np.ndarray, R_contact: np.ndarray,
                            R_contact_old: np.ndarray, R_friction: np.ndarray,
                            R_friction_old: np.ndarray, C: np.ndarray,
                            qp: np.ndarray, qp_old: np.ndarray) -> np.ndarray:
        """
        Compute acceleration using HHT-α method.
        
        M * a_new = (1-α)*R_new + α*R_old - (1-α)*C*v_new - α*C*v_old
        """
        R_total_new = R_internal + R_contact + R_friction
        R_total_old = R_internal_old + R_contact_old + R_friction_old
        
        force = ((1.0 - self.alpha) * R_total_new + 
                self.alpha * R_total_old -
                (1.0 - self.alpha) * (C @ qp) - 
                self.alpha * (C @ qp_old))
        
        return np.linalg.solve(M, force)


# ====================================================================
# MAIN SIMULATION
# ====================================================================

class ImpactSimulator:
    """Main simulation engine."""
    
    def __init__(self, params: SimulationParams):
        self.params = params
        self.setup()
    
    def setup(self):
        """Initialize simulation matrices and state."""
        p = self.params
        
        # Time discretization
        self.h = (p.T_int[1] - p.T_int[0]) / p.step
        self.t = np.linspace(p.T_int[0], p.T_int[1], p.step + 1)
        
        # Rotation matrix for impact angle
        c = np.cos(p.angle_rad)
        s = np.sin(p.angle_rad)
        self.rot = np.array([[c, -s], [s, c]])
        
        # Transform initial conditions
        xy_init = self.rot @ np.vstack([p.x_init, p.y_init])
        self.x_init = xy_init[0, :]
        self.y_init = xy_init[1, :]
        
        xp_init = np.full(p.n_masses, p.v0_init)
        yp_init = np.zeros(p.n_masses)
        xpyp_init = self.rot @ np.vstack([xp_init, yp_init])
        self.xp_init = xpyp_init[0, :]
        self.yp_init = xpyp_init[1, :]
        
        # Mass matrix
        self.M = np.diag(np.concatenate([p.masses, p.masses]))
        
        # Stiffness matrix
        k_init = p.fy / p.uy
        self.K = StructuralDynamics.build_stiffness_matrix_2d(
            p.n_masses, self.x_init, self.y_init, k_init
        )
        
        # Damping matrix
        self.C = StructuralDynamics.compute_rayleigh_damping(self.M, self.K)
        
        # Initial spring lengths
        self.u10 = np.zeros(p.n_masses - 1)
        for i in range(p.n_masses - 1):
            dx = self.x_init[i + 1] - self.x_init[i]
            dy = self.y_init[i + 1] - self.y_init[i]
            self.u10[i] = np.hypot(dx, dy)
        
        # Initialize integrator
        self.integrator = HHTAlphaIntegrator(p.alpha_hht)
    
    def run(self) -> pd.DataFrame:
        """
        Execute time-stepping simulation.
        
        Returns:
            DataFrame with time history results
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
        
        # Force tracking
        u_spring = np.zeros((n - 1, p.step + 1))
        u_contact = np.zeros((dof, p.step + 1))
        R_contact = np.zeros((dof, p.step + 1))
        R_friction = np.zeros((dof, p.step + 1))
        R_internal = np.zeros((dof, p.step + 1))
        
        # Hysteretic states
        X_bw = np.zeros((n - 1, p.step + 1))
        z_friction = np.zeros((dof, p.step + 1))
        
        # Contact tracking
        contact_active = False
        v0_contact = np.ones(dof)
        
        # Normal forces for friction
        FN = GRAVITY * (self.M @ np.ones(dof))
        
        vs = 0.001  # Stribeck velocity
        
        # Time-stepping loop
        for step_idx in range(p.step):
            qpp[:, step_idx + 1] = qpp[:, step_idx].copy()
            
            converged = False
            for iteration in range(p.max_iter):
                # Newmark predictor
                q[:, step_idx + 1], qp[:, step_idx + 1] = self.integrator.predict(
                    q[:, step_idx], qp[:, step_idx], 
                    qpp[:, step_idx], qpp[:, step_idx + 1], 
                    self.h
                )
                
                # Compute spring deformations (2D)
                for i in range(n - 1):
                    r1 = q[[i, n + i], step_idx + 1]
                    r2 = q[[i + 1, n + i + 1], step_idx + 1]
                    u_spring[i, step_idx + 1] = np.linalg.norm(r2 - r1) - self.u10[i]
                
                # Spring deformation rate
                if step_idx > 0:
                    du = (u_spring[:, step_idx + 1] - u_spring[:, step_idx]) / self.h
                else:
                    du = np.zeros(n - 1)
                
                # Internal forces (Bouc-Wen)
                R_spring_nodal, X_bw[:, step_idx + 1] = BoucWenModel.compute_forces(
                    -u_spring[:, step_idx + 1], -du, X_bw[:, step_idx],
                    p.uy, p.fy, self.h, p.bw_a, p.bw_A, 
                    p.bw_beta, p.bw_gamma, p.bw_n
                )
                
                R_internal[:, step_idx + 1] = np.concatenate([
                    R_spring_nodal, np.zeros(n)
                ])
                
                # Friction forces
                self._compute_friction(
                    step_idx, qp[:, step_idx + 1], FN, z_friction, 
                    R_friction, vs
                )
                
                # Contact forces
                contact_active, v0_contact = self._compute_contact(
                    step_idx, q[:, step_idx + 1], qp[:, step_idx + 1],
                    u_contact, R_contact, contact_active, v0_contact
                )
                
                # Corrector: update acceleration
                qpp_old = qpp[:, step_idx + 1].copy()
                
                qpp[:, step_idx + 1] = self.integrator.compute_acceleration(
                    self.M,
                    R_internal[:, step_idx + 1], R_internal[:, step_idx],
                    R_contact[:, step_idx + 1], R_contact[:, step_idx],
                    R_friction[:, step_idx + 1], R_friction[:, step_idx],
                    self.C, qp[:, step_idx + 1], qp[:, step_idx]
                )
                
                # Convergence check
                err = self._check_convergence(qpp[:, step_idx + 1], qpp_old)
                
                if err < p.newton_tol:
                    converged = True
                    break
            
            if not converged:
                st.warning(f"⚠️ Newton iteration did not converge at step {step_idx}")
        
        # Post-process results
        return self._build_results_dataframe(q, qp, qpp, R_contact, u_contact, X_bw)
    
    def _compute_friction(self, step_idx: int, qp: np.ndarray, FN: np.ndarray,
                         z_friction: np.ndarray, R_friction: np.ndarray, 
                         vs: float):
        """Compute friction forces."""
        p = self.params
        dof = 2 * p.n_masses
        
        Fc = p.mu_k * np.abs(FN)
        Fs = p.mu_s * np.abs(FN)
        
        for i in range(dof):
            v = qp[i]
            
            if p.friction_model == 'dahl':
                F, z = FrictionModels.dahl(
                    z_friction[i, step_idx], v, Fc[i], p.sigma_0, self.h
                )
            elif p.friction_model == 'lugre':
                F, z = FrictionModels.lugre(
                    z_friction[i, step_idx], v, Fc[i], Fs[i], vs,
                    p.sigma_0, p.sigma_1, p.sigma_2, self.h
                )
            elif p.friction_model == 'coulomb':
                F = FrictionModels.coulomb_stribeck(v, Fc[i], Fs[i], vs, p.sigma_2)
                z = z_friction[i, step_idx]
            elif p.friction_model == 'brown-mcphee':
                F = FrictionModels.brown_mcphee(v, Fc[i], Fs[i], vs)
                z = z_friction[i, step_idx]
            else:
                F, z = FrictionModels.lugre(
                    z_friction[i, step_idx], v, Fc[i], Fs[i], vs,
                    p.sigma_0, p.sigma_1, p.sigma_2, self.h
                )
            
            R_friction[i, step_idx + 1] = -F
            z_friction[i, step_idx + 1] = z
    
    def _compute_contact(self, step_idx: int, q: np.ndarray, qp: np.ndarray,
                        u_contact: np.ndarray, R_contact: np.ndarray,
                        contact_active: bool, v0_contact: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Compute contact forces at wall and update contact state."""
        p = self.params
        n = p.n_masses
        dof = 2 * n
        
        # Reset penetration for this step (important)
        u_contact[:, step_idx + 1] = 0.0
        
        # Check for contact (x < 0)
        if np.any(q[:n] < 0.0):
            # Set penetrations (negative displacements)
            for i in range(n):
                if q[i] < 0.0:
                    u_contact[i, step_idx + 1] = q[i]
            
            # Penetration rate
            if step_idx > 0:
                du_contact = (u_contact[:, step_idx + 1] - 
                             u_contact[:, step_idx]) / self.h
            else:
                du_contact = np.zeros(dof)
            
            # Detect first contact to freeze v0_contact
            if (not contact_active) and np.any(du_contact[:n] < 0.0):
                contact_active = True
                v0_contact = np.where(du_contact < 0.0, du_contact, 1.0)
                v0_contact[v0_contact == 0.0] = 1.0  # avoid division by zero
            
            # Contact forces
            R = ContactModels.compute_force(
                u_contact[:, step_idx + 1], du_contact, v0_contact,
                p.k_wall, p.cr_wall, p.contact_model
            )
            
            R_contact[:, step_idx + 1] = -R
        
        # Loss of contact: reset state
        elif step_idx > 0 and np.any(u_contact[:n, step_idx] < 0.0):
            contact_active = False
            v0_contact = np.ones_like(v0_contact)
        
        return contact_active, v0_contact
    
    def _check_convergence(self, qpp_new: np.ndarray, qpp_old: np.ndarray) -> float:
        """Check Newton iteration convergence."""
        delta = qpp_new - qpp_old
        norm_new = np.linalg.norm(qpp_new) + 1e-16
        return np.linalg.norm(delta) / norm_new
    
    def _build_results_dataframe(self, q: np.ndarray, qp: np.ndarray, 
                                 qpp: np.ndarray, R_contact: np.ndarray,
                                 u_contact: np.ndarray, X_bw: np.ndarray) -> pd.DataFrame:
        """Build results DataFrame for export."""
        F_total = R_contact[0, :]  # Positive reaction force at the wall (MN)
        u_pen = -u_contact[0, :] * 1000  # Penetration in mm
        a_front = qpp[0, :] / GRAVITY  # Acceleration in g
        
        df = pd.DataFrame({
            'Time_s': self.t,
            'Time_ms': self.t * 1000.0,
            'Impact_Force_MN': F_total / 1e6,
            'Penetration_mm': u_pen,
            'Acceleration_g': a_front,
            'Velocity_m_s': qp[0, :],
            'Position_x_m': q[0, :],
            'BoucWen_State_1': X_bw[0, :] if len(X_bw) > 0 else np.zeros_like(self.t),
        })
        
        return df


# ====================================================================
# UTILITIES
# ====================================================================

def display_header():
    """Display header with institutional logos and research information."""
    
    # Create columns for logos
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### KIT")
        st.markdown("**Karlsruher Institut für Technologie**")
    
    with col2:
        st.markdown("### EBA")
        st.markdown("**Eisenbahn-Bundesamt**")
    
    with col3:
        st.markdown("### DZSF")
        st.markdown("**Deutsches Zentrum für Schienenverkehrsforschung**")
    
    st.markdown("---")
    
    # Research information
    st.markdown("""
    ### Research Background
    
    **Report Title (German):**  
    *Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr*
    
    **Report Title (English):**  
    *Review and Adjustment of Impact Loads from Railway Traffic*
    
    **Authors:**
    - Univ.-Prof. Dr.-Ing. Lothar Stempniewski (KIT)
    - Dipl.-Ing. Sebastián Labbé (KIT)
    - Dr.-Ing. Steffen Siegel (Siegel und Wünschel PartG mbB)
    - Robin Bosch, M.Sc. (Siegel und Wünschel PartG mbB)
    
    **Research Institutions:**
    - Karlsruher Institut für Technologie (KIT)  
      Institut für Massivbau und Baustofftechnologie
    - Siegel und Wünschel beratende Ingenieure PartG mbB, Ettlingen
    
    **Publication:**  
    DZSF Bericht 53 (2024)  
    Project Number: 2018-08-U-1217  
    Study Completion: June 2021  
    Publication Date: June 2024
    
    **DOI:** [10.48755/dzsf.240006.01](https://doi.org/10.48755/dzsf.240006.01)  
    **ISSN:** 2629-7973  
    **License:** CC BY 4.0
    
    **Download Report:**  
    [DZSF Forschungsbericht 53/2024 (PDF)](https://www.dzsf.bund.de/SharedDocs/Downloads/DZSF/Veroeffentlichungen/Forschungsberichte/2024/ForBe_53_2024_Anpralllasten.pdf?__blob=publicationFile&v=2)
    
    **Commissioned by:**  
    Eisenbahn-Bundesamt (EBA)
    
    **Published by:**  
    Deutsches Zentrum für Schienenverkehrsforschung (DZSF)
    """)
    
    st.markdown("---")


def to_excel(df: pd.DataFrame) -> bytes:
    """Generate Excel file for download."""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Dynamic Load History')
    except ImportError:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Dynamic Load History')
    return output.getvalue()


def create_results_plots(df: pd.DataFrame) -> go.Figure:
    """Create comprehensive results visualization."""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Force', 'Penetration', 'Acceleration', 'Hysteresis'),
        vertical_spacing=0.08
    )
    
    # Force vs Time
    fig.add_trace(
        go.Scatter(
            x=df['Time_ms'],
            y=df['Impact_Force_MN'],
            line=dict(width=2, color='#1f77b4')
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Force (MN)", row=1, col=1)
    
    # Penetration vs Time
    fig.add_trace(
        go.Scatter(
            x=df['Time_ms'],
            y=df['Penetration_mm'],
            line=dict(width=2, color='#ff7f0e')
        ),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Penetration (mm)", row=2, col=1)
    
    # Acceleration vs Time
    fig.add_trace(
        go.Scatter(
            x=df['Time_ms'],
            y=df['Acceleration_g'],
            line=dict(width=2, color='#2ca02c')
        ),
        row=3, col=1
    )
    fig.update_yaxes(title_text="Acceleration (g)", row=3, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=3, col=1)
    
    # Hysteresis Loop
    fig.add_trace(
        go.Scatter(
            x=df['Penetration_mm'],
            y=df['Impact_Force_MN'],
            mode='markers',
            marker=dict(
                size=3,
                color=df['Time_ms'],
                colorscale='Viridis',
                colorbar=dict(title="Time (ms)", x=1.1)
            ),
        ),
        row=4, col=1
    )
    fig.update_xaxes(title_text="Penetration (mm)", row=4, col=1)
    fig.update_yaxes(title_text="Force (MN)", row=4, col=1)
    
    fig.update_layout(height=1400, showlegend=False)
    
    return fig


# ====================================================================
# STREAMLIT UI
# ====================================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        layout="wide", 
        page_title="Railway Impact Simulator - DZSF Research",
        page_icon="🚂"
    )
    
    st.title("🚂 Railway Impact Simulator")
    st.markdown("**HHT-α Implicit Integration with Bouc-Wen Hysteresis**")
    
    # Display institutional header and research information
    display_header()
    
    # Sidebar parameters
    params = build_parameter_ui()
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Configuration")
        st.metric("Velocity", f"{-params['v0_init'] * 3.6:.1f} km/h")
        st.metric("Masses", params['n_masses'])
        st.metric("Time Steps", params['step'])
        st.markdown("---")
        run_btn = st.button("▶️ **Run Simulation**", type="primary", 
                           use_container_width=True)
    
    with col2:
        if run_btn:
            execute_simulation(params)
        else:
            st.info("👈 Configure parameters and press **Run Simulation**")


def build_parameter_ui() -> Dict[str, Any]:
    """Build parameter input UI in sidebar."""
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        params = {}
        
        # Time & Integration
        with st.expander("🕐 Time & Integration", expanded=True):
            v0_kmh = st.slider("Impact Velocity (km/h)", 10, 200, 56, 1)
            params['v0_init'] = -v0_kmh / 3.6
            
            T_final = st.number_input("Simulation Time (s)", 0.01, 1.0, 0.3, 0.01)
            params['step'] = st.number_input("Time Steps", 1000, 200000, 10000, 1000)
            params['T_int'] = (0.0, T_final)
            
            angle_deg = st.number_input("Impact Angle (°)", 0.0, 45.0, 0.0, 0.1)
            params['angle_rad'] = angle_deg * np.pi / 180
            
            params['alpha_hht'] = st.slider(
                "HHT-α parameter", -0.3, 0.0, -0.1, 0.01,
                help="Negative values add numerical damping"
            )
            
            params['newton_tol'] = st.number_input(
                "Convergence tolerance", 1e-8, 1e-2, 1e-4, format="%.1e"
            )
            
            params['max_iter'] = st.number_input(
                "Max iterations", 5, 100, 50, 1
            )
        
        # Train Geometry
        train_params = build_train_geometry_ui()
        params.update(train_params)
        
        # Material Properties
        material_params = build_material_ui(params['n_masses'])
        params.update(material_params)
        
        # Contact & Friction
        contact_params = build_contact_friction_ui()
        params.update(contact_params)
    
    return params


def build_train_geometry_ui() -> Dict[str, Any]:
    """Build train geometry UI."""
    with st.expander("🚃 Train Geometry", expanded=True):
        config_mode = st.radio(
            "Configuration mode",
            ("Research locomotive model", "Example trains"),
            index=0
        )
        
        if config_mode == "Research locomotive model":
            n_masses = st.slider("Number of Masses", 2, 20, 7)
            
            # Default 7-mass Pioneer wagon validation model
            # Based on Pioneer crash test validation (Labbé dissertation, Table 6.3)
            # Total mass: 40 tons, Length: ~23 m, Speed: 56.32 km/h (35 mph)
            # Validated against FRA crash test data [Tyrell et al., 1999]
            default_masses = np.array([4, 10, 4, 4, 4, 10, 4]) * 1000.0  # kg
            default_x = np.array([1.5, 4.5, 8.0, 11.5, 15.0, 18.5, 21.5])  # m
            default_y = np.zeros(7)
            
            if n_masses == 7:
                masses = default_masses
                x_init = default_x
                y_init = default_y
            else:
                M_total = st.number_input("Total Mass (kg)", 100.0, 1e6, 40000.0, 100.0)
                masses = np.ones(n_masses) * M_total / n_masses
                
                L_total = st.number_input("Total Length (m)", 1.0, 200.0, 20.0, 0.1)
                x_init = np.linspace(0.02, 0.02 + L_total, n_masses)
                y_init = np.zeros(n_masses)
        
        else:
            train_config = build_example_train_ui()
            n_masses, masses, x_init, y_init = TrainBuilder.build_train(train_config)
        
        return {
            'n_masses': n_masses,
            'masses': masses,
            'x_init': x_init,
            'y_init': y_init
        }


def build_example_train_ui() -> TrainConfig:
    """Build example train configuration UI."""
    preset = st.selectbox(
        "Train preset",
        ["Generic European", "ICE3-like", "TGV-like", "TRAXX freight"],
    )
    
    presets = {
        "ICE3-like": (7, 70.0, 48.0, 25.0, 25.0),
        "TGV-like": (7, 68.0, 31.0, 22.0, 20.0),
        "TRAXX freight": (6, 84.0, 80.0, 19.0, 15.0),
        "Generic European": (7, 80.0, 50.0, 20.0, 20.0),
    }
    
    n_wag, m_lok, m_wag, L_lok, L_wag = presets[preset]
    
    n_wagons = int(st.number_input("Number of wagons", 0, 20, n_wag, 1))
    mass_lok_t = st.number_input("Locomotive mass (t)", 10.0, 200.0, m_lok, 0.5)
    mass_wagon_t = st.number_input("Wagon mass (t)", 10.0, 200.0, m_wag, 0.5)
    L_lok_val = st.number_input("Locomotive length (m)", 5.0, 40.0, L_lok, 0.1)
    L_wagon_val = st.number_input("Wagon length (m)", 5.0, 40.0, L_wag, 0.1)
    gap = st.number_input("Gap between cars (m)", 0.0, 5.0, 1.0, 0.1)
    
    mass_points_lok = st.radio("Mass points (loco)", [2, 3], index=1, horizontal=True)
    mass_points_wagon = st.radio("Mass points (wagon)", [2, 3], index=0, horizontal=True)
    
    return TrainConfig(
        n_wagons=n_wagons,
        mass_lok_t=mass_lok_t,
        mass_wagon_t=mass_wagon_t,
        L_lok=L_lok_val,
        L_wagon=L_wagon_val,
        mass_points_lok=mass_points_lok,
        mass_points_wagon=mass_points_wagon,
        gap=gap
    )


def build_material_ui(n_masses: int) -> Dict[str, Any]:
    """Build Bouc-Wen material parameters UI."""
    with st.expander("🔧 Bouc-Wen Material", expanded=True):
        fy_MN = st.number_input("Yield Force Fy (MN)", 0.1, 100.0, 15.0, 0.1)
        fy = np.ones(n_masses - 1) * fy_MN * 1e6
        
        uy_mm = st.number_input("Yield Deformation uy (mm)", 1.0, 500.0, 200.0, 1.0)
        uy = np.ones(n_masses - 1) * uy_mm / 1000
        
        st.write(f"Stiffness: {(fy[0]/uy[0])/1e6:.1f} MN/m")
        
        return {
            'fy': fy,
            'uy': uy,
            'bw_a': st.slider("Elastic ratio (a)", 0.0, 1.0, 0.0, 0.05),
            'bw_A': st.number_input("A", 0.1, 10.0, 1.0, 0.1),
            'bw_beta': st.number_input("β", 0.0, 5.0, 0.1, 0.05),
            'bw_gamma': st.number_input("γ", 0.0, 5.0, 0.9, 0.05),
            'bw_n': int(st.number_input("n", 1, 20, 8, 1)),
        }


def build_contact_friction_ui() -> Dict[str, Any]:
    """Build contact and friction parameters UI."""
    params = {}
    
    with st.expander("💥 Contact", expanded=True):
        params['k_wall'] = st.number_input("Wall Stiffness (MN/m)", 
                                          1.0, 100.0, 45.0, 1.0) * 1e6
        params['cr_wall'] = st.slider("Coeff. of Restitution", 0.1, 0.99, 0.8, 0.01)
        params['contact_model'] = st.selectbox(
            "Contact model",
            ["anagnostopoulos", "ye", "hooke", "hertz", "hunt-crossley", 
             "lankarani-nikravesh", "flores", "gonthier", "pant-wijeyewickrema"],
            index=0,
            help="""Contact force models:
            - Anagnostopoulos (1988): Linear viscoelastic (Kelvin-Voigt)
            - Ye et al. (2009): Linear spring with damping
            - Hooke: Linear elastic
            - Hertz: Nonlinear elastic (δ^1.5)
            - Hunt-Crossley: Hertz with damping
            - Lankarani-Nikravesh: Modified Hunt-Crossley
            - Flores: Energy-based dissipation
            - Gonthier: Compliant contact
            - Pant-Wijeyewickrema: Enhanced linear damping
            """
        )
    
    with st.expander("🛞 Friction", expanded=True):
        params['friction_model'] = st.selectbox(
            "Friction model",
            ['lugre', 'dahl', 'coulomb', 'brown-mcphee'],
            index=0
        )
        params['mu_s'] = st.slider("μs (static)", 0.0, 1.0, 0.4, 0.01)
        params['mu_k'] = st.slider("μk (kinetic)", 0.0, 1.0, 0.3, 0.01)
        params['sigma_0'] = st.number_input("σ₀", 1e3, 1e7, 1e5, format="%.0e")
        params['sigma_1'] = st.number_input("σ₁", 1.0, 1e5, 316.0, 1.0)
        params['sigma_2'] = st.number_input("σ₂ (viscous)", 0.0, 2.0, 0.4, 0.1)
    
    return params


def execute_simulation(params: Dict[str, Any]):
    """Execute simulation and display results."""
    sim_params = SimulationParams(**params)
    
    with st.spinner("Running HHT-α simulation..."):
        try:
            simulator = ImpactSimulator(sim_params)
            df = simulator.run()
            st.success("✅ Complete!")
        except Exception as e:
            st.error(f"Simulation error: {e}")
            raise
    
    # Display metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Force", f"{df['Impact_Force_MN'].max():.2f} MN")
    c2.metric("Max Penetration", f"{df['Penetration_mm'].max():.2f} mm")
    c3.metric("Max Acceleration", f"{df['Acceleration_g'].max():.1f} g")
    
    # Create and display plots
    fig = create_results_plots(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.subheader("📥 Export")
    e1, e2, e3 = st.columns(3)
    
    e1.download_button(
        "📄 CSV",
        df.to_csv(index=False).encode(),
        "results.csv",
        "text/csv",
        use_container_width=True
    )
    
    e2.download_button(
        "📝 TXT",
        df.to_string(index=False).encode(),
        "results.txt",
        "text/plain",
        use_container_width=True
    )
    
    e3.download_button(
        "📊 XLSX",
        to_excel(df),
        "results.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    
    # Citation information
    st.markdown("---")
    st.markdown("""
    ### 📚 Citation
    
    If you use this simulator in your research, please cite the original research report:
    
    **Plain Text:**
    ```
    Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024).
    Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr.
    Berichte des Deutschen Zentrums für Schienenverkehrsforschung, Bericht 53.
    Deutsches Zentrum für Schienenverkehrsforschung beim Eisenbahn-Bundesamt.
    https://doi.org/10.48755/dzsf.240006.01
    ```
    
    **BibTeX:**
    ```bibtex
    @techreport{Stempniewski2024Anpralllasten,
      author       = {Stempniewski, Lothar and 
                      Labbé, Sebastián and 
                      Siegel, Steffen and 
                      Bosch, Robin},
      title        = {Überprüfung und Anpassung der Anpralllasten 
                      aus dem Eisenbahnverkehr},
      institution  = {Deutsches Zentrum für Schienenverkehrsforschung 
                      beim Eisenbahn-Bundesamt},
      year         = {2024},
      type         = {Bericht},
      number       = {53},
      address      = {Dresden, Germany},
      note         = {Projektnummer 2018-08-U-1217, 
                      Commissioned by Eisenbahn-Bundesamt},
      doi          = {10.48755/dzsf.240006.01},
      issn         = {2629-7973},
      url          = {https://www.dzsf.bund.de/SharedDocs/Downloads/DZSF/Veroeffentlichungen/Forschungsberichte/2024/ForBe_53_2024_Anpralllasten.pdf}
    }
    ```
    
    **APA 7th Edition:**
    ```
    Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024). 
    Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr 
    (DZSF Bericht No. 53). Deutsches Zentrum für Schienenverkehrsforschung 
    beim Eisenbahn-Bundesamt. https://doi.org/10.48755/dzsf.240006.01
    ```
    
    ---
    
    **License:** This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
    """)


if __name__ == "__main__":
    main()
