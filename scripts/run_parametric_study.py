#!/usr/bin/env python3
"""
Batch Runner for Dissertation Parametric Studies (Chapter 8)

This script runs all 101 parametric simulations required for Chapter 8:
- 8.1 Speed & Restitution Coefficient (32 runs)
- 8.2 Number of Cars (2 runs)
- 8.3 Friction (32 runs)
- 8.4 Structure Stiffness (24 runs)
- 8.5 Train Materials (2 runs)
- 8.6 Contact Models (9 runs)

Usage:
    python scripts/run_parametric_study.py --all
    python scripts/run_parametric_study.py --section 8.1
    python scripts/run_parametric_study.py --section 8.6 --dry-run

Author: Railway Impact Simulator
Date: 2026-01-10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from railway_simulator.core.engine import run_simulation, NonConvergenceError
from railway_simulator.core.contact import ContactModels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

BASE_CONFIG_DIR = Path(__file__).parent.parent / "configs"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "dissertation_ch8"

# Base configurations
ICE1_FULL_CONFIG = BASE_CONFIG_DIR / "ice1_full_dissertation.yml"
ICE1_SINGLE_WAGON_CONFIG = BASE_CONFIG_DIR / "ice1_single_wagon_dissertation.yml"
ICE1_STEEL_CONFIG = BASE_CONFIG_DIR / "ice1_full_steel_dissertation.yml"

# Speed conversions
KMH_TO_MS = 1.0 / 3.6

# Physical plausibility thresholds (Herr Facke method)
PLAUSIBILITY_CHECKS = {
    "daf_max": 3.0,                    # Dynamic amplification factor
    "energy_balance_error_max": 0.01,  # 1% of initial energy
    "acceleration_max_g": 50.0,        # Peak acceleration (excl. initial contact)
    "contact_duration_range_s": (0.5, 2.0),  # Expected range for ICE 1 @ 80 km/h
    "force_plateau_tolerance": 0.3,    # 30% tolerance on fy
}

# Retry policy for entire cases (dt reduction at case level)
MAX_CASE_RETRIES = 2
CASE_DT_REDUCTION_FACTOR = 0.5
CASE_DT_MIN = 1e-7


@dataclass
class SimulationCase:
    """Definition of a single simulation case."""
    name: str
    section: str
    base_config: Path
    overrides: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_case_dir(
    output_dir: Path,
    case: SimulationCase,
    attempt_dir_name: Optional[str] = None,
) -> Path:
    """Resolve the case directory, optionally nested for a given attempt."""
    case_dir = output_dir / case.section / case.name
    if attempt_dir_name:
        case_dir = case_dir / attempt_dir_name
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def save_results(
    df: pd.DataFrame,
    case: SimulationCase,
    output_dir: Path,
    metrics: Dict[str, Any],
    attempt_dir_name: Optional[str] = None,
) -> Path:
    """Save simulation results to CSV and metrics to JSON."""
    case_dir = get_case_dir(output_dir, case, attempt_dir_name)

    # Save time history
    csv_path = case_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    # Save metrics
    metrics_path = case_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    return csv_path


def compute_metrics(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute post-processing metrics from simulation results."""
    metrics = {}

    # Peak force
    force_col = "Impact_Force_MN"
    if force_col in df.columns:
        metrics["peak_force_MN"] = float(df[force_col].max())
        metrics["time_of_peak_s"] = float(df.loc[df[force_col].idxmax(), "Time_s"])

    # Peak penetration
    if "Penetration_mm" in df.columns:
        metrics["peak_penetration_mm"] = float(df["Penetration_mm"].max())

    # Peak acceleration
    if "Acceleration_g" in df.columns:
        metrics["peak_acceleration_g"] = float(df["Acceleration_g"].abs().max())

    # Contact duration (time where force > 0.1 MN)
    if force_col in df.columns:
        in_contact = df[force_col] > 0.1
        if in_contact.any():
            first_contact = df.loc[in_contact, "Time_s"].min()
            last_contact = df.loc[in_contact, "Time_s"].max()
            metrics["contact_duration_s"] = float(last_contact - first_contact)
            metrics["t_contact_start"] = float(first_contact)
            metrics["t_contact_end"] = float(last_contact)
            if "Velocity_m_s" in df.columns:
                metrics["v_front_at_contact_start"] = float(df.loc[in_contact, "Velocity_m_s"].iloc[0])
        else:
            metrics["contact_duration_s"] = 0.0
            metrics["t_contact_start"] = None
            metrics["t_contact_end"] = None
            metrics["v_front_at_contact_start"] = None

    # Front position metrics
    if "Position_x_m" in df.columns:
        x_front_min = float(df["Position_x_m"].min())
        metrics["x_front_min"] = x_front_min
        impact_occurred = x_front_min <= 0.0
        metrics["impact_occurred"] = bool(impact_occurred)
        if not impact_occurred:
            metrics["no_impact_reason"] = "x_front_min>0"

    # Energy metrics
    if "E_num_ratio" in df.columns:
        metrics["energy_balance_error"] = float(df["E_num_ratio"].abs().max())

    # Initial kinetic energy
    if "E_kin_J" in df.columns:
        metrics["E_kin_initial_J"] = float(df["E_kin_J"].iloc[0])

    # Dynamic amplification factor (peak force / yield force)
    if force_col in df.columns and "fy" in params:
        fy_arr = np.array(params["fy"])
        fy_MN = fy_arr[0] / 1e6 if len(fy_arr) > 0 else 15.0
        metrics["DAF"] = metrics.get("peak_force_MN", 0) / fy_MN

    for key in ("fallback_used", "max_residual_seen", "max_iters_step", "converged_all_steps"):
        if key in df.attrs:
            metrics[key] = df.attrs[key]

    return metrics


def check_plausibility(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check physical plausibility of results."""
    warnings = []
    passed = True

    # DAF check
    if metrics.get("DAF", 0) > PLAUSIBILITY_CHECKS["daf_max"]:
        warnings.append(
            f"DAF = {metrics['DAF']:.2f} exceeds max {PLAUSIBILITY_CHECKS['daf_max']}"
        )
        passed = False

    # Energy balance
    if metrics.get("energy_balance_error", 0) > PLAUSIBILITY_CHECKS["energy_balance_error_max"]:
        warnings.append(
            f"Energy balance error = {metrics['energy_balance_error']:.4f} "
            f"exceeds {PLAUSIBILITY_CHECKS['energy_balance_error_max']}"
        )
        passed = False

    # Acceleration
    if metrics.get("peak_acceleration_g", 0) > PLAUSIBILITY_CHECKS["acceleration_max_g"]:
        warnings.append(
            f"Peak acceleration = {metrics['peak_acceleration_g']:.1f}g "
            f"exceeds {PLAUSIBILITY_CHECKS['acceleration_max_g']}g"
        )

    # Contact duration (only warn, don't fail)
    dur = metrics.get("contact_duration_s", 0)
    dur_min, dur_max = PLAUSIBILITY_CHECKS["contact_duration_range_s"]
    if dur < dur_min or dur > dur_max:
        warnings.append(
            f"Contact duration = {dur:.3f}s outside expected range [{dur_min}, {dur_max}]s"
        )

    return passed, warnings


def save_diagnostics(case: SimulationCase, output_dir: Path, diagnostics: Dict[str, Any]) -> Path:
    """Save diagnostics for a failed case."""
    case_dir = get_case_dir(output_dir, case)
    diag_path = case_dir / "diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    return diag_path


def save_failed_metrics(
    case: SimulationCase,
    output_dir: Path,
    metrics: Dict[str, Any],
    attempt_dir_name: Optional[str] = None,
) -> Path:
    """Save metrics.json for a failed case."""
    case_dir = get_case_dir(output_dir, case, attempt_dir_name)
    metrics_path = case_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    return metrics_path


def save_diagnostics_attempt(
    case: SimulationCase,
    output_dir: Path,
    diagnostics: Dict[str, Any],
    attempt_dir_name: Optional[str] = None,
) -> Path:
    """Save diagnostics for a failed case attempt."""
    case_dir = get_case_dir(output_dir, case, attempt_dir_name)
    diag_path = case_dir / "diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    return diag_path


def save_initial_state_snapshot(
    case: SimulationCase,
    output_dir: Path,
    snapshot: Dict[str, Any],
    attempt_dir_name: Optional[str] = None,
) -> Path:
    """Save initial-state snapshot for a case attempt."""
    case_dir = get_case_dir(output_dir, case, attempt_dir_name)
    snapshot_path = case_dir / "initial_state.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    return snapshot_path


def recommended_action_from_error(error: NonConvergenceError, params: Dict[str, Any]) -> str:
    """Recommend next action based on error snapshot and solver params."""
    snapshot = error.state_snapshot
    if snapshot.get("in_contact", False):
        if str(params.get("newton_jacobian_mode", "per_step")).lower() != "each_iter":
            return "enable newton_jacobian_mode each_iter"
        max_iters = params.get("newton_max_iters")
        if max_iters is not None and error.iter_count >= int(max_iters):
            return "increase max_iter"
        return "reduce h_init"
    if snapshot.get("x_front_last", 0) > 0:
        v_front = snapshot.get("v_front_last", 0.0)
        if abs(v_front) < 1e-3:
            return "damping too high"
        return "check initial conditions"
    return "check initial conditions"


def run_case(case: SimulationCase, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    """Run a single simulation case.

    Returns metrics dict on success, or a dict with status='failed' on failure.
    Never raises exceptions - all errors are captured and returned as failed cases.
    """
    logger.info(f"Running case: {case.section}/{case.name}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would run with overrides: {case.overrides}")
        return None

    # Load and modify config
    base_params = load_config(case.base_config)
    base_params.update(case.overrides)

    t_max = float(base_params.get("T_max", 0.0))
    t_int = base_params.get("T_int")
    if t_max <= 0.0 and isinstance(t_int, (list, tuple)) and len(t_int) == 2:
        t_max = float(t_int[1]) - float(t_int[0])

    max_case_retries = MAX_CASE_RETRIES
    dt_factor = CASE_DT_REDUCTION_FACTOR
    dt_min = CASE_DT_MIN

    attempt_records: List[Dict[str, Any]] = []
    h_init_used = float(base_params.get("h_init", 1e-4))
    success_attempt_index: Optional[int] = None
    success_metrics: Optional[Dict[str, Any]] = None
    success_attempt_dir: Optional[Path] = None

    for attempt_index in range(max_case_retries + 1):
        attempt_dir_name = f"attempt_{attempt_index}"
        attempt_params = deepcopy(base_params)
        attempt_initial_snapshot: Dict[str, Any] = {}

        def capture_initial_state(snapshot: Dict[str, Any]) -> None:
            attempt_initial_snapshot.update(snapshot)

        if attempt_index > 0:
            attempt_params["h_init"] = h_init_used
            step_used = int(np.ceil(t_max / h_init_used)) if t_max > 0.0 else int(attempt_params.get("step", 0))
            attempt_params["step"] = step_used
            attempt_params["T_int"] = (0.0, float(t_max))
        else:
            step_used = int(attempt_params.get("step", np.ceil(t_max / h_init_used))) if t_max > 0.0 else int(
                attempt_params.get("step", 0)
            )
            attempt_params["step"] = step_used
            attempt_params["h_init"] = h_init_used

        rerun_with_reduced_dt = attempt_index > 0
        logger.info(
            "  Attempt %d/%d | h_init=%.3e | step=%d | rerun_with_reduced_dt=%s",
            attempt_index + 1,
            max_case_retries + 1,
            h_init_used,
            step_used,
            rerun_with_reduced_dt,
        )

        start_time = time.time()
        try:
            df = run_simulation(attempt_params, initial_state_callback=capture_initial_state)
            elapsed = time.time() - start_time
            if attempt_initial_snapshot:
                save_initial_state_snapshot(case, RESULTS_DIR, attempt_initial_snapshot, attempt_dir_name)
                logger.info(
                    "  Initial state snapshot saved (engine=%s integrator=%s q=%s qp=%s)",
                    attempt_initial_snapshot.get("engine_id"),
                    attempt_initial_snapshot.get("integrator_id"),
                    attempt_initial_snapshot.get("q_id"),
                    attempt_initial_snapshot.get("qp_id"),
                )

            metrics = compute_metrics(df, attempt_params)
            metrics["status"] = "success"
            metrics["wall_time_s"] = elapsed
            metrics["case_name"] = case.name
            metrics["section"] = case.section
            metrics["attempt_index"] = attempt_index
            metrics["h_init_used"] = h_init_used
            metrics["step_used"] = step_used
            metrics["rerun_with_reduced_dt"] = rerun_with_reduced_dt
            metrics.update(case.meta)

            passed, warnings = check_plausibility(metrics)
            metrics["plausibility_passed"] = passed
            metrics["plausibility_warnings"] = warnings

            if warnings:
                for w in warnings:
                    logger.warning(f"  {w}")

            results_path = save_results(df, case, RESULTS_DIR, metrics, attempt_dir_name)
            metrics_path = results_path.with_name("metrics.json")
            attempt_records.append({"metrics": metrics, "metrics_path": metrics_path})

            logger.info(
                f"  Completed in {elapsed:.1f}s | "
                f"Peak force: {metrics.get('peak_force_MN', 0):.2f} MN | "
                f"DAF: {metrics.get('DAF', 0):.2f}"
            )

            success_attempt_index = attempt_index
            success_metrics = metrics
            success_attempt_dir = results_path.parent
            break

        except NonConvergenceError as e:
            elapsed = time.time() - start_time
            logger.error(f"  NON-CONVERGENCE at step {e.step_idx}, t={e.t:.4f}s: {e}")

            metrics = {
                "status": "failed",
                "case_name": case.name,
                "section": case.section,
                "wall_time_s": elapsed,
                "error_type": "NonConvergenceError",
                "error_message": str(e)[:500],
                "failure_stage": e.failure_stage,
                "last_step_index": e.step_idx,
                "t_last": e.t,
                "residual_norm": e.residual_norm,
                "iter_count": e.iter_count,
                "dt_effective": e.dt_effective,
                "dt_reductions_used": e.dt_reductions_used,
                "fallback_attempted": e.fallback_attempted,
                "attempt_index": attempt_index,
                "h_init_used": h_init_used,
                "step_used": step_used,
                "rerun_with_reduced_dt": rerun_with_reduced_dt,
                "peak_force_MN": float('nan'),
                "peak_penetration_mm": float('nan'),
                "peak_acceleration_g": float('nan'),
                "contact_duration_s": float('nan'),
                "DAF": float('nan'),
                "plausibility_passed": False,
                "plausibility_warnings": ["Case failed to converge"],
            }
            metrics.update(case.meta)

            for key, val in e.state_snapshot.items():
                metrics[f"snapshot_{key}"] = val

            diagnostics = e.to_diagnostics_dict()
            diagnostics["case_name"] = case.name
            diagnostics["section"] = case.section
            diagnostics["attempt_index"] = attempt_index
            diagnostics["params_used"] = {
                "solver": attempt_params.get("solver"),
                "h_init": attempt_params.get("h_init"),
                "step": attempt_params.get("step"),
                "newton_max_iters": attempt_params.get("newton_max_iters"),
                "newton_jacobian_mode": attempt_params.get("newton_jacobian_mode"),
                "newton_tol": attempt_params.get("newton_tol"),
            }
            if attempt_initial_snapshot:
                diagnostics["initial_state_snapshot"] = attempt_initial_snapshot
                save_initial_state_snapshot(case, RESULTS_DIR, attempt_initial_snapshot, attempt_dir_name)

            if e.state_snapshot.get("in_contact", False):
                diagnostics["diagnosis"] = "Failed during contact phase - may need smaller dt or higher max_iter"
            elif e.state_snapshot.get("x_front_last", 0) > 0:
                diagnostics["diagnosis"] = "Failed before impact - check initial conditions or damping"
            else:
                diagnostics["diagnosis"] = "Unknown - check residual norm and iter count"

            diagnostics["recommended_action"] = recommended_action_from_error(e, attempt_params)

            save_diagnostics_attempt(case, RESULTS_DIR, diagnostics, attempt_dir_name)
            metrics_path = save_failed_metrics(case, RESULTS_DIR, metrics, attempt_dir_name)
            attempt_records.append({"metrics": metrics, "metrics_path": metrics_path})

            next_h_init = h_init_used * dt_factor
            if attempt_index >= max_case_retries or next_h_init < dt_min:
                break
            h_init_used = next_h_init

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  FAILED (unexpected): {type(e).__name__}: {e}")

            metrics = {
                "status": "failed",
                "case_name": case.name,
                "section": case.section,
                "wall_time_s": elapsed,
                "error_type": type(e).__name__,
                "error_message": str(e)[:500],
                "failure_stage": "unknown",
                "last_step_index": -1,
                "t_last": 0.0,
                "fallback_attempted": False,
                "attempt_index": attempt_index,
                "h_init_used": h_init_used,
                "step_used": step_used,
                "rerun_with_reduced_dt": rerun_with_reduced_dt,
                "peak_force_MN": float('nan'),
                "peak_penetration_mm": float('nan'),
                "peak_acceleration_g": float('nan'),
                "contact_duration_s": float('nan'),
                "DAF": float('nan'),
                "plausibility_passed": False,
                "plausibility_warnings": [f"Case failed: {type(e).__name__}"],
            }
            metrics.update(case.meta)

            diagnostics = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "case_name": case.name,
                "section": case.section,
                "attempt_index": attempt_index,
                "params_used": {
                    "solver": attempt_params.get("solver"),
                    "h_init": attempt_params.get("h_init"),
                    "step": attempt_params.get("step"),
                },
                "diagnosis": "Unexpected error - check logs for stack trace",
                "recommended_action": "check logs",
            }
            if attempt_initial_snapshot:
                diagnostics["initial_state_snapshot"] = attempt_initial_snapshot
                save_initial_state_snapshot(case, RESULTS_DIR, attempt_initial_snapshot, attempt_dir_name)
            save_diagnostics_attempt(case, RESULTS_DIR, diagnostics, attempt_dir_name)
            metrics_path = save_failed_metrics(case, RESULTS_DIR, metrics, attempt_dir_name)
            attempt_records.append({"metrics": metrics, "metrics_path": metrics_path})
            break

    n_attempts_total = len(attempt_records)
    for record in attempt_records:
        record["metrics"]["n_attempts_total"] = n_attempts_total
        record["metrics"]["converged_after_attempt"] = success_attempt_index
        with open(record["metrics_path"], "w", encoding="utf-8") as f:
            json.dump(record["metrics"], f, indent=2, default=str)

    case_dir = get_case_dir(RESULTS_DIR, case)
    if success_attempt_index is not None and success_attempt_dir:
        shutil.copy2(success_attempt_dir / "results.csv", case_dir / "results.csv")
        shutil.copy2(success_attempt_dir / "metrics.json", case_dir / "metrics.json")
        return success_metrics

    if attempt_records:
        last_attempt_dir = Path(attempt_records[-1]["metrics_path"]).parent
        if (last_attempt_dir / "metrics.json").exists():
            shutil.copy2(last_attempt_dir / "metrics.json", case_dir / "metrics.json")
        if (last_attempt_dir / "diagnostics.json").exists():
            shutil.copy2(last_attempt_dir / "diagnostics.json", case_dir / "diagnostics.json")
        return attempt_records[-1]["metrics"]

    return None


# =============================================================================
# Section 8.1: Speed & Restitution Coefficient
# =============================================================================

def generate_section_8_1_cases() -> List[SimulationCase]:
    """
    Section 8.1: Influence of Speed and Restitution Coefficient

    Figures: fig38-anagnostopoulos-flores50-80.pdf, fig38anagnostopoulos-flores120-200.pdf

    32 runs: 4 speeds x 4 cr values x 2 models
    """
    cases = []

    speeds_kmh = [50, 80, 120, 200]
    cr_values = [0.6, 0.7, 0.8, 0.9]
    models = ["anagnostopoulos", "flores"]

    for speed in speeds_kmh:
        for cr in cr_values:
            for model in models:
                name = f"v{speed}_cr{int(cr*10)}_{model}"
                cases.append(SimulationCase(
                    name=name,
                    section="8.1_speed_restitution",
                    base_config=ICE1_FULL_CONFIG,
                    overrides={
                        "v0_init": -speed * KMH_TO_MS,
                        "cr_wall": cr,
                        "contact_model": model,
                    },
                    meta={
                        "speed_kmh": speed,
                        "cr_wall": cr,
                        "contact_model": model,
                    },
                ))

    return cases


# =============================================================================
# Section 8.2: Number of Cars
# =============================================================================

def generate_section_8_2_cases() -> List[SimulationCase]:
    """
    Section 8.2: Influence of Number of Cars

    Figure: fig39numbercars.pdf

    2 runs: Single wagon (40t) vs Full ICE 1 (14 units)
    """
    cases = []

    # Single wagon
    cases.append(SimulationCase(
        name="single_wagon_40t",
        section="8.2_number_cars",
        base_config=ICE1_SINGLE_WAGON_CONFIG,
        overrides={
            "v0_init": -80 * KMH_TO_MS,
            "cr_wall": 0.8,
            "contact_model": "anagnostopoulos",
        },
        meta={
            "config": "single_wagon",
            "n_cars": 1,
            "total_mass_t": 40,
        },
    ))

    # Full train
    cases.append(SimulationCase(
        name="full_ice1_14cars",
        section="8.2_number_cars",
        base_config=ICE1_FULL_CONFIG,
        overrides={
            "v0_init": -80 * KMH_TO_MS,
            "cr_wall": 0.8,
            "contact_model": "anagnostopoulos",
        },
        meta={
            "config": "full_ice1",
            "n_cars": 14,
            "total_mass_t": 772.8,
        },
    ))

    return cases


# =============================================================================
# Section 8.3: Friction
# =============================================================================

def generate_section_8_3_cases() -> List[SimulationCase]:
    """
    Section 8.3: Influence of Friction

    Figures: fig36.pdf (Flores), fig37.pdf (Anagnostopoulos)

    32 runs: 4 mu combinations x 4 distances x 2 models
    """
    cases = []

    mu_combinations = [
        (0.3, 0.2),
        (0.5, 0.4),
        (0.7, 0.6),
        (1.0, 0.9),
    ]
    distances_m = [3, 6, 9, 12]
    models = ["anagnostopoulos", "flores"]

    for mu_s, mu_k in mu_combinations:
        for dist in distances_m:
            for model in models:
                name = f"mu{int(mu_s*10)}{int(mu_k*10)}_d{dist}m_{model}"
                cases.append(SimulationCase(
                    name=name,
                    section="8.3_friction",
                    base_config=ICE1_FULL_CONFIG,
                    overrides={
                        "v0_init": -80 * KMH_TO_MS,
                        "cr_wall": 0.8,
                        "contact_model": model,
                        "mu_s": mu_s,
                        "mu_k": mu_k,
                        "d0": dist,  # Initial distance to wall
                    },
                    meta={
                        "mu_s": mu_s,
                        "mu_k": mu_k,
                        "distance_m": dist,
                        "contact_model": model,
                    },
                ))

    return cases


# =============================================================================
# Section 8.4: Structure Stiffness
# =============================================================================

def generate_section_8_4_cases() -> List[SimulationCase]:
    """
    Section 8.4: Influence of Structure Stiffness

    Figure: fig40stiffness.pdf

    24 runs: 3 K factors x 4 cr values x 2 models
    """
    cases = []

    # Base k_wall = 60 MN/m
    k_base = 60e6
    k_factors = [0.8, 1.0, 2.2]
    cr_values = [0.6, 0.7, 0.8, 0.9]
    models = ["anagnostopoulos", "flores"]

    for k_factor in k_factors:
        for cr in cr_values:
            for model in models:
                k_wall = k_base * k_factor
                name = f"k{int(k_factor*10)}_cr{int(cr*10)}_{model}"
                cases.append(SimulationCase(
                    name=name,
                    section="8.4_stiffness",
                    base_config=ICE1_FULL_CONFIG,
                    overrides={
                        "v0_init": -80 * KMH_TO_MS,
                        "k_wall": k_wall,
                        "cr_wall": cr,
                        "contact_model": model,
                    },
                    meta={
                        "k_factor": k_factor,
                        "k_wall_MNm": k_wall / 1e6,
                        "cr_wall": cr,
                        "contact_model": model,
                    },
                ))

    return cases


# =============================================================================
# Section 8.5: Train Materials
# =============================================================================

def generate_section_8_5_cases() -> List[SimulationCase]:
    """
    Section 8.5: Influence of Train Materials

    Figures: fig33.pdf (quasi-static), fig34.pdf (comparison)

    2 runs: Aluminium vs Steel S355
    """
    cases = []

    # Aluminium (ICE 1 original)
    cases.append(SimulationCase(
        name="aluminium_fy15_uy200",
        section="8.5_materials",
        base_config=ICE1_FULL_CONFIG,
        overrides={
            "v0_init": -80 * KMH_TO_MS,
            "cr_wall": 0.8,
            "contact_model": "anagnostopoulos",
        },
        meta={
            "material": "aluminium",
            "fy_MN": 15,
            "uy_mm": 200,
        },
    ))

    # Steel S355
    cases.append(SimulationCase(
        name="steel_s355_fy18_uy40",
        section="8.5_materials",
        base_config=ICE1_STEEL_CONFIG,
        overrides={
            "v0_init": -80 * KMH_TO_MS,
            "cr_wall": 0.8,
            "contact_model": "anagnostopoulos",
        },
        meta={
            "material": "steel_s355",
            "fy_MN": 18,
            "uy_mm": 40,
        },
    ))

    return cases


# =============================================================================
# Section 8.6: Contact Models Comparison
# =============================================================================

def generate_section_8_6_cases() -> List[SimulationCase]:
    """
    Section 8.6: Comparison of Contact Models

    Figure: ComparisonContactModels2.pdf

    9 runs: All contact models
    """
    cases = []

    contact_models = [
        "hooke",
        "hertz",
        "hunt-crossley",
        "lankarani-nikravesh",
        "flores",
        "gonthier",
        "ye",
        "pant-wijeyewickrema",
        "anagnostopoulos",
    ]

    for model in contact_models:
        name = f"model_{model.replace('-', '_')}"
        cases.append(SimulationCase(
            name=name,
            section="8.6_contact_models",
            base_config=ICE1_FULL_CONFIG,
            overrides={
                "v0_init": -80 * KMH_TO_MS,
                "cr_wall": 0.8,
                "contact_model": model,
                "mu_s": 0.4,
                "mu_k": 0.3,
            },
            meta={
                "contact_model": model,
            },
        ))

    return cases


# =============================================================================
# Main Runner
# =============================================================================

SECTION_GENERATORS = {
    "8.1": generate_section_8_1_cases,
    "8.2": generate_section_8_2_cases,
    "8.3": generate_section_8_3_cases,
    "8.4": generate_section_8_4_cases,
    "8.5": generate_section_8_5_cases,
    "8.6": generate_section_8_6_cases,
}


def generate_all_cases() -> List[SimulationCase]:
    """Generate all simulation cases."""
    cases = []
    for gen in SECTION_GENERATORS.values():
        cases.extend(gen())
    return cases


def run_study(
    sections: Optional[List[str]] = None,
    dry_run: bool = False,
    resume_from: Optional[str] = None,
    max_cases: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Run parametric study for specified sections.

    Args:
        sections: List of section IDs to run (e.g., ["8.1", "8.3"]).
        dry_run: If True, only print what would be run.
        resume_from: If set, resume from previous partial run.
        max_cases: If set, limit to this many cases (for quick testing).
        seed: Random seed for shuffling cases when using max_cases.

    Returns:
        Summary DataFrame with all results (including failed cases).
    """

    # Generate cases
    if sections is None or "all" in sections:
        cases = generate_all_cases()
    else:
        cases = []
        for sec in sections:
            if sec in SECTION_GENERATORS:
                cases.extend(SECTION_GENERATORS[sec]())
            else:
                logger.warning(f"Unknown section: {sec}")

    # Apply max_cases limit with optional shuffle
    if max_cases is not None and max_cases > 0 and max_cases < len(cases):
        if seed is not None:
            import random
            random.seed(seed)
            random.shuffle(cases)
        cases = cases[:max_cases]
        logger.info(f"Limited to {max_cases} cases (seed={seed})")

    logger.info(f"Total cases to run: {len(cases)}")

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resume logic
    completed_cases = set()
    if resume_from:
        summary_path = RESULTS_DIR / "summary.csv"
        if summary_path.exists():
            prev_summary = pd.read_csv(summary_path)
            completed_cases = set(prev_summary["case_name"].tolist())
            logger.info(f"Resuming: {len(completed_cases)} cases already completed")

    # Run simulations
    results = []
    total = len(cases)

    for i, case in enumerate(cases, 1):
        if case.name in completed_cases:
            logger.info(f"[{i}/{total}] Skipping (already done): {case.name}")
            continue

        logger.info(f"[{i}/{total}] {case.section}/{case.name}")

        metrics = run_case(case, dry_run=dry_run)
        if metrics:
            results.append(metrics)

        # Save intermediate summary
        if results and not dry_run and i % 10 == 0:
            pd.DataFrame(results).to_csv(
                RESULTS_DIR / "summary_partial.csv", index=False
            )

    # Final summary
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = RESULTS_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to: {summary_path}")

        # Compute statistics
        n_total = len(results)
        n_converged = sum(1 for r in results if r.get("status") == "success")
        n_failed = sum(1 for r in results if r.get("status") == "failed")
        failed_cases = [r.get("case_name") for r in results if r.get("status") == "failed"]

        # Print statistics
        logger.info("\n" + "=" * 60)
        logger.info("STUDY COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total runs: {n_total}")
        logger.info(f"Converged: {n_converged}")
        logger.info(f"Failed: {n_failed}")

        if failed_cases:
            logger.info(f"Failed cases: {failed_cases}")

        if "plausibility_passed" in summary_df.columns:
            passed = summary_df["plausibility_passed"].sum()
            logger.info(f"Plausibility passed: {passed}/{n_total}")

        # Save extended summary with convergence info
        summary_meta = {
            "n_total": n_total,
            "n_converged": n_converged,
            "n_failed": n_failed,
            "failed_cases": failed_cases,
            "sections_run": sections or ["all"],
        }
        summary_meta_path = RESULTS_DIR / "summary_meta.json"
        with open(summary_meta_path, "w") as f:
            json.dump(summary_meta, f, indent=2)
        logger.info(f"Summary metadata saved to: {summary_meta_path}")

        return summary_df

    return pd.DataFrame()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run dissertation parametric studies (Chapter 8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sections:
  8.1  Speed & Restitution Coefficient (32 runs)
  8.2  Number of Cars (2 runs)
  8.3  Friction (32 runs)
  8.4  Structure Stiffness (24 runs)
  8.5  Train Materials (2 runs)
  8.6  Contact Models (9 runs)

Examples:
  python run_parametric_study.py --all
  python run_parametric_study.py --section 8.1 8.6
  python run_parametric_study.py --section 8.6 --dry-run
  python run_parametric_study.py --all --resume
  python run_parametric_study.py --section 8.3 --max-cases 8 --seed 42
        """,
    )

    parser.add_argument(
        "--all", action="store_true", help="Run all sections"
    )
    parser.add_argument(
        "--section", "-s", nargs="+",
        choices=list(SECTION_GENERATORS.keys()),
        help="Sections to run (e.g., 8.1 8.2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without executing"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous partial run"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all cases without running"
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Limit to N cases (for quick testing)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for shuffling when using --max-cases"
    )

    args = parser.parse_args()

    if args.list:
        cases = generate_all_cases()
        print(f"\nTotal cases: {len(cases)}\n")
        for sec in SECTION_GENERATORS:
            sec_cases = SECTION_GENERATORS[sec]()
            print(f"Section {sec}: {len(sec_cases)} cases")
            for c in sec_cases[:3]:
                print(f"  - {c.name}")
            if len(sec_cases) > 3:
                print(f"  ... and {len(sec_cases) - 3} more")
        return

    if not args.all and not args.section:
        parser.print_help()
        return

    sections = None if args.all else args.section
    resume = "partial" if args.resume else None

    run_study(
        sections=sections,
        dry_run=args.dry_run,
        resume_from=resume,
        max_cases=args.max_cases,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
