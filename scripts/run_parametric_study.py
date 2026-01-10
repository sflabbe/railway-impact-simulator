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

from railway_simulator.core.engine import run_simulation
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


def save_results(
    df: pd.DataFrame,
    case: SimulationCase,
    output_dir: Path,
    metrics: Dict[str, Any],
) -> Path:
    """Save simulation results to CSV and metrics to JSON."""
    case_dir = output_dir / case.section / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

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


def run_case(case: SimulationCase, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    """Run a single simulation case."""
    logger.info(f"Running case: {case.section}/{case.name}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would run with overrides: {case.overrides}")
        return None

    # Load and modify config
    params = load_config(case.base_config)
    params.update(case.overrides)

    # Run simulation
    start_time = time.time()
    try:
        df = run_simulation(params)
        elapsed = time.time() - start_time

        # Compute metrics
        metrics = compute_metrics(df, params)
        metrics["wall_time_s"] = elapsed
        metrics["case_name"] = case.name
        metrics["section"] = case.section
        metrics.update(case.meta)

        # Check plausibility
        passed, warnings = check_plausibility(metrics)
        metrics["plausibility_passed"] = passed
        metrics["plausibility_warnings"] = warnings

        if warnings:
            for w in warnings:
                logger.warning(f"  {w}")

        # Save results
        save_results(df, case, RESULTS_DIR, metrics)

        logger.info(
            f"  Completed in {elapsed:.1f}s | "
            f"Peak force: {metrics.get('peak_force_MN', 0):.2f} MN | "
            f"DAF: {metrics.get('DAF', 0):.2f}"
        )

        return metrics

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return {"error": str(e), "case_name": case.name, "section": case.section}


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
) -> pd.DataFrame:
    """Run parametric study for specified sections."""

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

        # Print statistics
        logger.info("\n" + "=" * 60)
        logger.info("STUDY COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total runs: {len(results)}")
        logger.info(f"Successful: {sum(1 for r in results if 'error' not in r)}")
        logger.info(f"Failed: {sum(1 for r in results if 'error' in r)}")

        if "plausibility_passed" in summary_df.columns:
            passed = summary_df["plausibility_passed"].sum()
            logger.info(f"Plausibility passed: {passed}/{len(results)}")

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

    run_study(sections=sections, dry_run=args.dry_run, resume_from=resume)


if __name__ == "__main__":
    main()
