"""
Command line interface for the Railway Impact Simulator.

Usage (after `pip install -e .` from the repo root):

    railway-sim --help
    railway-sim                          # run built-in default scenario
    railway-sim --config config.yml      # run with external config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore[import]
except ImportError:
    yaml = None

from railway_simulator.core.engine import SimulationParams, run_simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config_dict() -> Dict[str, Any]:
    """
    Built-in demo configuration (research loco, 7 masses, 56 km/h).

    This is intentionally simple but physically reasonable and matches
    the defaults of the Streamlit UI as closely as possible.
    """
    # --- geometry: 7-mass research locomotive model ---
    masses = [4, 10, 4, 4, 4, 10, 4]           # [t]
    masses = [m * 1000.0 for m in masses]      # -> kg

    x_init = [0.02, 3.02, 6.52, 10.02, 13.52, 17.02, 20.02]  # m
    y_init = [0.0] * 7

    n_masses = len(masses)

    # --- material: Bouc–Wen springs ---
    fy_MN = 15.0
    uy_mm = 200.0
    fy = [fy_MN * 1e6] * (n_masses - 1)        # N
    uy = [uy_mm / 1000.0] * (n_masses - 1)     # m

    # --- time / integration ---
    h_init = 0.0001          # 0.1 ms
    T_max = 0.30             # s
    step = int(T_max / h_init)

    return {
        # geometry & kinematics
        "n_masses": n_masses,
        "masses": masses,
        "x_init": x_init,
        "y_init": y_init,
        "v0_init": -(56.0 / 3.6),   # 56 km/h towards the wall
        "angle_rad": 0.0,
        "d0": 0.01,                 # 1 cm initial gap

        # material
        "fy": fy,
        "uy": uy,

        # contact
        "k_wall": 45.0 * 1e6,       # 45 MN/m
        "cr_wall": 0.8,
        "contact_model": "lankarani-nikravesh",

        # SDOF building (disabled by default in CLI)
        "building_enable": False,
        "building_mass": 0.0,
        "building_zeta": 0.05,
        "building_height": 0.0,
        "building_model": "Linear elastic SDOF",
        "building_uy": 0.01,
        "building_uy_mm": 10.0,
        "building_alpha": 0.05,
        "building_gamma": 0.4,

        # friction
        "mu_s": 0.4,
        "mu_k": 0.3,
        "sigma_0": 1e5,
        "sigma_1": 316.0,
        "sigma_2": 0.4,
        "friction_model": "lugre",

        # Bouc–Wen parameters
        "bw_a": 0.0,
        "bw_A": 1.0,
        "bw_beta": 0.1,
        "bw_gamma": 0.9,
        "bw_n": 8,

        # HHT-α integration
        "alpha_hht": -0.1,
        "newton_tol": 1e-4,
        "max_iter": 50,
        "h_init": h_init,
        "T_max": T_max,
        "step": step,
        "T_int": (0.0, T_max),
    }


def _load_config_file(path: Path) -> Dict[str, Any]:
    """Load a JSON or YAML configuration file into a dict."""
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(path.read_text())

    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is not installed. Install it with "
                "`pip install pyyaml` to use YAML configs."
            )
        return yaml.safe_load(path.read_text())

    raise ValueError(f"Unsupported config format: {suffix}")


def _merge_with_defaults(user_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flat merge: user_cfg overrides defaults.
    (Both are expected to be flat dicts with SimulationParams keys.)
    """
    cfg = _default_config_dict()
    cfg.update(user_cfg)
    return cfg


def _dict_to_simulation_params(cfg: Dict[str, Any]) -> SimulationParams:
    """
    Convert (possibly JSON-loaded) config dict to SimulationParams.
    Handles list -> np.ndarray conversions and fills a few inferred fields.
    """
    # Convert selected keys to numpy arrays
    array_keys = ("masses", "x_init", "y_init", "fy", "uy")
    for key in array_keys:
        if key in cfg:
            cfg[key] = np.asarray(cfg[key], dtype=float)

    # Time info: if step or T_int missing, infer them
    if "h_init" not in cfg:
        raise ValueError("Config must specify 'h_init' (time step in seconds).")
    if "T_max" not in cfg:
        raise ValueError("Config must specify 'T_max' (final time in seconds).")

    h = float(cfg["h_init"])
    T_max = float(cfg["T_max"])

    cfg.setdefault("step", int(T_max / h))
    cfg.setdefault("T_int", (0.0, T_max))

    # All remaining fields are passed directly to SimulationParams
    return SimulationParams(**cfg)


# ---------------------------------------------------------------------------
# Main CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="railway-sim",
        description="Railway Impact Simulator (HHT-α + Bouc–Wen).",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a simulation config file (JSON or YAML). "
             "If omitted, a built-in demo setup is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for simulation results (default: data/processed).",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="Base name for output files (default: config file stem or 'simulation').",
    )

    args = parser.parse_args(argv)

    # --- build configuration dict ---
    if args.config:
        cfg_path = Path(args.config)
        try:
            user_cfg = _load_config_file(cfg_path)
        except Exception as exc:  # pragma: no cover - simple CLI error path
            print(f"[railway-sim] Error loading config '{cfg_path}': {exc}", file=sys.stderr)
            sys.exit(1)
        cfg = _merge_with_defaults(user_cfg or {})
        basename = args.basename or cfg_path.stem
    else:
        cfg = _default_config_dict()
        basename = args.basename or "simulation"

    # --- simulate ---
    try:
        params = _dict_to_simulation_params(cfg)
        df = run_simulation(params)
    except Exception as exc:  # pragma: no cover
        print(f"[railway-sim] Simulation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- write outputs ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{basename}.csv"
    xlsx_path = out_dir / f"{basename}.xlsx"

    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        # Excel export is nice-to-have, not mandatory
        xlsx_path = None

    # --- tiny summary on stdout ---
    max_force = float(df["Impact_Force_MN"].max())
    max_pen = float(df["Penetration_mm"].max())
    max_acc = float(df["Acceleration_g"].max())

    print(f"[railway-sim] Simulation completed.")
    print(f"  Max force        : {max_force:.3f} MN")
    print(f"  Max penetration  : {max_pen:.2f} mm")
    print(f"  Max acceleration : {max_acc:.2f} g")
    print(f"  CSV  written to  : {csv_path}")
    if xlsx_path is not None:
        print(f"  XLSX written to  : {xlsx_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
