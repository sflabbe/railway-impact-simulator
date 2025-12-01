# src/railway_simulator/cli.py

from __future__ import annotations

import json
import logging
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import yaml

from .core.engine import run_simulation
from .core.parametric import ScenarioDefinition, run_parametric_envelope

app = typer.Typer(
    add_completion=False,
    help=(
        "Railway impact simulator CLI\n\n"
        "Run HHT-α/Bouc–Wen impact simulations based on DZSF Bericht 53 (2024).\n"
        "Use 'run' for a single scenario or 'parametric' for speed mixes\n"
        "with envelopes, performance metrics, and optional PDF reports."
    ),
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    """Load a YAML or JSON configuration file into a dict."""
    if not path.is_file():
        raise typer.BadParameter(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    elif suffix == ".json":
        return json.loads(text)
    else:
        raise typer.BadParameter(
            f"Unsupported config extension '{suffix}'. Use .yml, .yaml or .json."
        )


def _parse_speeds_spec(spec: str) -> Tuple[List[float], List[float]]:
    """
    Parse a string like

        "320:0.2,200:0.4,120:0.4"

    into lists speeds_kmh=[320,200,120] and weights=[0.2,0.4,0.4].

    If weights are omitted (e.g. "80,120,160"), all weights = 1.0.
    """
    spec = spec.strip()
    if not spec:
        raise typer.BadParameter("Empty --speeds specification.")

    speeds: List[float] = []
    weights: List[float] = []

    tokens = [t.strip() for t in spec.split(",") if t.strip()]

    has_colon = any(":" in t for t in tokens)

    for tok in tokens:
        if ":" in tok:
            v_str, w_str = tok.split(":", 1)
            v = float(v_str)
            w = float(w_str)
        else:
            v = float(tok)
            w = 1.0
        speeds.append(v)
        weights.append(w)

    # Normalise weights if any colon was used
    if has_colon:
        total_w = sum(weights)
        if total_w <= 0.0:
            raise typer.BadParameter("Sum of weights must be > 0.")
        weights = [w / total_w for w in weights]

    return speeds, weights


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _setup_logger(output_dir: Path, log_stem: str) -> logging.Logger:
    """
    Set up a per-run logger writing to <output_dir>/<log_stem>.log.
    """
    _ensure_output_dir(output_dir)
    logger = logging.getLogger(f"railway_simulator.cli.{log_stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_file = output_dir / f"{log_stem}.log"
    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def _open_file(path: Path, logger: logging.Logger) -> None:
    """
    Try to open a file with the system's default application.
    """
    try:
        if not path.is_file():
            logger.error("File not found, cannot open: %s", path)
            return

        logger.debug("Attempting to open file: %s", path)

        if sys.platform.startswith("darwin"):
            subprocess.Popen(
                ["open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        logger.info("Requested OS to open PDF report: %s", path)

    except Exception:
        logger.exception("Failed to open PDF report.")


def _print_and_log(logger: logging.Logger, msg: str) -> None:
    typer.echo(msg)
    logger.info(msg)


def _ascii_plot(
    x: np.ndarray,
    y: np.ndarray,
    y_label: str,
    x_label: str,
    width: int = 70,
    height: int = 20,
) -> str:
    """
    Very simple ASCII plot: x ∈ [0, max], y ∈ [0, max].
    """
    if len(x) == 0 or len(y) == 0:
        return ""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        x_min, x_max = 0.0, 1.0

    y_min = 0.0
    y_max = float(np.max(y))
    if y_max <= y_min:
        y_max = y_min + 1.0

    grid = [[" " for _ in range(width)] for _ in range(height)]

    for xi, yi in zip(x, y):
        if yi < y_min:
            continue
        cx = (xi - x_min) / (x_max - x_min + 1e-12)
        cy = (yi - y_min) / (y_max - y_min + 1e-12)
        col = int(cx * (width - 1))
        row = int(cy * (height - 1))
        row_idx = height - 1 - row
        if 0 <= row_idx < height and 0 <= col < width:
            grid[row_idx][col] = "*"

    lines = [f"# {y_label} envelope"]
    for r in grid:
        lines.append("".join(r).rstrip())
    lines.append(f"# {x_label} ({int(round(x_min))} – {int(round(x_max))})")
    return "\n".join(lines)


def _compute_single_run_performance(
    results_df: pd.DataFrame,
    wall_time: float,
    params: dict,
) -> dict:
    time_s = results_df["Time_s"].to_numpy()
    if len(time_s) < 2:
        return {
            "wall_time": wall_time,
            "T_max": None,
            "steps": None,
            "dt_mean": None,
            "dt_min": None,
            "dt_max": None,
            "real_time_factor": None,
            "linear_solves": None,
            "n_dof": None,
            "flops_lu": None,
            "mflops": None,
            "mflops_per_s": None,
        }

    dts = np.diff(time_s)
    steps = len(dts)
    T_max = float(time_s[-1] - time_s[0])
    dt_mean = float(np.mean(dts))
    dt_min = float(np.min(dts))
    dt_max = float(np.max(dts))
    real_time_factor = T_max / wall_time if wall_time > 0 else None

    n_masses = int(params.get("n_masses", len(params.get("masses", []))))
    n_dof = 2 * n_masses

    # Try to get actual metrics from DataFrame attributes
    n_lu_actual = results_df.attrs.get("n_lu", None)
    if n_lu_actual is not None and n_lu_actual > 0:
        linear_solves = int(n_lu_actual)
    else:
        # Fallback to heuristic estimate: ~3 Newton iterations per time step
        avg_newton = 3.0
        linear_solves = int(steps * avg_newton)

    flops_per_lu = (2.0 / 3.0) * (n_dof ** 3)
    flops_lu = flops_per_lu * linear_solves
    mflops = flops_lu / 1e6
    mflops_per_s = mflops / wall_time if wall_time > 0 else None

    return {
        "wall_time": wall_time,
        "T_max": T_max,
        "steps": steps,
        "dt_mean": dt_mean,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "real_time_factor": real_time_factor,
        "linear_solves": linear_solves,
        "n_dof": n_dof,
        "flops_lu": flops_lu,
        "mflops": mflops,
        "mflops_per_s": mflops_per_s,
    }


def _compute_parametric_performance(
    envelope_df: pd.DataFrame,
    wall_time: float,
    base_params: dict,
    n_scenarios: int,
) -> dict:
    time_s = envelope_df["Time_s"].to_numpy()
    if len(time_s) < 2:
        return {
            "wall_time": wall_time,
            "T_max": None,
            "steps": None,
            "dt_mean": None,
            "dt_min": None,
            "dt_max": None,
            "real_time_factor": None,
            "linear_solves": None,
            "n_dof": None,
            "flops_lu": None,
            "mflops": None,
            "mflops_per_s": None,
        }

    dts = np.diff(time_s)
    steps = len(dts)
    T_max = float(time_s[-1] - time_s[0])
    dt_mean = float(np.mean(dts))
    dt_min = float(np.min(dts))
    dt_max = float(np.max(dts))
    real_time_factor = T_max / wall_time if wall_time > 0 else None

    n_masses = int(base_params.get("n_masses", len(base_params.get("masses", []))))
    n_dof = 2 * n_masses

    # Heuristic estimate: ~3 Newton iterations per time step per scenario
    avg_newton = 3.0
    linear_solves = int(steps * n_scenarios * avg_newton)

    flops_per_lu = (2.0 / 3.0) * (n_dof ** 3)
    flops_lu = flops_per_lu * linear_solves
    mflops = flops_lu / 1e6
    mflops_per_s = mflops / wall_time if wall_time > 0 else None

    return {
        "wall_time": wall_time,
        "T_max": T_max,
        "steps": steps,
        "dt_mean": dt_mean,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "real_time_factor": real_time_factor,
        "linear_solves": linear_solves,
        "n_dof": n_dof,
        "flops_lu": flops_lu,
        "mflops": mflops,
        "mflops_per_s": mflops_per_s,
    }


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        readable=True,
        help="YAML/JSON configuration file (train, contact, material, etc.).",
    ),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output-dir",
        "-o",
        help="Directory for result files.",
    ),
    prefix: str = typer.Option(
        "",
        "--prefix",
        "-p",
        help="Optional filename prefix for output files.",
    ),
    # Override impact speed using km/h
    speed_kmh: Optional[float] = typer.Option(
        None,
        "--speed-kmh",
        "-v",
        help=(
            "Impact speed in km/h. "
            "If given, overrides v0_init from the config. "
            "Use positive values; the code applies the negative sign "
            "for motion towards the barrier."
        ),
    ),
    # Direct override of v0_init in m/s
    v0_init: Optional[float] = typer.Option(
        None,
        "--v0-init",
        help=(
            "Impact velocity [m/s]. If given, overrides both --speed-kmh and any "
            "v0_init found in the config. "
            "Use a negative value for motion towards the barrier."
        ),
    ),
    ascii_plot: bool = typer.Option(
        False,
        "--ascii-plot",
        help="Print an ASCII plot of impact force vs time to the console.",
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        help="Show a matplotlib window with impact force vs time.",
    ),
    pdf_report: bool = typer.Option(
        False,
        "--pdf-report",
        help="Generate a single-run PDF report and open it.",
    ),
) -> None:
    """
    Run a single impact simulation.

    Examples
    --------
    Use a standard train configuration from YAML and set only the speed:

        railway-sim run \\
          --config configs/trains/ice1_aluminum.yml \\
          --speed-kmh 80 \\
          --output-dir results/ice1_80

    Same run with ASCII plot, matplotlib popup and PDF report:

        railway-sim run \\
          --config configs/trains/ice1_aluminum.yml \\
          --speed-kmh 80 \\
          --output-dir results/ice1_80 \\
          --ascii-plot --plot --pdf-report
    """
    # ------------------------------------------------------------------
    # Setup I/O and logging
    # ------------------------------------------------------------------
    _ensure_output_dir(output_dir)

    filename_prefix = f"{prefix}_" if prefix else ""
    log_stem = f"{filename_prefix}run"
    logger = _setup_logger(output_dir, log_stem)

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    _print_and_log(logger, f"Loading config: {config}")
    params = _load_config(config)

    # ------------------------------------------------------------------
    # Velocity handling (CLI overrides vs config)
    # ------------------------------------------------------------------
    velocity_source = "config"

    if v0_init is not None:
        # Strongest override: explicit v0 in m/s
        params["v0_init"] = float(v0_init)
        velocity_source = "CLI v0-init [m/s]"
        _print_and_log(
            logger,
            f"Overriding v0_init from CLI: v0_init = {params['v0_init']:.4f} m/s",
        )
    elif speed_kmh is not None:
        # Second level: speed in km/h
        params["v0_init"] = -float(speed_kmh) / 3.6
        velocity_source = "CLI speed-kmh"
        _print_and_log(
            logger,
            f"Overriding speed from CLI: speed = {speed_kmh:.2f} km/h "
            f"(v0_init = {params['v0_init']:.4f} m/s towards barrier)",
        )
    else:
        # No CLI override -> rely on config, but normalise type and check existence
        if "v0_init" not in params:
            raise typer.BadParameter(
                "No impact speed specified. Provide v0_init in the config, "
                "--speed-kmh, or --v0-init."
            )
        try:
            params["v0_init"] = float(params["v0_init"])
        except (TypeError, ValueError):
            raise typer.BadParameter(
                f"Config parameter 'v0_init' must be a float-compatible value, "
                f"got {params['v0_init']!r}"
            )
        _print_and_log(
            logger,
            f"Using v0_init from config: v0_init = {params['v0_init']:.4f} m/s",
        )

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    _print_and_log(logger, "Running simulation ...")
    t0 = time.perf_counter()
    results_df = run_simulation(params)
    wall_time = time.perf_counter() - t0

    perf = _compute_single_run_performance(results_df, wall_time, params)
    perf["velocity_source"] = velocity_source

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    csv_path = output_dir / f"{filename_prefix}results.csv"
    _print_and_log(logger, f"Writing time history to {csv_path}")
    results_df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # Performance output (console + log)
    # ------------------------------------------------------------------
    typer.echo("")
    typer.echo("Single-run performance:")
    typer.echo(f"  Wall-clock time       : {perf['wall_time']:.3f} s")
    typer.echo(f"  Simulated time span   : {perf['T_max']:.6f} s")
    typer.echo(f"  Time steps            : {perf['steps']}")
    typer.echo(f"  Mean Δt               : {perf['dt_mean']:.6e} s")
    typer.echo(
        f"  Min Δt / max Δt       : {perf['dt_min']:.6e} s / {perf['dt_max']:.6e} s"
    )
    typer.echo(f"  Real-time factor      : {perf['real_time_factor']:.2f}x")
    typer.echo(f"  Linear solves (LU)    : {perf['linear_solves']}, n_dof ≈ {perf['n_dof']}")
    typer.echo(
        f"  Estimated FLOPs (LU)  : {perf['mflops']:.2f} MFLOP"
    )
    typer.echo(
        f"  Estimated rate        : {perf['mflops_per_s']:.2f} MFLOP/s"
    )
    typer.echo(f"  Velocity source       : {perf['velocity_source']}")

    logger.info("Single-run performance: %s", perf)

    # ------------------------------------------------------------------
    # PDF report (before any blocking plot)
    # ------------------------------------------------------------------
    if pdf_report:
        from .core.report import generate_single_run_report

        pdf_path = output_dir / f"{filename_prefix}report.pdf"
        logger.info("Generating PDF report: %s", pdf_path)
        try:
            generate_single_run_report(results_df, perf, params, pdf_path)
            if pdf_path.is_file():
                logger.info("PDF report successfully written to %s", pdf_path)
                _open_file(pdf_path, logger)
            else:
                logger.error("PDF report was not created: %s", pdf_path)
        except Exception:
            logger.exception("Failed to generate single-run PDF report.")

    # ------------------------------------------------------------------
    # ASCII plot
    # ------------------------------------------------------------------
    if ascii_plot:
        typer.echo("")
        typer.echo("ASCII impact plot (Impact_Force_MN vs Time_ms):")
        ascii_str = _ascii_plot(
            results_df["Time_ms"].to_numpy(),
            results_df["Impact_Force_MN"].to_numpy(),
            "Impact_Force_MN",
            "Time [ms]",
        )
        typer.echo(ascii_str)
        logger.info("ASCII plot:\n%s", ascii_str)

    # ------------------------------------------------------------------
    # Matplotlib plot (blocking)
    # ------------------------------------------------------------------
    if plot:
        t_ms = results_df["Time_ms"].to_numpy()
        F = results_df["Impact_Force_MN"].to_numpy()
        fig, ax = plt.subplots()
        ax.plot(t_ms, F, label="Impact_Force_MN")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Impact force [MN]")
        ax.set_title("Impact force vs time")
        ax.grid(True)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend()
        plt.show()

    # ------------------------------------------------------------------
    # Final log
    # ------------------------------------------------------------------
    log_file = output_dir / f"{log_stem}.log"
    typer.echo(f"\nDetailed log written to {log_file}")
    logger.info("Run completed.")


@app.command()
def parametric(
    base_config: Path = typer.Option(
        ...,
        "--base-config",
        "-b",
        exists=True,
        readable=True,
        help="Base YAML/JSON configuration file (train, contact, material, etc.).",
    ),
    speeds: str = typer.Option(
        ...,
        "--speeds",
        "-s",
        help=(
            'Speed/weight specification. Examples:\n'
            '  "320:0.2,200:0.4,120:0.4"  (TGV/IC/cargo mix)\n'
            '  "80,120,160"               (equal weights)\n'
            "Speeds in km/h; weights are optional and normalised."
        ),
    ),
    quantity: str = typer.Option(
        "Impact_Force_MN",
        "--quantity",
        "-q",
        help="Result column to envelope (e.g. Impact_Force_MN, Acceleration_g, ...).",
    ),
    output_dir: Path = typer.Option(
        Path("results_parametric"),
        "--output-dir",
        "-o",
        help="Directory for envelope result files.",
    ),
    prefix: str = typer.Option(
        "mix",
        "--prefix",
        "-p",
        help="Filename prefix for parametric output.",
    ),
    ascii_plot: bool = typer.Option(
        False,
        "--ascii-plot",
        help="Print an ASCII plot of the envelope to the console.",
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        help="Show a matplotlib window with the envelope.",
    ),
    pdf_report: bool = typer.Option(
        False,
        "--pdf-report",
        help="Generate a parametric PDF report and open it.",
    ),
) -> None:
    """
    Run a speed-based parametric study and compute an envelope
    ('Umhüllende') and a weighted mean history.

    Example: TGV / IC / freight line mix
    ------------------------------------
    20 % TGV at 320 km/h, 40 % IC at 200 km/h, 40 % freight at 120 km/h,
    envelope on impact force and PDF report:

        railway-sim parametric \\
          --base-config configs/ice1_80kmh.yml \\
          --speeds "320:0.2,200:0.4,120:0.4" \\
          --quantity Impact_Force_MN \\
          --output-dir results_parametric/track_mix \\
          --prefix track_mix \\
          --ascii-plot --pdf-report
    """
    _ensure_output_dir(output_dir)

    base_name = f"{prefix}_{quantity}" if prefix else quantity
    log_stem = f"{base_name}_parametric"
    logger = _setup_logger(output_dir, log_stem)

    _print_and_log(logger, f"Loading base config: {base_config}")
    base_params = _load_config(base_config)

    _print_and_log(logger, f"Parsing speeds specification: {speeds}")
    speeds_kmh, weights = _parse_speeds_spec(speeds)

    _print_and_log(logger, "Building scenarios:")
    scenarios: List[ScenarioDefinition] = []
    for v_kmh, w in zip(speeds_kmh, weights):
        name = f"v{int(round(v_kmh))}"
        msg = f"  - {name}: {v_kmh:.1f} km/h, weight = {w:.3f}"
        _print_and_log(logger, msg)

        params_i = dict(base_params)
        params_i["v0_init"] = -v_kmh / 3.6  # m/s, negative towards barrier

        scen = ScenarioDefinition(
            name=name,
            params=params_i,
            weight=w,
            meta={"speed_kmh": v_kmh},
        )
        scenarios.append(scen)

    _print_and_log(logger, "Running parametric envelope ...")
    t0 = time.perf_counter()
    result = run_parametric_envelope(scenarios, quantity=quantity)

    extra: Any = None
    if isinstance(result, tuple):
        if len(result) == 3:
            envelope_df, summary_df, extra = result
        elif len(result) == 2:
            envelope_df, summary_df = result
        else:
            envelope_df = result[0]
            summary_df = result[1] if len(result) > 1 else pd.DataFrame()
    else:
        envelope_df = result
        summary_df = pd.DataFrame()

    wall_time = time.perf_counter() - t0

    perf = _compute_parametric_performance(
        envelope_df, wall_time, base_params, n_scenarios=len(scenarios)
    )
    if isinstance(extra, dict):
        # If engine already returns more accurate metrics, prefer them
        for key, val in extra.items():
            if key in perf and val is not None:
                perf[key] = val
        # Refine performance metrics using summary_df if available
    if not summary_df.empty:
        if "n_dof" in summary_df.columns:
            perf["n_dof"] = int(summary_df["n_dof"].max())

        if "n_lu" in summary_df.columns:
            total_lu = int(summary_df["n_lu"].sum())
            perf["linear_solves"] = total_lu

            n_dof = perf.get("n_dof", 0) or 0
            if n_dof > 0 and total_lu > 0:
                flops_per_lu = (2.0 / 3.0) * (n_dof ** 3)
                flops_lu = flops_per_lu * total_lu
                mflops = flops_lu / 1e6
                perf["flops_lu"] = flops_lu
                perf["mflops"] = mflops
                perf["mflops_per_s"] = (
                    mflops / perf["wall_time"] if perf["wall_time"] > 0 else None
                )


    base = base_name
    env_csv = output_dir / f"{base}_envelope.csv"
    sum_csv = output_dir / f"{base}_summary.csv"

    _print_and_log(logger, f"Writing envelope time history to {env_csv}")
    envelope_df.to_csv(env_csv, index=False)

    _print_and_log(logger, f"Writing scenario summary to {sum_csv}")
    summary_df.to_csv(sum_csv, index=False)

    # Performance output
    typer.echo("")
    typer.echo("Parametric study performance:")
    typer.echo(f"  Wall-clock time       : {perf['wall_time']:.3f} s")
    typer.echo(f"  Simulated time span   : {perf['T_max']:.6f} s")
    typer.echo(f"  Time steps            : {perf['steps']}")
    typer.echo(f"  Mean Δt               : {perf['dt_mean']:.6e} s")
    typer.echo(
        f"  Min Δt / max Δt       : {perf['dt_min']:.6e} s / {perf['dt_max']:.6e} s"
    )
    typer.echo(f"  Real-time factor      : {perf['real_time_factor']:.2f}x")
    typer.echo(
        f"  Linear solves (LU)    : {perf['linear_solves']}, "
        f"n_dof ≈ {perf['n_dof']}"
    )
    typer.echo(
        f"  Estimated FLOPs (LU)  : {perf['mflops']:.2f} MFLOP"
    )
    typer.echo(
        f"  Estimated rate        : {perf['mflops_per_s']:.2f} MFLOP/s"
    )

    logger.info("Parametric performance: %s", perf)
    
    # PDF report FIRST – so it exists even if the plot blocks
    if pdf_report:
        from .core.report import generate_parametric_report

        pdf_path = output_dir / f"{base}_report.pdf"
        logger.info("Generating parametric PDF report: %s", pdf_path)
        try:
            generate_parametric_report(
                envelope_df, summary_df, perf, quantity, pdf_path
            )
            if pdf_path.is_file():
                logger.info(
                    "Parametric PDF report successfully written to %s", pdf_path
                )
                _open_file(pdf_path, logger)
            else:
                logger.error("Parametric PDF report was not created: %s", pdf_path)
        except Exception:
            logger.exception("Failed to generate parametric PDF report.")

    # ASCII envelope plot
    if ascii_plot:
        typer.echo("")
        typer.echo(f"ASCII envelope plot ({quantity} vs Time_ms):")

        # Resolve correct column name, e.g. "Impact_Force_MN_envelope"
        if quantity in envelope_df.columns:
            y_col = quantity
        else:
            env_col = f"{quantity}_envelope"
            if env_col in envelope_df.columns:
                y_col = env_col
            else:
                # Fallback: first non-time column
                non_time_cols = [
                    c for c in envelope_df.columns
                    if c not in ("Time_s", "Time_ms")
                ]
                if not non_time_cols:
                    raise RuntimeError(
                        f"Could not find column for quantity '{quantity}' in envelope_df."
                    )
                y_col = non_time_cols[0]

        ascii_str = _ascii_plot(
            envelope_df["Time_ms"].to_numpy(),
            envelope_df[y_col].to_numpy(),
            y_col,
            "Time [ms]",
        )
        typer.echo(ascii_str)
        logger.info("ASCII envelope plot (column=%s):\n%s", y_col, ascii_str)

    # Matplotlib envelope plot
    if plot:
        # same column resolution logic as above
        if quantity in envelope_df.columns:
            y_col = quantity
        else:
            env_col = f"{quantity}_envelope"
            if env_col in envelope_df.columns:
                y_col = env_col
            else:
                non_time_cols = [
                    c for c in envelope_df.columns
                    if c not in ("Time_s", "Time_ms")
                ]
                if not non_time_cols:
                    raise RuntimeError(
                        f"Could not find column for quantity '{quantity}' in envelope_df."
                    )
                y_col = non_time_cols[0]

        t_ms = envelope_df["Time_ms"].to_numpy()
        y = envelope_df[y_col].to_numpy()
        fig, ax = plt.subplots()
        ax.plot(t_ms, y, label=f"Envelope {y_col}")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} envelope vs time")
        ax.grid(True)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend()
        plt.show()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
