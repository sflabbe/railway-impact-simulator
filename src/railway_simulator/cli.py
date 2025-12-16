# src/railway_simulator/cli.py

from __future__ import annotations

import json
import logging
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Studies commands (convergence / sensitivity / fixed DIF, etc.)
from .studies.cli import register_study_commands
register_study_commands(app)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Conversion factor from km/h to m/s
KMH_TO_MS = 3.6


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Server address (use 0.0.0.0 to expose)."),
    port: int = typer.Option(8501, help="Server port."),
    headless: bool = typer.Option(False, help="Run Streamlit in headless mode."),
) -> None:
    """Launch the Streamlit UI (requires the optional 'ui' dependencies)."""
    app_py = Path(__file__).resolve().parent / "core" / "app.py"

    # Provide a friendly error if streamlit isn't installed
    try:
        import streamlit  # noqa: F401
    except Exception:
        raise typer.BadParameter(
            "Streamlit is not installed. Install UI extras with:\n\n"
            "  pip install 'railway-impact-simulator[ui]'\n"
        )

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_py),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    if headless:
        cmd += ["--server.headless", "true"]

    raise typer.Exit(subprocess.call(cmd))

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _speed_kmh_to_v0_init(speed_kmh: float) -> float:
    """
    Convert speed in km/h to initial velocity in m/s towards barrier.

    Args:
        speed_kmh: Speed magnitude in km/h (positive value)

    Returns:
        Velocity in m/s (negative, towards barrier)
    """
    return -float(speed_kmh) / KMH_TO_MS


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


def _resolve_envelope_column(
    envelope_df: pd.DataFrame,
    quantity: str,
) -> str:
    """
    Resolve the correct column name for the given quantity in an envelope DataFrame.

    Tries in order:
    1. Exact match: quantity
    2. With suffix: f"{quantity}_envelope"
    3. Fallback: first non-time column

    Returns the resolved column name.
    Raises RuntimeError if no suitable column is found.
    """
    if quantity in envelope_df.columns:
        return quantity

    env_col = f"{quantity}_envelope"
    if env_col in envelope_df.columns:
        return env_col

    # Fallback: first non-time column
    non_time_cols = [
        c for c in envelope_df.columns
        if c not in ("Time_s", "Time_ms")
    ]
    if not non_time_cols:
        raise RuntimeError(
            f"Could not find column for quantity '{quantity}' in envelope_df."
        )
    return non_time_cols[0]


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


def _compute_performance_metrics(
    time_series_df: pd.DataFrame,
    wall_time: float,
    params: dict,
    n_scenarios: int = 1,
) -> dict:
    """
    Compute performance metrics from simulation results.

    Args:
        time_series_df: DataFrame with Time_s column (results or envelope)
        wall_time: Wall clock time in seconds
        params: Simulation parameters dict
        n_scenarios: Number of scenarios run (1 for single run, >1 for parametric)

    Returns:
        Dictionary of performance metrics
    """
    empty_metrics = {
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

    time_s = time_series_df["Time_s"].to_numpy()
    if len(time_s) < 2:
        return empty_metrics

    # Time-stepping metrics
    dts = np.diff(time_s)
    steps = len(dts)
    T_max = float(time_s[-1] - time_s[0])
    dt_mean = float(np.mean(dts))
    dt_min = float(np.min(dts))
    dt_max = float(np.max(dts))
    real_time_factor = T_max / wall_time if wall_time > 0 else None

    # DOF calculation
    n_masses = int(params.get("n_masses", len(params.get("masses", []))))
    n_dof = 2 * n_masses

    # Linear solves estimation
    # For single runs, try to get actual count from DataFrame attributes
    n_lu_actual = time_series_df.attrs.get("n_lu", None)
    if n_scenarios == 1 and n_lu_actual is not None and n_lu_actual > 0:
        linear_solves = int(n_lu_actual)
    else:
        # Fallback heuristic: ~3 Newton iterations per time step per scenario
        avg_newton = 3.0
        linear_solves = int(steps * n_scenarios * avg_newton)

    # FLOPS estimation
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


def _compute_single_run_performance(
    results_df: pd.DataFrame,
    wall_time: float,
    params: dict,
) -> dict:
    """Backward-compatible wrapper for single run performance computation."""
    return _compute_performance_metrics(results_df, wall_time, params, n_scenarios=1)


def _compute_parametric_performance(
    envelope_df: pd.DataFrame,
    wall_time: float,
    base_params: Dict[str, Any],
    n_scenarios: int,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for parametric performance computation."""
    return _compute_performance_metrics(envelope_df, wall_time, base_params, n_scenarios=n_scenarios)


# ----------------------------------------------------------------------
# Run Command Helpers
# ----------------------------------------------------------------------


def _handle_velocity_override(
    params: Dict[str, Any],
    v0_init: Optional[float],
    speed_kmh: Optional[float],
    logger: logging.Logger,
) -> str:
    """
    Handle velocity CLI overrides and validate configuration.

    Args:
        params: Configuration parameters (modified in-place)
        v0_init: Optional velocity override in m/s
        speed_kmh: Optional speed override in km/h
        logger: Logger instance

    Returns:
        String describing the velocity source

    Raises:
        typer.BadParameter: If velocity is not specified or invalid
    """
    if v0_init is not None:
        # Strongest override: explicit v0 in m/s
        params["v0_init"] = float(v0_init)
        _print_and_log(
            logger,
            f"Overriding v0_init from CLI: v0_init = {params['v0_init']:.4f} m/s",
        )
        return "CLI v0-init [m/s]"

    if speed_kmh is not None:
        # Second level: speed in km/h
        params["v0_init"] = _speed_kmh_to_v0_init(speed_kmh)
        _print_and_log(
            logger,
            f"Overriding speed from CLI: speed = {speed_kmh:.2f} km/h "
            f"(v0_init = {params['v0_init']:.4f} m/s towards barrier)",
        )
        return "CLI speed-kmh"

    # No CLI override -> rely on config, but normalize type and check existence
    if "v0_init" not in params:
        raise typer.BadParameter(
            "No impact speed specified. Provide v0_init in the config, "
            "--speed-kmh, or --v0-init."
        )
    try:
        params["v0_init"] = float(params["v0_init"])
    except (TypeError, ValueError) as e:
        raise typer.BadParameter(
            f"Config parameter 'v0_init' must be a float-compatible value, "
            f"got {params['v0_init']!r}"
        ) from e

    _print_and_log(
        logger,
        f"Using v0_init from config: v0_init = {params['v0_init']:.4f} m/s",
    )
    return "config"


def _print_performance_metrics(
    perf: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Print performance metrics to console and log.

    Args:
        perf: Performance metrics dictionary
        logger: Logger instance
    """
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


def _generate_pdf_report(
    results_df: pd.DataFrame,
    perf: Dict[str, Any],
    params: Dict[str, Any],
    output_dir: Path,
    filename_prefix: str,
    logger: logging.Logger,
) -> None:
    """
    Generate and open PDF report for single run.

    Args:
        results_df: Simulation results DataFrame
        perf: Performance metrics
        params: Simulation parameters
        output_dir: Output directory
        filename_prefix: File name prefix
        logger: Logger instance
    """
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


def _show_ascii_plot(
    results_df: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    """
    Display ASCII plot of impact force vs time.

    Args:
        results_df: Simulation results DataFrame
        logger: Logger instance
    """
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


def _show_matplotlib_plot(results_df: pd.DataFrame) -> None:
    """
    Display matplotlib plot of impact force vs time.

    Args:
        results_df: Simulation results DataFrame
    """
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


# ----------------------------------------------------------------------
# Parametric Command Helpers
# ----------------------------------------------------------------------


def _build_speed_scenarios(
    base_params: Dict[str, Any],
    speeds_kmh: List[float],
    weights: List[float],
    logger: logging.Logger,
) -> List[ScenarioDefinition]:
    """
    Build scenario definitions for parametric speed study.

    Args:
        base_params: Base configuration parameters
        speeds_kmh: List of speeds in km/h
        weights: List of weights (probabilities)
        logger: Logger instance

    Returns:
        List of ScenarioDefinition objects
    """
    _print_and_log(logger, "Building scenarios:")
    scenarios: List[ScenarioDefinition] = []
    for v_kmh, w in zip(speeds_kmh, weights):
        name = f"v{int(round(v_kmh))}"
        msg = f"  - {name}: {v_kmh:.1f} km/h, weight = {w:.3f}"
        _print_and_log(logger, msg)

        params_i = dict(base_params)
        params_i["v0_init"] = _speed_kmh_to_v0_init(v_kmh)

        scen = ScenarioDefinition(
            name=name,
            params=params_i,
            weight=w,
            meta={"speed_kmh": v_kmh},
        )
        scenarios.append(scen)
    return scenarios


def _unpack_parametric_result(
    result: Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Unpack result from run_parametric_envelope which can return varying formats.

    Args:
        result: Result from run_parametric_envelope

    Returns:
        Tuple of (envelope_df, summary_df, extra_dict)
    """
    extra: Optional[Dict[str, Any]] = None

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

    return envelope_df, summary_df, extra


def _refine_parametric_performance(
    perf: Dict[str, Any],
    extra: Optional[Dict[str, Any]],
    summary_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Refine parametric performance metrics with extra info and summary data.

    Args:
        perf: Initial performance metrics
        extra: Extra metrics dictionary from engine
        summary_df: Summary DataFrame from parametric run

    Returns:
        Refined performance metrics dictionary
    """
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

    return perf


def _write_parametric_outputs(
    envelope_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    base_name: str,
    logger: logging.Logger,
) -> None:
    """
    Write parametric study outputs to CSV files.

    Args:
        envelope_df: Envelope DataFrame
        summary_df: Summary DataFrame
        output_dir: Output directory
        base_name: Base name for output files
        logger: Logger instance
    """
    env_path = output_dir / f"{base_name}_envelope.csv"
    _print_and_log(logger, f"Writing envelope to {env_path}")
    envelope_df.to_csv(env_path, index=False)

    if not summary_df.empty:
        summ_path = output_dir / f"{base_name}_summary.csv"
        _print_and_log(logger, f"Writing summary to {summ_path}")
        summary_df.to_csv(summ_path, index=False)


def _print_parametric_performance(
    perf: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Print parametric performance metrics to console and log.

    Args:
        perf: Performance metrics dictionary
        logger: Logger instance
    """
    typer.echo("")
    typer.echo("Parametric performance:")
    typer.echo(f"  Wall-clock time       : {perf['wall_time']:.3f} s")
    typer.echo(f"  Linear solves (LU)    : {perf['linear_solves']}, n_dof ≈ {perf['n_dof']}")
    typer.echo(f"  Estimated FLOPs (LU)  : {perf['mflops']:.2f} MFLOP")
    typer.echo(f"  Estimated rate        : {perf['mflops_per_s']:.2f} MFLOP/s")

    logger.info("Parametric performance: %s", perf)


def _generate_parametric_pdf_report(
    envelope_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    perf: Dict[str, Any],
    base_params: Dict[str, Any],
    output_dir: Path,
    base_name: str,
    quantity: str,
    logger: logging.Logger,
) -> None:
    """
    Generate and open PDF report for parametric study.

    Args:
        envelope_df: Envelope DataFrame
        summary_df: Summary DataFrame
        perf: Performance metrics
        base_params: Base configuration parameters
        output_dir: Output directory
        base_name: Base name for output files
        quantity: Quantity name
        logger: Logger instance
    """
    from .core.report import generate_parametric_report

    pdf_path = output_dir / f"{base_name}_report.pdf"
    logger.info("Generating parametric PDF report: %s", pdf_path)
    try:
        generate_parametric_report(
            envelope_df,
            summary_df,
            perf,
            base_params,
            quantity,
            pdf_path,
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
    velocity_source = _handle_velocity_override(params, v0_init, speed_kmh, logger)

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
    _print_performance_metrics(perf, logger)

    # ------------------------------------------------------------------
    # PDF report (before any blocking plot)
    # ------------------------------------------------------------------
    if pdf_report:
        _generate_pdf_report(results_df, perf, params, output_dir, filename_prefix, logger)

    # ------------------------------------------------------------------
    # ASCII plot
    # ------------------------------------------------------------------
    if ascii_plot:
        _show_ascii_plot(results_df, logger)

    # ------------------------------------------------------------------
    # Matplotlib plot (blocking)
    # ------------------------------------------------------------------
    if plot:
        _show_matplotlib_plot(results_df)

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

    scenarios = _build_speed_scenarios(base_params, speeds_kmh, weights, logger)

    _print_and_log(logger, "Running parametric envelope ...")
    t0 = time.perf_counter()
    result = run_parametric_envelope(scenarios, quantity=quantity)
    wall_time = time.perf_counter() - t0

    envelope_df, summary_df, extra = _unpack_parametric_result(result)

    perf = _compute_parametric_performance(
        envelope_df, wall_time, base_params, n_scenarios=len(scenarios)
    )
    perf = _refine_parametric_performance(perf, extra, summary_df)

    _write_parametric_outputs(envelope_df, summary_df, output_dir, base_name, logger)

    # Performance output
    _print_parametric_performance(perf, logger)

    # PDF report FIRST – so it exists even if the plot blocks
    if pdf_report:
        _generate_parametric_pdf_report(
            envelope_df, summary_df, perf, base_params, output_dir, base_name, quantity, logger
        )

    # ASCII envelope plot
    if ascii_plot:
        typer.echo("")
        typer.echo(f"ASCII envelope plot ({quantity} vs Time_ms):")

        y_col = _resolve_envelope_column(envelope_df, quantity)

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
        y_col = _resolve_envelope_column(envelope_df, quantity)

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
