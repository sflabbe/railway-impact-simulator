"""
Typer CLI commands for studies.

Imported and registered from `railway_simulator.cli`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml

import re


def _parse_floats_csv(s: str) -> List[float]:
    """Parse comma/space-separated floats, e.g. "1e-4,5e-5"."""
    s = (s or '').strip()
    if not s:
        return []
    parts = [p for p in re.split(r'[\s,]+', s) if p]
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise typer.BadParameter(f'Could not parse floats from: {s!r}') from e

# Backwards-compatible alias used throughout this module
parse_floats_csv = _parse_floats_csv


def _load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise typer.BadParameter("YAML config must be a mapping/dict at the top level.")
    return cfg


def register_study_commands(app: typer.Typer) -> None:
    @app.command("convergence")
    def convergence_cmd(
        config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Base config YAML"),
        dts: str = typer.Option("2e-4,1e-4,5e-5,2.5e-5", "--dts", help="Comma/space-separated dt values [s]"),
        quantity: str = typer.Option("Impact_Force_MN", "--quantity", help="Output column to analyze"),
        out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
        save_timeseries: bool = typer.Option(False, "--save-timeseries", help="Save timeseries CSVs for each run"),
    ) -> None:
        """Run time-step convergence study."""
        from .convergence import run_convergence_study

        cfg = _load_config(config)
        dt_list = parse_floats_csv(dts)
        summary = run_convergence_study(
            cfg, dt_list, quantity=quantity, out_dir=out, save_timeseries=save_timeseries
        )
        typer.echo(summary.to_string(index=False))
        typer.echo(f"Saved to: {out}")

    @app.command("sensitivity")
    def sensitivity_cmd(
        config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Base config YAML"),
        param_path: str = typer.Argument(..., help="Parameter path, e.g. k_wall or fy[0]"),
        values: str = typer.Option(..., "--values", help="Comma/space-separated values"),
        quantity: str = typer.Option("Impact_Force_MN", "--quantity", help="Output column to analyze"),
        out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
        save_timeseries: bool = typer.Option(False, "--save-timeseries", help="Save timeseries CSVs for each run"),
    ) -> None:
        """Run single-parameter sensitivity study."""
        from .sensitivity import run_sensitivity_study

        cfg = _load_config(config)
        val_list = parse_floats_csv(values)
        summary = run_sensitivity_study(
            cfg,
            param_path=param_path,
            values=val_list,
            quantity=quantity,
            out_dir=out,
            save_timeseries=save_timeseries,
        )
        typer.echo(summary.to_string(index=False))
        typer.echo(f"Saved to: {out}")

    @app.command("strain-rate-sensitivity")
    def strain_rate_sensitivity_cmd(
        config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Base config YAML"),
        difs: str = typer.Option("1.0,1.1,1.2", "--difs", help="Comma/space-separated DIF multipliers"),
        k_path: str = typer.Option("k_wall", "--k-path", help="Stiffness-like parameter path to scale"),
        quantity: str = typer.Option("Impact_Force_MN", "--quantity", help="Output column to analyze"),
        out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
        save_timeseries: bool = typer.Option(False, "--save-timeseries", help="Save timeseries CSVs for each run"),
    ) -> None:
        """Fixed DIF study by scaling a stiffness parameter (default: k_wall)."""
        from .strain_rate_sensitivity import run_fixed_dif_sensitivity

        cfg = _load_config(config)
        dif_list = parse_floats_csv(difs)
        summary = run_fixed_dif_sensitivity(
            cfg,
            dif_list,
            k_path=k_path,
            quantity=quantity,
            out_dir=out,
            save_timeseries=save_timeseries,
        )
        typer.echo(summary.to_string(index=False))
        typer.echo(f"Saved to: {out}")

    @app.command("numerics-sensitivity")
    def numerics_sensitivity_cmd(
        config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Base config YAML"),
        dts: Optional[str] = typer.Option(None, "--dts", help="Comma/space-separated dt values [s]"),
        alphas: Optional[str] = typer.Option(None, "--alphas", help="Comma/space-separated alpha_hht values"),
        tols: Optional[str] = typer.Option(None, "--tols", help="Comma/space-separated newton_tol values"),
        quantity: str = typer.Option("Impact_Force_MN", "--quantity", help="Output column to analyze"),
        out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
        save_timeseries: bool = typer.Option(False, "--save-timeseries", help="Save timeseries CSVs for each run"),
    ) -> None:
        """Sweep dt/alpha/tolerance combinations and summarize numerical sensitivity."""
        from .numerics_sensitivity import run_numerics_sensitivity

        cfg = _load_config(config)
        dt_list = parse_floats_csv(dts) if dts else None
        alpha_list = parse_floats_csv(alphas) if alphas else None
        tol_list = parse_floats_csv(tols) if tols else None

        summary = run_numerics_sensitivity(
            cfg,
            dt_values=dt_list,
            alpha_values=alpha_list,
            tol_values=tol_list,
            quantity=quantity,
            out_dir=out,
            save_timeseries=save_timeseries,
        )
        typer.echo(summary.to_string(index=False))
        typer.echo(f"Saved to: {out}")