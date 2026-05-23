#!/usr/bin/env python3
"""Deprecated legacy generator for deterministic clearance-runout tables.

TODO(legacy): keep this script frozen for reproducibility. New production
workflows should use neutral project/study naming instead.

This script is intentionally not a hazard Monte Carlo.  It evaluates prescribed
scenarios (v0, wall offset a, friction mu, derailment angle beta_d), computes
post-runout impact speed and wall-normal velocity, and optionally attaches a
single-mass response surrogate for F_peak and F_eq.

Example:
    PYTHONPATH=src python scripts/legacy/generate_stempi_runout_table.py \
        --config configs/ice1_powercar.yml \
        --response-model single-mass-surrogate \
        --outdir results/stempi_runout
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from railway_simulator.config.loader import load_simulation_config
from railway_simulator.core.engine import run_simulation
from railway_simulator.hazard.deterministic import (
    DEFAULT_GUIDANCE_THRESHOLD_MS,
    LogPchipResponseProvider,
    compact_summary_table,
    deterministic_runout_grid,
)
from railway_simulator.hazard.lateral import (
    SDOFSettings,
    build_single_mass_reference_params,
    single_mass_reference_feq_MN,
)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def response_grid_values(text: str) -> list[float]:
    vals = parse_float_list(text)
    if sorted(vals) != vals or len(set(vals)) != len(vals):
        raise ValueError("response grid must be strictly increasing")
    return vals


def build_single_mass_response_provider(
    *,
    config_path: Path,
    vn_grid_ms: Sequence[float],
    outdir: Path,
    sdof: SDOFSettings,
    force_recompute: bool = False,
) -> LogPchipResponseProvider:
    """Run or load the single-mass response grid and return a surrogate."""
    outdir.mkdir(parents=True, exist_ok=True)
    grid_csv = outdir / "single_mass_response_grid.csv"
    if grid_csv.exists() and not force_recompute:
        grid_df = pd.read_csv(grid_csv)
    else:
        base_cfg = load_simulation_config(config_path)
        rows = []
        for vn in vn_grid_ms:
            params = build_single_mass_reference_params(
                base_cfg,
                v_n_ms=float(vn),
                mass_kg=13_000.0,
                # Keep a fixed complete enough contact window for the single-mass reference.
                t_max_s=0.45,
            )
            df = run_simulation(params, emit_peak_diagnostics=False)
            rows.append(
                {
                    "v_n_ms": float(vn),
                    "f_peak_MN": float(df["Impact_Force_MN"].max()),
                    "f_eq_MN": float(single_mass_reference_feq_MN(df, sdof=sdof)),
                    "n_steps": int(len(df)),
                }
            )
        grid_df = pd.DataFrame(rows)
        grid_df.to_csv(grid_csv, index=False)
    return LogPchipResponseProvider(
        grid_df["v_n_ms"].to_numpy(),
        grid_df["f_peak_MN"].to_numpy(),
        grid_df["f_eq_MN"].to_numpy(),
        name="single_mass_reference_log_pchip",
    )


def write_markdown_table(df: pd.DataFrame, path: Path, max_rows: int | None = None) -> None:
    view = df if max_rows is None else df.head(max_rows)
    path.write_text(view.to_markdown(index=False), encoding="utf-8")


def make_beta_crit_plot(outdir: Path, threshold_ms: float) -> None:
    v_kmh = np.linspace(5.0, 200.0, 600)
    v_ms = v_kmh / 3.6
    beta_crit = np.full_like(v_ms, np.nan)
    mask = v_ms >= threshold_ms
    beta_crit[mask] = np.degrees(np.arcsin(threshold_ms / v_ms[mask]))

    fig = plt.figure(figsize=(7.5, 4.8))
    plt.plot(v_kmh, beta_crit)
    for speed in [20, 40, 80, 120, 160]:
        v = speed / 3.6
        if v >= threshold_ms:
            b = math.degrees(math.asin(threshold_ms / v))
            plt.plot([speed], [b], marker="o")
            plt.text(speed, b, f" {speed} km/h: {b:.1f}°", va="bottom")
    plt.xlabel(r"post-runout impact speed $v_{imp}$ [km/h]")
    plt.ylabel(r"critical angle $\beta_{crit}$ [deg]")
    plt.title(r"Guidance threshold angle: $\beta_{crit}=\arcsin(1.7/v_{imp})$")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(outdir / "beta_crit_vs_vimp.png", dpi=180)
    plt.close(fig)


def make_vn_grid_plot(df: pd.DataFrame, outdir: Path) -> None:
    # Compact plot: normal velocity vs beta for each initial speed at mu=0.3 and a=3m.
    sub = df[(df["mu"] == 0.3) & (df["a_m"] == 3.0)].copy()
    if sub.empty:
        return
    fig = plt.figure(figsize=(7.5, 4.8))
    for v0, g in sub.groupby("v0_kmh"):
        g = g.sort_values("beta_d_deg")
        plt.plot(g["beta_d_deg"], g["v_n_ms"], marker="o", label=f"v0={int(v0)} km/h")
    plt.axhline(DEFAULT_GUIDANCE_THRESHOLD_MS, linestyle="--", label="1.7 m/s threshold")
    plt.xlabel(r"prescribed derailment angle $\beta_d$ [deg]")
    plt.ylabel(r"normal velocity $v_n$ [m/s]")
    plt.title(r"Deterministic normal velocity, $a=3$ m, $\mu=0.3$")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(outdir / "normal_velocity_grid_a3_mu03.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ice1_powercar.yml")
    parser.add_argument("--vehicle", default="ICE1 powercar")
    parser.add_argument("--speeds", default="40,80,120,160", help="Comma-separated v0 values [km/h]")
    parser.add_argument("--distances", default="1,2,3,5,10", help="Comma-separated wall offsets [m]")
    parser.add_argument("--mu", default="0.3,0.5", help="Comma-separated friction coefficients")
    parser.add_argument("--betas", default="2,5,10,20", help="Comma-separated beta_d values [deg]")
    parser.add_argument("--threshold-ms", type=float, default=DEFAULT_GUIDANCE_THRESHOLD_MS)
    parser.add_argument(
        "--response-model",
        choices=("none", "single-mass-surrogate"),
        default="single-mass-surrogate",
        help="Attach F_peak/F_eq using the specified response model.",
    )
    parser.add_argument(
        "--response-grid-ms",
        default="0.5,1.0,1.7,2.5,5.0,7.5,10.0,15.0,20.0",
        help="Comma-separated v_n grid for the single-mass response surrogate [m/s]",
    )
    parser.add_argument("--force-response-recompute", action="store_true")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "stempi_runout")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    response_provider = None
    response_meta = {"response_model": "none"}
    if args.response_model == "single-mass-surrogate":
        sdof = SDOFSettings(Tn_s=0.100, zeta=0.05)
        response_provider = build_single_mass_response_provider(
            config_path=args.config,
            vn_grid_ms=response_grid_values(args.response_grid_ms),
            outdir=outdir,
            sdof=sdof,
            force_recompute=args.force_response_recompute,
        )
        response_meta = {
            "response_model": "single_mass_reference_log_pchip",
            "sdof_Tn_s": sdof.Tn_s,
            "sdof_zeta": sdof.zeta,
            "response_grid_ms": response_grid_values(args.response_grid_ms),
        }

    df = deterministic_runout_grid(
        vehicle=args.vehicle,
        speeds_kmh=parse_float_list(args.speeds),
        distances_m=parse_float_list(args.distances),
        mu_values=parse_float_list(args.mu),
        beta_values_deg=parse_float_list(args.betas),
        threshold_ms=args.threshold_ms,
        response_provider=response_provider,
    )
    full_csv = outdir / "stempi_deterministic_runout_full.csv"
    df.to_csv(full_csv, index=False)

    compact = compact_summary_table(df)
    compact_csv = outdir / "stempi_deterministic_runout_compact.csv"
    compact.to_csv(compact_csv, index=False)

    write_markdown_table(compact, outdir / "stempi_deterministic_runout_compact.md")

    # One-page table candidate: a=3 m and mu split preserved for all speeds/betas.
    one_page = df[df["a_m"] == 3.0].copy()
    columns = [
        "v0_kmh", "mu", "beta_d_deg", "v_imp_kmh", "v_n_ms", "beta_crit_deg",
        "regime",
    ]
    if "f_peak_MN" in one_page.columns:
        columns += ["f_peak_MN", "f_eq_MN"]
    one_page = one_page[columns]
    one_page.to_csv(outdir / "stempi_one_page_candidate_a3m.csv", index=False)
    write_markdown_table(one_page.round(3), outdir / "stempi_one_page_candidate_a3m.md")

    make_beta_crit_plot(outdir, args.threshold_ms)
    make_vn_grid_plot(df, outdir)

    meta = {
        "config": str(args.config),
        "vehicle": args.vehicle,
        "speeds_kmh": parse_float_list(args.speeds),
        "distances_m": parse_float_list(args.distances),
        "mu_values": parse_float_list(args.mu),
        "beta_values_deg": parse_float_list(args.betas),
        "guidance_threshold_ms": args.threshold_ms,
        **response_meta,
        "interpretation": (
            "Deterministic scenario table. No probability distribution over beta_d, mu, "
            "or v_y is assumed. beta_crit is used only to classify prescribed angles."
        ),
    }
    (outdir / "stempi_runout_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote full CSV: {full_csv}")
    print(f"Wrote compact CSV: {compact_csv}")
    print(f"Rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
