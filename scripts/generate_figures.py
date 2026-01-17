#!/usr/bin/env python3
"""
Figure Generation for Dissertation Chapter 8

Generates all required figures from parametric study results:
- fig38-anagnostopoulos-flores50-80.pdf (Section 8.1)
- fig38anagnostopoulos-flores120-200.pdf (Section 8.1)
- fig39numbercars.pdf (Section 8.2)
- fig36.pdf, fig37.pdf (Section 8.3)
- fig40stiffness.pdf (Section 8.4)
- fig33.pdf, fig34.pdf (Section 8.5)
- ComparisonContactModels2.pdf (Section 8.6)
- fig44.pdf (Section 8.7 - Force-Deformation)
- fig45.pdf, fig46.pdf, fig47.pdf (Section 8.8 - Accelerations)

Usage:
    python scripts/generate_figures.py --all
    python scripts/generate_figures.py --figure fig38
    python scripts/generate_figures.py --section 8.6

Author: Railway Impact Simulator
Date: 2026-01-10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure matplotlib for publication quality
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results" / "dissertation_ch8"
FIGURES_DIR = Path(__file__).parent.parent / "figures" / "chapter8"

# Color schemes
COLORS = {
    "anagnostopoulos": "#1f77b4",  # Blue
    "flores": "#ff7f0e",           # Orange
    "ye": "#2ca02c",               # Green
    "hertz": "#d62728",            # Red
    "hooke": "#9467bd",            # Purple
    "hunt-crossley": "#8c564b",    # Brown
    "lankarani-nikravesh": "#e377c2",  # Pink
    "gonthier": "#7f7f7f",         # Gray
    "pant-wijeyewickrema": "#bcbd22",  # Yellow-green
}

LINE_STYLES = {
    0.6: "-",
    0.7: "--",
    0.8: "-.",
    0.9: ":",
}

MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h"]


def load_case_results(section: str, case_name: str) -> Tuple[pd.DataFrame, Dict]:
    """Load results CSV and metrics JSON for a case."""
    case_dir = RESULTS_DIR / section / case_name
    df = pd.read_csv(case_dir / "results.csv")
    with open(case_dir / "metrics.json", "r") as f:
        metrics = json.load(f)
    return df, metrics


def load_section_results(section: str) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
    """Load all results for a section."""
    section_dir = RESULTS_DIR / section
    results = {}
    if section_dir.exists():
        for case_dir in section_dir.iterdir():
            if case_dir.is_dir() and (case_dir / "results.csv").exists():
                results[case_dir.name] = load_case_results(section, case_dir.name)
    return results


# =============================================================================
# Section 8.1: Speed & Restitution Coefficient
# =============================================================================

def generate_fig38_speed_restitution():
    """
    Generate figures for Section 8.1: Speed & Restitution

    Creates two PDFs:
    - fig38-anagnostopoulos-flores50-80.pdf (50, 80 km/h)
    - fig38anagnostopoulos-flores120-200.pdf (120, 200 km/h)
    """
    results = load_section_results("8.1_speed_restitution")
    if not results:
        print("No results found for section 8.1")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Group by speed range
    speed_groups = {
        "50-80": [50, 80],
        "120-200": [120, 200],
    }

    for group_name, speeds in speed_groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Influence of Speed and Restitution Coefficient ({group_name} km/h)")

        for idx, speed in enumerate(speeds):
            ax_anag = axes[idx, 0]
            ax_flores = axes[idx, 1]

            ax_anag.set_title(f"Anagnostopoulos - {speed} km/h")
            ax_flores.set_title(f"Flores - {speed} km/h")

            for cr in [0.6, 0.7, 0.8, 0.9]:
                # Anagnostopoulos
                case_anag = f"v{speed}_cr{int(cr*10)}_anagnostopoulos"
                if case_anag in results:
                    df, _ = results[case_anag]
                    ax_anag.plot(
                        df["Time_ms"], df["Impact_Force_MN"],
                        label=f"cr = {cr}",
                        linestyle=LINE_STYLES[cr],
                        color=COLORS["anagnostopoulos"],
                    )

                # Flores
                case_flores = f"v{speed}_cr{int(cr*10)}_flores"
                if case_flores in results:
                    df, _ = results[case_flores]
                    ax_flores.plot(
                        df["Time_ms"], df["Impact_Force_MN"],
                        label=f"cr = {cr}",
                        linestyle=LINE_STYLES[cr],
                        color=COLORS["flores"],
                    )

            for ax in [ax_anag, ax_flores]:
                ax.set_xlabel("Time [ms]")
                ax.set_ylabel("Impact Force [MN]")
                ax.legend(loc="upper right")
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

        plt.tight_layout()
        pdf_path = FIGURES_DIR / f"fig38-anagnostopoulos-flores{group_name}.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Generated: {pdf_path}")


# =============================================================================
# Section 8.2: Number of Cars
# =============================================================================

def generate_fig39_number_cars():
    """Generate figure for Section 8.2: Number of Cars"""
    results = load_section_results("8.2_number_cars")
    if not results:
        print("No results found for section 8.2")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Force comparison
    ax1 = axes[0]
    ax1.set_title("Impact Force Comparison")

    for case_name, (df, metrics) in results.items():
        label = "Single Wagon (40t)" if "single" in case_name else "Full ICE 1 (14 cars, 773t)"
        color = "#1f77b4" if "single" in case_name else "#ff7f0e"
        ax1.plot(df["Time_ms"], df["Impact_Force_MN"], label=label, color=color)

    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Impact Force [MN]")
    ax1.legend()
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # Peak force bar chart
    ax2 = axes[1]
    ax2.set_title("Peak Force Comparison")

    names = []
    peaks = []
    for case_name, (_, metrics) in results.items():
        names.append("Single\nWagon" if "single" in case_name else "Full\nICE 1")
        peaks.append(metrics.get("peak_force_MN", 0))

    bars = ax2.bar(names, peaks, color=["#1f77b4", "#ff7f0e"])
    ax2.set_ylabel("Peak Force [MN]")

    # Add value labels
    for bar, peak in zip(bars, peaks):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{peak:.1f}", ha="center", va="bottom")

    plt.tight_layout()
    pdf_path = FIGURES_DIR / "fig39numbercars.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Generated: {pdf_path}")


# =============================================================================
# Section 8.3: Friction
# =============================================================================

def generate_fig36_37_friction():
    """
    Generate figures for Section 8.3: Friction

    - fig36.pdf: Flores model
    - fig37.pdf: Anagnostopoulos model
    """
    results = load_section_results("8.3_friction")
    if not results:
        print("No results found for section 8.3")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mu_combinations = [(0.3, 0.2), (0.5, 0.4), (0.7, 0.6), (1.0, 0.9)]
    distances = [3, 6, 9, 12]

    for model, fig_name in [("flores", "fig36"), ("anagnostopoulos", "fig37")]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Friction Influence - {model.capitalize()} Model")

        for idx, (mu_s, mu_k) in enumerate(mu_combinations):
            ax = axes[idx // 2, idx % 2]
            ax.set_title(f"$\\mu_s$ = {mu_s}, $\\mu_k$ = {mu_k}")

            for d_idx, dist in enumerate(distances):
                case_name = f"mu{int(mu_s*10)}{int(mu_k*10)}_d{dist}m_{model}"
                if case_name in results:
                    df, _ = results[case_name]
                    ax.plot(
                        df["Time_ms"], df["Impact_Force_MN"],
                        label=f"d = {dist} m",
                        marker=MARKERS[d_idx],
                        markevery=50,
                        markersize=4,
                    )

            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Impact Force [MN]")
            ax.legend(loc="upper right")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        pdf_path = FIGURES_DIR / f"{fig_name}.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Generated: {pdf_path}")


# =============================================================================
# Section 8.4: Structure Stiffness
# =============================================================================

def generate_fig40_stiffness():
    """Generate figure for Section 8.4: Structure Stiffness"""
    results = load_section_results("8.4_stiffness")
    if not results:
        print("No results found for section 8.4")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    k_factors = [0.8, 1.0, 2.2]
    cr_values = [0.6, 0.7, 0.8, 0.9]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Influence of Structure Stiffness")

    for k_idx, k_factor in enumerate(k_factors):
        # Anagnostopoulos
        ax_anag = axes[0, k_idx]
        ax_anag.set_title(f"Anagnostopoulos - K = {k_factor}x K_eff")

        # Flores
        ax_flores = axes[1, k_idx]
        ax_flores.set_title(f"Flores - K = {k_factor}x K_eff")

        for cr in cr_values:
            for model, ax in [("anagnostopoulos", ax_anag), ("flores", ax_flores)]:
                case_name = f"k{int(k_factor*10)}_cr{int(cr*10)}_{model}"
                if case_name in results:
                    df, _ = results[case_name]
                    ax.plot(
                        df["Time_ms"], df["Impact_Force_MN"],
                        label=f"cr = {cr}",
                        linestyle=LINE_STYLES[cr],
                    )

        for ax in [ax_anag, ax_flores]:
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Impact Force [MN]")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    pdf_path = FIGURES_DIR / "fig40stiffness.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Generated: {pdf_path}")


# =============================================================================
# Section 8.5: Train Materials
# =============================================================================

def generate_fig33_34_materials():
    """
    Generate figures for Section 8.5: Train Materials

    - fig33.pdf: Quasi-static force-deformation curves
    - fig34.pdf: Dynamic comparison
    """
    results = load_section_results("8.5_materials")
    if not results:
        print("No results found for section 8.5")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # fig33: Quasi-static curves (analytical)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Quasi-Static Force-Deformation Curves")

    # Aluminium: fy=15 MN, uy=200 mm
    u_al = np.linspace(0, 500, 100)  # mm
    f_al = np.minimum(15 * u_al / 200, 15)  # MN
    ax.plot(u_al, f_al, label="Aluminium (ICE 1)", color="#1f77b4", linewidth=2)

    # Steel S355: fy=18 MN, uy=40 mm
    u_st = np.linspace(0, 500, 100)  # mm
    f_st = np.minimum(18 * u_st / 40, 18)  # MN
    ax.plot(u_st, f_st, label="Steel S355", color="#ff7f0e", linewidth=2)

    ax.set_xlabel("Deformation [mm]")
    ax.set_ylabel("Force [MN]")
    ax.legend()
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 25)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig33.pdf")
    plt.close(fig)
    print(f"Generated: {FIGURES_DIR / 'fig33.pdf'}")

    # fig34: Dynamic comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.set_title("Impact Force vs Time")

    for case_name, (df, metrics) in results.items():
        label = "Aluminium" if "aluminium" in case_name else "Steel S355"
        color = "#1f77b4" if "aluminium" in case_name else "#ff7f0e"
        ax1.plot(df["Time_ms"], df["Impact_Force_MN"], label=label, color=color)

    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Impact Force [MN]")
    ax1.legend()
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.set_title("Force-Penetration Hysteresis")

    for case_name, (df, metrics) in results.items():
        label = "Aluminium" if "aluminium" in case_name else "Steel S355"
        color = "#1f77b4" if "aluminium" in case_name else "#ff7f0e"
        ax2.plot(df["Penetration_mm"], df["Impact_Force_MN"], label=label, color=color)

    ax2.set_xlabel("Penetration [mm]")
    ax2.set_ylabel("Impact Force [MN]")
    ax2.legend()
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig34.pdf")
    plt.close(fig)
    print(f"Generated: {FIGURES_DIR / 'fig34.pdf'}")


# =============================================================================
# Section 8.6: Contact Models Comparison
# =============================================================================

def generate_comparison_contact_models():
    """Generate figure for Section 8.6: Contact Models Comparison"""
    results = load_section_results("8.6_contact_models")
    if not results:
        print("No results found for section 8.6")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Main comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Force vs Time
    ax1 = axes[0, 0]
    ax1.set_title("Impact Force vs Time")

    # Force vs Penetration
    ax2 = axes[0, 1]
    ax2.set_title("Force-Penetration Hysteresis")

    # Peak force bar chart
    ax3 = axes[1, 0]
    ax3.set_title("Peak Force Comparison")

    # Energy dissipation
    ax4 = axes[1, 1]
    ax4.set_title("Contact Duration")

    model_names = []
    peak_forces = []
    durations = []

    for case_name, (df, metrics) in sorted(results.items()):
        model = metrics.get("contact_model", case_name)
        color = COLORS.get(model, "#333333")

        ax1.plot(df["Time_ms"], df["Impact_Force_MN"], label=model, color=color)
        ax2.plot(df["Penetration_mm"], df["Impact_Force_MN"], label=model, color=color)

        model_names.append(model)
        peak_forces.append(metrics.get("peak_force_MN", 0))
        durations.append(metrics.get("contact_duration_s", 0) * 1000)  # to ms

    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Impact Force [MN]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax2.set_xlabel("Penetration [mm]")
    ax2.set_ylabel("Impact Force [MN]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    # Bar charts
    x = np.arange(len(model_names))
    colors = [COLORS.get(m, "#333333") for m in model_names]

    bars1 = ax3.bar(x, peak_forces, color=colors)
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Peak Force [MN]")

    bars2 = ax4.bar(x, durations, color=colors)
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Contact Duration [ms]")

    plt.tight_layout()
    pdf_path = FIGURES_DIR / "ComparisonContactModels2.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Generated: {pdf_path}")


# =============================================================================
# Section 8.7 & 8.8: Deformation and Acceleration Analysis
# =============================================================================

def generate_fig44_deformation():
    """Generate figure for Section 8.7: Force-Deformation curves"""
    results = load_section_results("8.6_contact_models")
    if not results:
        print("No results found for section 8.6 (needed for fig44)")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    target_models = ["anagnostopoulos", "flores", "ye"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Force-Deformation Curves")

    for idx, model in enumerate(target_models):
        ax = axes[idx]
        case_name = f"model_{model.replace('-', '_')}"

        if case_name in results:
            df, _ = results[case_name]
            ax.plot(df["Penetration_mm"], df["Impact_Force_MN"],
                    color=COLORS.get(model, "#333333"), linewidth=2)
            ax.set_title(model.capitalize())
            ax.set_xlabel("Penetration [mm]")
            ax.set_ylabel("Impact Force [MN]")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    pdf_path = FIGURES_DIR / "fig44.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Generated: {pdf_path}")


def generate_fig45_47_accelerations():
    """
    Generate figures for Section 8.8: Acceleration Analysis

    - fig45.pdf: FFT analysis
    - fig46.pdf, fig47.pdf: Nodal accelerations
    """
    results = load_section_results("8.6_contact_models")
    if not results:
        print("No results found for section 8.6 (needed for acceleration figures)")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    target_models = ["anagnostopoulos", "flores", "ye"]

    # fig46, fig47: Nodal accelerations
    for fig_num, model in zip([46, 47], ["anagnostopoulos", "flores"]):
        case_name = f"model_{model.replace('-', '_')}"

        if case_name not in results:
            continue

        df, _ = results[case_name]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Nodal Accelerations - {model.capitalize()}")

        # Plot front mass acceleration
        ax.plot(df["Time_ms"], df["Acceleration_g"], label="Front mass", linewidth=2)

        # Try to find additional mass accelerations
        for col in df.columns:
            if col.startswith("Acceleration_mass_") and col.endswith("_g"):
                mass_num = col.split("_")[2]
                ax.plot(df["Time_ms"], df[col], label=f"Mass {mass_num}", alpha=0.7)

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Acceleration [g]")
        ax.legend(loc="upper right")
        ax.set_xlim(left=0)

        plt.tight_layout()
        pdf_path = FIGURES_DIR / f"fig{fig_num}.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Generated: {pdf_path}")

    # fig45: FFT analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("FFT of Acceleration Signals")

    for idx, model in enumerate(target_models):
        ax = axes[idx]
        case_name = f"model_{model.replace('-', '_')}"

        if case_name not in results:
            continue

        df, _ = results[case_name]

        # Compute FFT
        dt = (df["Time_s"].iloc[1] - df["Time_s"].iloc[0])
        n = len(df)
        acc = df["Acceleration_g"].values

        freq = np.fft.rfftfreq(n, dt)
        fft_mag = np.abs(np.fft.rfft(acc)) / n

        ax.semilogy(freq, fft_mag, color=COLORS.get(model, "#333333"))
        ax.set_title(model.capitalize())
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(0, 500)

    plt.tight_layout()
    pdf_path = FIGURES_DIR / "fig45.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Generated: {pdf_path}")


# =============================================================================
# Summary Figure
# =============================================================================

def generate_summary_figure():
    """Generate a summary figure with key results from all sections."""
    summary_path = RESULTS_DIR / "summary.csv"
    if not summary_path.exists():
        print("No summary.csv found")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # Peak force distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Peak Force Distribution")
    if "peak_force_MN" in summary.columns:
        ax1.hist(summary["peak_force_MN"].dropna(), bins=20, edgecolor="black")
        ax1.set_xlabel("Peak Force [MN]")
        ax1.set_ylabel("Count")

    # DAF distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Dynamic Amplification Factor (DAF)")
    if "DAF" in summary.columns:
        ax2.hist(summary["DAF"].dropna(), bins=20, edgecolor="black")
        ax2.set_xlabel("DAF")
        ax2.set_ylabel("Count")
        ax2.axvline(x=3.0, color="red", linestyle="--", label="Limit = 3.0")
        ax2.legend()

    # Energy balance
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Energy Balance Error")
    if "energy_balance_error" in summary.columns:
        ax3.hist(summary["energy_balance_error"].dropna() * 100, bins=20, edgecolor="black")
        ax3.set_xlabel("Energy Balance Error [%]")
        ax3.set_ylabel("Count")

    # Wall time distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Simulation Wall Time")
    if "wall_time_s" in summary.columns:
        ax4.hist(summary["wall_time_s"].dropna(), bins=20, edgecolor="black")
        ax4.set_xlabel("Wall Time [s]")
        ax4.set_ylabel("Count")

    # Plausibility check results
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title("Results by Section")

    if "section" in summary.columns:
        section_stats = summary.groupby("section").agg({
            "peak_force_MN": ["mean", "std", "count"],
        }).round(2)

        sections = section_stats.index.tolist()
        means = section_stats[("peak_force_MN", "mean")].values
        stds = section_stats[("peak_force_MN", "std")].values
        counts = section_stats[("peak_force_MN", "count")].values

        x = np.arange(len(sections))
        bars = ax5.bar(x, means, yerr=stds, capsize=5)
        ax5.set_xticks(x)
        ax5.set_xticklabels([s.replace("_", "\n") for s in sections], fontsize=8)
        ax5.set_ylabel("Mean Peak Force [MN]")

        # Add count labels
        for bar, count in zip(bars, counts):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"n={int(count)}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    pdf_path = FIGURES_DIR / "summary_all_results.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Generated: {pdf_path}")


# =============================================================================
# Main
# =============================================================================

FIGURE_GENERATORS = {
    "fig38": generate_fig38_speed_restitution,
    "fig39": generate_fig39_number_cars,
    "fig36": generate_fig36_37_friction,
    "fig37": generate_fig36_37_friction,  # Same function
    "fig40": generate_fig40_stiffness,
    "fig33": generate_fig33_34_materials,
    "fig34": generate_fig33_34_materials,  # Same function
    "contact_models": generate_comparison_contact_models,
    "fig44": generate_fig44_deformation,
    "fig45": generate_fig45_47_accelerations,
    "fig46": generate_fig45_47_accelerations,
    "fig47": generate_fig45_47_accelerations,
    "summary": generate_summary_figure,
}

SECTION_FIGURES = {
    "8.1": ["fig38"],
    "8.2": ["fig39"],
    "8.3": ["fig36", "fig37"],
    "8.4": ["fig40"],
    "8.5": ["fig33", "fig34"],
    "8.6": ["contact_models"],
    "8.7": ["fig44"],
    "8.8": ["fig45", "fig46", "fig47"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate dissertation figures from parametric study results"
    )
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--figure", "-f", nargs="+", help="Specific figures to generate")
    parser.add_argument("--section", "-s", nargs="+", help="Generate figures for sections")
    parser.add_argument("--list", action="store_true", help="List available figures")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable figures:")
        for fig, gen in FIGURE_GENERATORS.items():
            print(f"  {fig}: {gen.__doc__.split(chr(10))[0] if gen.__doc__ else 'No description'}")
        print("\nSections and their figures:")
        for sec, figs in SECTION_FIGURES.items():
            print(f"  {sec}: {', '.join(figs)}")
        return

    if args.all:
        # Run each generator only once
        seen = set()
        for gen in FIGURE_GENERATORS.values():
            if gen not in seen:
                gen()
                seen.add(gen)
        return

    if args.section:
        for sec in args.section:
            if sec in SECTION_FIGURES:
                for fig in SECTION_FIGURES[sec]:
                    if fig in FIGURE_GENERATORS:
                        FIGURE_GENERATORS[fig]()
            else:
                print(f"Unknown section: {sec}")
        return

    if args.figure:
        for fig in args.figure:
            if fig in FIGURE_GENERATORS:
                FIGURE_GENERATORS[fig]()
            else:
                print(f"Unknown figure: {fig}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
