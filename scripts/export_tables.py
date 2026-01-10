#!/usr/bin/env python3
"""
LaTeX Table Export for Dissertation Chapter 8

Generates LaTeX tables from parametric study results with proper escaping.

Usage:
    python scripts/export_tables.py --all
    python scripts/export_tables.py --section 8.1
    python scripts/export_tables.py --summary

Author: Railway Impact Simulator
Date: 2026-01-10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results" / "dissertation_ch8"
TABLES_DIR = Path(__file__).parent.parent / "tables" / "chapter8"


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        text = str(text)

    # Characters that need escaping
    special_chars = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }

    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)

    return text


def format_number(value: float, decimals: int = 2, scientific: bool = False) -> str:
    """Format number for LaTeX table."""
    if pd.isna(value):
        return "---"

    if scientific and (abs(value) > 1e4 or (abs(value) < 1e-2 and value != 0)):
        mantissa = value / (10 ** int(f"{value:.0e}".split("e")[1]))
        exp = int(f"{value:.0e}".split("e")[1])
        return f"${mantissa:.{decimals}f} \\times 10^{{{exp}}}$"

    return f"{value:.{decimals}f}"


def df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: Optional[str] = None,
    header_map: Optional[Dict[str, str]] = None,
    escape_content: bool = True,
    float_format: str = "%.2f",
    position: str = "htbp",
) -> str:
    """
    Convert DataFrame to LaTeX table with proper formatting.

    Parameters
    ----------
    df : pd.DataFrame
        Data to convert
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    column_format : str, optional
        Column alignment specification (e.g., "l|ccc|r")
    header_map : dict, optional
        Mapping of column names to display names
    escape_content : bool
        Whether to escape special characters in content
    float_format : str
        Format string for floating point numbers
    position : str
        Table position specifier

    Returns
    -------
    str
        LaTeX table code
    """
    # Prepare dataframe
    df_out = df.copy()

    # Apply header mapping
    if header_map:
        df_out = df_out.rename(columns=header_map)

    # Escape content if needed
    if escape_content:
        for col in df_out.columns:
            if df_out[col].dtype == object:
                df_out[col] = df_out[col].apply(
                    lambda x: escape_latex(str(x)) if pd.notna(x) else "---"
                )

    # Generate column format if not provided
    if column_format is None:
        n_cols = len(df_out.columns)
        column_format = "l" + "c" * (n_cols - 1)

    # Build LaTeX
    lines = [
        f"\\begin{{table}}[{position}]",
        "\\centering",
        f"\\caption{{{escape_latex(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
    ]

    # Header
    headers = " & ".join(escape_latex(str(col)) for col in df_out.columns)
    lines.append(f"{headers} \\\\")
    lines.append("\\midrule")

    # Data rows
    for _, row in df_out.iterrows():
        values = []
        for val in row:
            if isinstance(val, float):
                values.append(float_format % val if pd.notna(val) else "---")
            else:
                values.append(str(val))
        lines.append(" & ".join(values) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def load_section_metrics(section: str) -> pd.DataFrame:
    """Load all metrics for a section into a DataFrame."""
    section_dir = RESULTS_DIR / section
    records = []

    if section_dir.exists():
        for case_dir in section_dir.iterdir():
            metrics_path = case_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    metrics["case_name"] = case_dir.name
                    records.append(metrics)

    return pd.DataFrame(records)


# =============================================================================
# Section-specific tables
# =============================================================================

def export_table_8_1():
    """Export table for Section 8.1: Speed & Restitution."""
    df = load_section_metrics("8.1_speed_restitution")
    if df.empty:
        print("No data for section 8.1")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Create pivot table: rows = speed x cr, columns = model
    df["speed_cr"] = df.apply(
        lambda r: f"{int(r['speed_kmh'])} / {r['cr_wall']:.1f}", axis=1
    )

    # Peak force comparison
    pivot = df.pivot_table(
        values="peak_force_MN",
        index=["speed_kmh", "cr_wall"],
        columns="contact_model",
        aggfunc="first",
    ).round(2)

    pivot = pivot.reset_index()
    pivot.columns.name = None

    latex = df_to_latex(
        pivot,
        caption="Peak impact force [MN] for different speeds and coefficients of restitution",
        label="tab:speed_restitution_peak_force",
        header_map={
            "speed_kmh": "$v$ [km/h]",
            "cr_wall": "$c_r$ [-]",
            "anagnostopoulos": "Anagnostopoulos",
            "flores": "Flores",
        },
    )

    output_path = TABLES_DIR / "table_8_1_speed_restitution.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Generated: {output_path}")


def export_table_8_2():
    """Export table for Section 8.2: Number of Cars."""
    df = load_section_metrics("8.2_number_cars")
    if df.empty:
        print("No data for section 8.2")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Select relevant columns
    cols = ["config", "n_cars", "total_mass_t", "peak_force_MN", "contact_duration_s", "DAF"]
    df_out = df[cols].copy()

    latex = df_to_latex(
        df_out,
        caption="Comparison of single wagon vs full train",
        label="tab:number_cars",
        header_map={
            "config": "Configuration",
            "n_cars": "$n$ cars",
            "total_mass_t": "Mass [t]",
            "peak_force_MN": "$F_{max}$ [MN]",
            "contact_duration_s": "$t_{contact}$ [s]",
            "DAF": "DAF [-]",
        },
    )

    output_path = TABLES_DIR / "table_8_2_number_cars.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Generated: {output_path}")


def export_table_8_3():
    """Export table for Section 8.3: Friction."""
    df = load_section_metrics("8.3_friction")
    if df.empty:
        print("No data for section 8.3")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Create summary by friction coefficient
    summary = df.groupby(["mu_s", "mu_k", "contact_model"]).agg({
        "peak_force_MN": "mean",
        "contact_duration_s": "mean",
    }).round(3).reset_index()

    latex = df_to_latex(
        summary,
        caption="Average peak force and contact duration for different friction coefficients",
        label="tab:friction_summary",
        header_map={
            "mu_s": "$\\mu_s$ [-]",
            "mu_k": "$\\mu_k$ [-]",
            "contact_model": "Model",
            "peak_force_MN": "$\\bar{F}_{max}$ [MN]",
            "contact_duration_s": "$\\bar{t}_{contact}$ [s]",
        },
    )

    output_path = TABLES_DIR / "table_8_3_friction.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Generated: {output_path}")


def export_table_8_4():
    """Export table for Section 8.4: Structure Stiffness."""
    df = load_section_metrics("8.4_stiffness")
    if df.empty:
        print("No data for section 8.4")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Pivot by k_factor and cr
    pivot = df.pivot_table(
        values="peak_force_MN",
        index=["k_factor", "contact_model"],
        columns="cr_wall",
        aggfunc="first",
    ).round(2).reset_index()

    pivot.columns.name = None

    # Rename cr columns
    col_rename = {c: f"$c_r$ = {c}" if isinstance(c, float) else c for c in pivot.columns}
    pivot = pivot.rename(columns=col_rename)

    latex = df_to_latex(
        pivot,
        caption="Peak force [MN] for different structure stiffness and restitution coefficients",
        label="tab:stiffness",
        header_map={
            "k_factor": "$K/K_{eff}$ [-]",
            "contact_model": "Model",
        },
    )

    output_path = TABLES_DIR / "table_8_4_stiffness.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Generated: {output_path}")


def export_table_8_5():
    """Export table for Section 8.5: Train Materials."""
    df = load_section_metrics("8.5_materials")
    if df.empty:
        print("No data for section 8.5")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["material", "fy_MN", "uy_mm", "peak_force_MN", "peak_penetration_mm", "DAF"]
    df_out = df[cols].copy()

    latex = df_to_latex(
        df_out,
        caption="Comparison of train materials",
        label="tab:materials",
        header_map={
            "material": "Material",
            "fy_MN": "$f_y$ [MN]",
            "uy_mm": "$u_y$ [mm]",
            "peak_force_MN": "$F_{max}$ [MN]",
            "peak_penetration_mm": "$\\delta_{max}$ [mm]",
            "DAF": "DAF [-]",
        },
    )

    output_path = TABLES_DIR / "table_8_5_materials.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Generated: {output_path}")


def export_table_8_6():
    """Export table for Section 8.6: Contact Models."""
    df = load_section_metrics("8.6_contact_models")
    if df.empty:
        print("No data for section 8.6")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["contact_model", "peak_force_MN", "time_of_peak_s", "contact_duration_s", "DAF"]
    df_out = df[cols].copy().sort_values("contact_model")

    latex = df_to_latex(
        df_out,
        caption="Comparison of contact models at 80 km/h, $c_r$ = 0.8",
        label="tab:contact_models",
        header_map={
            "contact_model": "Contact Model",
            "peak_force_MN": "$F_{max}$ [MN]",
            "time_of_peak_s": "$t_{peak}$ [s]",
            "contact_duration_s": "$t_{contact}$ [s]",
            "DAF": "DAF [-]",
        },
        float_format="%.3f",
    )

    output_path = TABLES_DIR / "table_8_6_contact_models.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Generated: {output_path}")


def export_summary_table():
    """Export overall summary table."""
    summary_path = RESULTS_DIR / "summary.csv"
    if not summary_path.exists():
        print("No summary.csv found")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)

    # Group by section
    summary = df.groupby("section").agg({
        "peak_force_MN": ["count", "mean", "std", "min", "max"],
        "DAF": ["mean", "max"],
        "energy_balance_error": "max",
        "wall_time_s": "sum",
    }).round(3)

    # Flatten columns
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary = summary.reset_index()

    latex = df_to_latex(
        summary,
        caption="Summary of parametric study results by section",
        label="tab:summary",
        header_map={
            "section": "Section",
            "peak_force_MN_count": "$n$",
            "peak_force_MN_mean": "$\\bar{F}_{max}$",
            "peak_force_MN_std": "$\\sigma_F$",
            "peak_force_MN_min": "$F_{min}$",
            "peak_force_MN_max": "$F_{max}$",
            "DAF_mean": "$\\bar{DAF}$",
            "DAF_max": "$DAF_{max}$",
            "energy_balance_error_max": "$\\epsilon_{max}$",
            "wall_time_s_sum": "$t_{total}$ [s]",
        },
    )

    output_path = TABLES_DIR / "table_summary.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Generated: {output_path}")


def export_plausibility_table():
    """Export plausibility check results."""
    summary_path = RESULTS_DIR / "summary.csv"
    if not summary_path.exists():
        print("No summary.csv found")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)

    # Count plausibility results
    if "plausibility_passed" in df.columns:
        plaus = df.groupby("section").agg({
            "plausibility_passed": ["sum", "count"],
        })
        plaus.columns = ["passed", "total"]
        plaus["failed"] = plaus["total"] - plaus["passed"]
        plaus["pass_rate"] = (plaus["passed"] / plaus["total"] * 100).round(1)
        plaus = plaus.reset_index()

        latex = df_to_latex(
            plaus,
            caption="Physical plausibility check results",
            label="tab:plausibility",
            header_map={
                "section": "Section",
                "passed": "Passed",
                "failed": "Failed",
                "total": "Total",
                "pass_rate": "Pass Rate [\\%]",
            },
        )

        output_path = TABLES_DIR / "table_plausibility.tex"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"Generated: {output_path}")


# =============================================================================
# Main
# =============================================================================

TABLE_GENERATORS = {
    "8.1": export_table_8_1,
    "8.2": export_table_8_2,
    "8.3": export_table_8_3,
    "8.4": export_table_8_4,
    "8.5": export_table_8_5,
    "8.6": export_table_8_6,
    "summary": export_summary_table,
    "plausibility": export_plausibility_table,
}


def main():
    parser = argparse.ArgumentParser(
        description="Export dissertation tables to LaTeX"
    )
    parser.add_argument("--all", action="store_true", help="Export all tables")
    parser.add_argument("--section", "-s", nargs="+", help="Sections to export")
    parser.add_argument("--summary", action="store_true", help="Export summary tables")
    parser.add_argument("--list", action="store_true", help="List available tables")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable tables:")
        for name in TABLE_GENERATORS:
            print(f"  {name}")
        return

    if args.all:
        for gen in TABLE_GENERATORS.values():
            gen()
        return

    if args.summary:
        export_summary_table()
        export_plausibility_table()
        return

    if args.section:
        for sec in args.section:
            if sec in TABLE_GENERATORS:
                TABLE_GENERATORS[sec]()
            else:
                print(f"Unknown section: {sec}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
