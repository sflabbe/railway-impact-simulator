"""railway_simulator.ui.export

Export utilities for the Railway Impact Simulator UI.

The Streamlit app relies on lightweight, in-memory exports that work without
optional heavy dependencies (e.g. Plotly kaleido). Therefore:

* Tabular exports: CSV / XLSX
* Plot exports: Plotly -> HTML (standalone), Matplotlib -> PNG/SVG (handled elsewhere)
* Bundles: ZIP files containing a consistent folder layout
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from typing import Any, Mapping

import pandas as pd


_FNAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9._\-+=@()\[\]{} ]+")


def sanitize_filename(name: str) -> str:
    """Best-effort filename sanitizer (keeps it readable)."""
    s = str(name).strip().replace("/", "-").replace("\\", "-")
    s = _FNAME_SAFE_RE.sub("_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:180] if len(s) > 180 else s


def utc_timestamp() -> str:
    """UTC timestamp suitable for filenames."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def to_excel(df: pd.DataFrame, *, sheet_name: str = "Sheet1") -> bytes:
    """Generate an Excel file for download."""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    except ImportError:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


def to_json_bytes(obj: Any, *, indent: int = 2) -> bytes:
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=str).encode("utf-8")


def make_zip_bytes(files: Mapping[str, bytes | str]) -> bytes:
    """Create a ZIP archive in memory.

    Parameters
    ----------
    files:
        Mapping from archive path -> bytes or string content.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, content in files.items():
            arc = sanitize_filename(path) if "/" not in path else "/".join(
                sanitize_filename(p) for p in path.split("/") if p
            )
            if isinstance(content, str):
                content_b = content.encode("utf-8")
            else:
                content_b = content
            zf.writestr(arc, content_b)
    return buf.getvalue()


def make_bundle_zip(
    *,
    study: str,
    metadata: dict[str, Any],
    dataframes: Mapping[str, pd.DataFrame] | None = None,
    plots_html: Mapping[str, bytes] | None = None,
    runs: Mapping[str, pd.DataFrame] | None = None,
    extra_files: Mapping[str, bytes | str] | None = None,
) -> bytes:
    """Create a standard parametric-study ZIP bundle.

    Layout:
      - meta.json
      - data/<name>.csv
      - data/<name>.xlsx
      - plots/<name>.html
      - runs/<scenario>.csv
      - extra/*
    """
    files: dict[str, bytes | str] = {}

    meta = dict(metadata or {})
    meta.setdefault("study", study)
    meta.setdefault("created_utc", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    files["meta.json"] = to_json_bytes(meta)

    if dataframes:
        for name, df in dataframes.items():
            if df is None:
                continue
            files[f"data/{name}.csv"] = df.to_csv(index=False)
            # Excel is convenient for reviewers
            try:
                files[f"data/{name}.xlsx"] = to_excel(df, sheet_name=sanitize_filename(name)[:30] or "Sheet1")
            except Exception:
                # If Excel writer is unavailable, CSV is enough.
                pass

    if plots_html:
        for name, html in plots_html.items():
            if html is None:
                continue
            files[f"plots/{name}.html"] = html

    if runs:
        for scen, df in runs.items():
            if df is None:
                continue
            files[f"runs/{scen}.csv"] = df.to_csv(index=False)

    if extra_files:
        for name, content in extra_files.items():
            files[f"extra/{name}"] = content

    return make_zip_bytes(files)


# -----------------------------------------------------------------------------
# Optional: Matplotlib replot helper (for paper-like figures)
# -----------------------------------------------------------------------------

_REPLOT_MATPLOTLIB_PY = r'''#!/usr/bin/env python3
"""replot_matplotlib.py

Replot an exported parametric-study bundle (ZIP layout produced by the Streamlit UI)
using Matplotlib, to obtain publication-friendly PNG/SVG outputs that are
independent of Plotly/Streamlit.

Usage
-----

1) Unzip a bundle:
   unzip speed_envelope__Impact_Force_MN__...zip -d bundle

2) Run:
   python replot_matplotlib.py --bundle bundle --outdir plots_mpl --dpi 300 --svg

This script is intentionally self-contained (only needs numpy/pandas/matplotlib).

What it generates
-----------------
- Overlay time histories (Impact_Force_MN / Acceleration_g / Penetration_mm)
- If envelope CSVs exist: envelope + weighted mean
- If force-penetration information exists: force–penetration scatter colored by time

Notes
-----
- Units are inferred from column names used in the simulator.
- If Time_ms is missing, Time_s is converted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _maybe_unzip(bundle: Path) -> Path:
    if bundle.is_dir():
        return bundle
    if bundle.suffix.lower() != ".zip":
        raise SystemExit(f"Bundle must be a directory or .zip file: {bundle}")

    tmp = Path(tempfile.mkdtemp(prefix="ris_bundle_"))
    with zipfile.ZipFile(bundle, "r") as zf:
        zf.extractall(tmp)

    # If the zip has a single top-level folder, use it.
    children = [p for p in tmp.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return tmp


def _load_meta(root: Path) -> dict:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _time_ms(df: pd.DataFrame) -> np.ndarray:
    if "Time_ms" in df.columns:
        return df["Time_ms"].to_numpy(dtype=float)
    if "Time_s" in df.columns:
        return df["Time_s"].to_numpy(dtype=float) * 1000.0
    return np.arange(len(df), dtype=float)


def _read_runs(root: Path) -> list[tuple[str, pd.DataFrame]]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []
    runs = []
    for f in sorted(runs_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f)
            runs.append((f.stem, df))
        except Exception:
            continue
    return runs


def _plot_overlay(
    runs: list[tuple[str, pd.DataFrame]],
    outdir: Path,
    quantity: str,
    *,
    title: str,
    ylabel: str,
    dpi: int,
    svg: bool,
) -> None:
    import matplotlib.pyplot as plt

    _ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    plotted = 0
    for name, df in runs:
        if quantity not in df.columns:
            continue
        t = _time_ms(df)
        ax.plot(t, df[quantity].to_numpy(dtype=float), label=name)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, frameon=False)

    fig.tight_layout()

    png_path = outdir / f"overlay__{quantity}.png"
    fig.savefig(png_path, dpi=dpi)
    if svg:
        fig.savefig(outdir / f"overlay__{quantity}.svg")

    plt.close(fig)


def _plot_envelope(root: Path, outdir: Path, quantity: str, *, dpi: int, svg: bool) -> None:
    import matplotlib.pyplot as plt

    candidates = [
        root / "data" / f"envelope_force.csv",
        root / "data" / f"envelope_acc.csv",
        root / "data" / f"envelope_pen.csv",
        root / "data" / f"envelope_{quantity}.csv",
    ]

    env_path = None
    for c in candidates:
        if c.exists():
            env_path = c
            break

    if env_path is None:
        return

    df = pd.read_csv(env_path)

    t = df["Time_ms"].to_numpy(dtype=float) if "Time_ms" in df.columns else df["Time_s"].to_numpy(dtype=float) * 1000.0

    env_col = f"{quantity}_envelope" if f"{quantity}_envelope" in df.columns else quantity
    mean_col = f"{quantity}_weighted_mean" if f"{quantity}_weighted_mean" in df.columns else None

    if env_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(t, df[env_col].to_numpy(dtype=float), label="Envelope")
    if mean_col and mean_col in df.columns:
        ax.plot(t, df[mean_col].to_numpy(dtype=float), label="Weighted mean", linestyle="--")

    ax.set_title(f"Envelope – {quantity}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(quantity)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()

    _ensure_dir(outdir)
    fig.savefig(outdir / f"envelope__{quantity}.png", dpi=dpi)
    if svg:
        fig.savefig(outdir / f"envelope__{quantity}.svg")
    plt.close(fig)


def _plot_force_penetration(runs: list[tuple[str, pd.DataFrame]], outdir: Path, *, dpi: int, svg: bool) -> None:
    import matplotlib.pyplot as plt

    for name, df in runs:
        if not ("Penetration_mm" in df.columns and "Impact_Force_MN" in df.columns):
            continue

        t = _time_ms(df)
        x = df["Penetration_mm"].to_numpy(dtype=float)
        y = df["Impact_Force_MN"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        sc = ax.scatter(x, y, c=t, s=8)
        ax.set_title(f"Force–penetration (time gradient) – {name}")
        ax.set_xlabel("Penetration (mm)")
        ax.set_ylabel("Impact force (MN)")
        ax.grid(True, alpha=0.25)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Time (ms)")
        fig.tight_layout()

        _ensure_dir(outdir)
        fig.savefig(outdir / f"force_penetration__{name}.png", dpi=dpi)
        if svg:
            fig.savefig(outdir / f"force_penetration__{name}.svg")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to extracted bundle folder or .zip")
    ap.add_argument("--outdir", default="plots_mpl", help="Output directory")
    ap.add_argument("--dpi", type=int, default=300, help="PNG DPI")
    ap.add_argument("--svg", action="store_true", help="Also write SVG")
    args = ap.parse_args()

    root_in = Path(args.bundle).expanduser().resolve()
    root = _maybe_unzip(root_in)

    outdir = Path(args.outdir).expanduser().resolve()
    _ensure_dir(outdir)

    meta = _load_meta(root)
    study = str(meta.get("study", meta.get("study_type", "study")))

    runs = _read_runs(root)

    # Core overlays
    _plot_overlay(runs, outdir, "Impact_Force_MN", title=f"Impact force – overlay ({study})", ylabel="Impact force (MN)", dpi=args.dpi, svg=args.svg)
    _plot_overlay(runs, outdir, "Acceleration_g", title=f"Acceleration – overlay ({study})", ylabel="Acceleration (g)", dpi=args.dpi, svg=args.svg)
    _plot_overlay(runs, outdir, "Penetration_mm", title=f"Penetration – overlay ({study})", ylabel="Penetration (mm)", dpi=args.dpi, svg=args.svg)

    # Envelope (if present)
    _plot_envelope(root, outdir, "Impact_Force_MN", dpi=args.dpi, svg=args.svg)
    _plot_envelope(root, outdir, "Acceleration_g", dpi=args.dpi, svg=args.svg)
    _plot_envelope(root, outdir, "Penetration_mm", dpi=args.dpi, svg=args.svg)

    # Force-penetration time gradient
    _plot_force_penetration(runs, outdir, dpi=args.dpi, svg=args.svg)

    # If we extracted a zip to temp, keep it so paths are stable only for this run.
    # (We do not delete to avoid accidentally removing user data.)

    print(f"Wrote Matplotlib plots to: {outdir}")


if __name__ == "__main__":
    main()
'''


def replot_matplotlib_script() -> bytes:
    """Return the Matplotlib replot helper as UTF-8 bytes."""
    return _REPLOT_MATPLOTLIB_PY.encode("utf-8")
