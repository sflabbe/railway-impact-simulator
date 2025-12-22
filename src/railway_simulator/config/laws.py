from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .models import InterfaceLawSpec, LinearLaw, PiecewiseLinearLaw, PlateauLaw, TabulatedCurveLaw


@dataclass(frozen=True)
class ForceDisplacementLaw:
    disp_m: np.ndarray
    force_N: np.ndarray
    gap_m: float = 0.0

    def evaluate(self, disp_m: float) -> float:
        if disp_m <= self.gap_m:
            return 0.0
        disp = disp_m - self.gap_m
        return float(np.interp(disp, self.disp_m, self.force_N, left=0.0, right=self.force_N[-1]))

    def tangent(self, disp_m: float) -> float:
        if disp_m <= self.gap_m:
            return 0.0
        disp = disp_m - self.gap_m
        idx = np.searchsorted(self.disp_m, disp, side="right") - 1
        idx = int(np.clip(idx, 0, len(self.disp_m) - 2))
        x0, x1 = self.disp_m[idx], self.disp_m[idx + 1]
        y0, y1 = self.force_N[idx], self.force_N[idx + 1]
        if x1 <= x0:
            return 0.0
        return float((y1 - y0) / (x1 - x0))

    def absorbed_energy(self, disp_m: float | None = None) -> float:
        if disp_m is None:
            x = self.disp_m
            y = self.force_N
            return float(np.trapezoid(y, x))
        if disp_m <= self.gap_m:
            return 0.0
        disp = disp_m - self.gap_m
        disp = float(np.clip(disp, 0.0, self.disp_m[-1]))
        x = np.concatenate([self.disp_m[self.disp_m <= disp], [disp]])
        y = np.concatenate([self.force_N[self.disp_m <= disp], [self.evaluate(disp_m)]])
        return float(np.trapezoid(y, x))


def compute_absorbed_energy(points: Iterable[Tuple[float, float]]) -> float:
    disp, force = _sorted_points(points)
    return float(np.trapezoid(force, disp))


def build_law_from_spec(spec: InterfaceLawSpec, *, config_dir: Path) -> ForceDisplacementLaw:
    if isinstance(spec, LinearLaw):
        disp = np.array([0.0, 1.0], dtype=float)
        force = np.array([0.0, spec.k_N_per_m], dtype=float)
        return ForceDisplacementLaw(disp, force, gap_m=spec.gap_m)
    if isinstance(spec, PiecewiseLinearLaw):
        disp, force = _sorted_points(spec.points)
        return ForceDisplacementLaw(disp, force, gap_m=spec.gap_m)
    if isinstance(spec, PlateauLaw):
        disp = np.array(
            [0.0, spec.ramp_disp_m, spec.plateau_disp_m, spec.final_disp_m],
            dtype=float,
        )
        force = np.array(
            [0.0, spec.ramp_to_force_N, spec.plateau_force_N, spec.final_force_N],
            dtype=float,
        )
        return ForceDisplacementLaw(disp, force, gap_m=spec.gap_m)
    if isinstance(spec, TabulatedCurveLaw):
        csv_path = (config_dir / spec.csv_path).resolve()
        df = pd.read_csv(csv_path)
        if "disp_m" not in df.columns or "force_N" not in df.columns:
            raise ValueError(f"CSV {csv_path} must include disp_m and force_N columns")
        disp = df["disp_m"].to_numpy(dtype=float)
        force = df["force_N"].to_numpy(dtype=float)
        disp, force = _sorted_points(zip(disp, force))
        return ForceDisplacementLaw(disp, force, gap_m=spec.gap_m)
    raise ValueError(f"Unsupported interface law: {spec}")


def _sorted_points(points: Iterable[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    pts: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in points]
    pts.sort(key=lambda item: item[0])
    disp = np.array([p[0] for p in pts], dtype=float)
    force = np.array([p[1] for p in pts], dtype=float)
    return disp, force
