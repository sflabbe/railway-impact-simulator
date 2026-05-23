"""Lateral wall-impact sensitivity utilities.

This module implements the final v2.3 specification layer without changing the
core HHT engine.  It builds single-mass and yawed lateral engine configurations,
post-processes per-mass wall reactions into global/local equivalent static
forces, and provides the explicit scenario-grid analytical exceedance model.

Important scope notes
---------------------
* The current core engine uses an infinite, frictionless wall at x=0.
* Tangential velocity is represented by a wall-tangential co-moving frame
  (baseline v_t = 0).
* Yaw is a fixed initial geometric input, not a dynamic rotational DOF.
* Local equivalent force is max_i SDOF(R_i^+(t)), not SDOF(max_i R_i^+(t)).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .engine_bridge import extract_engine_force_history
from .sdof import equivalent_static_force_sdof


V_N_MAX_DEFAULT = 20.0  # m/s, calibrated upper limit used by the spec
V_N_MIN_LAT_DEFAULT = 1.0  # m/s, practical initial lower limit for lateral runs
DEFAULT_TN_S = 0.100
DEFAULT_ZETA = 0.05
DEFAULT_SINGLE_MASS_KG = 13_000.0


@dataclass(frozen=True)
class SDOFSettings:
    """Equivalent-static SDOF response parameters."""

    Tn_s: float = DEFAULT_TN_S
    zeta: float = DEFAULT_ZETA
    oscillator_mass: float = 1.0


@dataclass(frozen=True)
class EquivalentDemandResult:
    """Global/local equivalent-force postprocessing result."""

    total_feq_MN: float
    local_feq_MN: float
    per_mass_feq_MN: tuple[float, ...]
    total_peak_MN: float
    local_peak_MN: float
    per_mass_peak_MN: tuple[float, ...]
    eta: tuple[float, ...] | None = None
    n_contacts: tuple[int, ...] | None = None
    first_contact_vx_m_s: tuple[float | None, ...] | None = None


@dataclass(frozen=True)
class VelocityFamily:
    """Uniform lateral-velocity supports for the three mechanism regimes."""

    name: str
    guided: tuple[float, float]
    excursion: tuple[float, float]
    severe: tuple[float, float]

    @property
    def intervals(self) -> dict[str, tuple[float, float]]:
        return {
            "guided": self.guided,
            "excursion": self.excursion,
            "severe": self.severe,
        }


@dataclass(frozen=True)
class MechanismWeights:
    """Scenario coordinates for mechanism composition."""

    w_guided: float
    w_excursion: float
    w_severe: float
    rho: float

    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "guided": self.w_guided,
            "excursion": self.w_excursion,
            "severe": self.w_severe,
        }


@dataclass(frozen=True)
class ScenarioExceedanceResult:
    """Analytical scenario exceedance for one speed/family/weight point."""

    pi: float
    feasible_weights: dict[str, float]
    truncated_intervals: dict[str, tuple[float, float]]
    active_regimes: tuple[str, ...]


def _positive_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _nonnegative_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _base_without_time_inconsistency(base_config: Mapping) -> dict:
    cfg = dict(base_config)
    # Avoid stale step/T_int when T_max or h_init is updated later.  The public
    # run_simulation wrapper will reconstruct a consistent grid if step is absent.
    cfg.pop("step", None)
    cfg.pop("T_int", None)
    return cfg


def build_single_mass_reference_params(
    base_config: Mapping,
    v_n_ms: float,
    mass_kg: float = DEFAULT_SINGLE_MASS_KG,
    t_max_s: float | None = None,
    h_init_s: float | None = None,
) -> dict:
    """Build a one-mass reference case for ``G_eq_single(v_n)``.

    The contact law, contact coefficient, restitution/damping parameter,
    integrator settings and solver settings are inherited from ``base_config``.
    Inter-mass springs are removed.
    """
    vn = _positive_float("v_n_ms", v_n_ms)
    mass = _positive_float("mass_kg", mass_kg)
    cfg = _base_without_time_inconsistency(base_config)

    if h_init_s is not None:
        cfg["h_init"] = _positive_float("h_init_s", h_init_s)
    if t_max_s is not None:
        cfg["T_max"] = _positive_float("t_max_s", t_max_s)
    elif "T_max" not in cfg or cfg.get("T_max") is None:
        cfg["T_max"] = 0.4

    cfg.update(
        {
            "n_masses": 1,
            "masses": [mass],
            "x_init": [0.0],
            "y_init": [0.0],
            "v0_init": -vn,
            "angle_rad": 0.0,
            "k_train": [],
            "fy": [],
            "uy": [],
            "case_name": f"single_mass_reference_vn_{vn:.3f}",
        }
    )
    return cfg


def _longitudinal_coordinates_from_config(base_config: Mapping) -> np.ndarray:
    x = np.asarray(base_config.get("x_init"), dtype=float)
    y = np.asarray(base_config.get("y_init", np.zeros_like(x)), dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("base_config x_init must be a non-empty 1D sequence")
    if y.shape != x.shape:
        raise ValueError("base_config y_init must have the same length as x_init")
    # Use arclength from the first mass based on the existing centerline.  For
    # the ICE1 config this is simply [0, 3.6, ..., 18.0].
    if x.size == 1:
        return np.array([0.0], dtype=float)
    diffs = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], np.cumsum(diffs)])


def lateral_initial_coordinates(
    base_config: Mapping,
    psi_rad: float,
    s_ref: str | float = "closest",
) -> tuple[np.ndarray, np.ndarray]:
    """Return unshifted ``x_init, y_init`` for yawed lateral wall impact.

    The core engine later adds ``d0`` to all x-coordinates.  This function
    therefore returns x-coordinates with minimum zero, so the closest mass starts
    at the local gap ``d0`` after engine setup.
    """
    psi = float(psi_rad)
    if not math.isfinite(psi):
        raise ValueError("psi_rad must be finite")

    s = _longitudinal_coordinates_from_config(base_config)
    if isinstance(s_ref, str):
        if s_ref == "front":
            s0 = float(s[0])
        elif s_ref == "center":
            s0 = float(0.5 * (s[0] + s[-1]))
        elif s_ref == "closest":
            s0 = float(s[0])
        else:
            raise ValueError("s_ref must be 'front', 'center', 'closest', or a float")
    else:
        s0 = float(s_ref)
        if not math.isfinite(s0):
            raise ValueError("s_ref must be finite")

    raw_x = (s - s0) * math.sin(psi)
    # Closest point should start at x=d0 after the core engine adds d0.
    x_init = raw_x - float(np.min(raw_x))
    y_init = (s - s0) * math.cos(psi)
    return x_init.astype(float), y_init.astype(float)


def estimate_t_last_s(x_init_unshifted: Sequence[float], d0_m: float, v_n_ms: float) -> float:
    """Estimate last first-contact time for a lateral configuration."""
    vn = _positive_float("v_n_ms", v_n_ms)
    d0 = _nonnegative_float("d0_m", d0_m)
    x = np.asarray(x_init_unshifted, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x_init_unshifted must be a finite non-empty 1D sequence")
    return float(np.max(x + d0) / vn)


def lateral_t_max_s(
    x_init_unshifted: Sequence[float],
    d0_m: float,
    v_n_ms: float,
    Tn_s: float = DEFAULT_TN_S,
    tail_min_s: float = 0.2,
) -> float:
    """Spec v2.3 lateral duration: t_last + max(3*Tn, tail_min)."""
    Tn = _positive_float("Tn_s", Tn_s)
    tail = max(3.0 * Tn, _positive_float("tail_min_s", tail_min_s))
    return estimate_t_last_s(x_init_unshifted, d0_m, v_n_ms) + tail


def build_lateral_yaw_params(
    base_config: Mapping,
    v_n_ms: float,
    psi_rad: float,
    *,
    s_ref: str | float = "closest",
    Tn_s: float = DEFAULT_TN_S,
    tail_min_s: float = 0.2,
    h_init_s: float | None = None,
    v_t_ms: float = 0.0,
) -> dict:
    """Build engine params for a yawed lateral-wall configuration.

    The current core engine cannot prescribe a tangential velocity independently
    from ``angle_rad``.  The baseline specification uses a wall-tangential
    co-moving frame, so this builder requires ``v_t_ms == 0``.
    """
    vn = _positive_float("v_n_ms", v_n_ms)
    vt = float(v_t_ms)
    if not math.isfinite(vt):
        raise ValueError("v_t_ms must be finite")
    if abs(vt) > 1e-12:
        raise NotImplementedError(
            "Non-zero tangential velocity is outside the current lateral builder; "
            "baseline uses a wall-tangential co-moving frame."
        )

    cfg = _base_without_time_inconsistency(base_config)
    x_init, y_init = lateral_initial_coordinates(base_config, psi_rad, s_ref=s_ref)
    d0 = float(cfg.get("d0", 0.0))
    if d0 < 0.0:
        raise ValueError("base_config d0 must be >= 0")

    if h_init_s is not None:
        cfg["h_init"] = _positive_float("h_init_s", h_init_s)
    cfg["T_max"] = lateral_t_max_s(x_init, d0, vn, Tn_s=Tn_s, tail_min_s=tail_min_s)

    cfg.update(
        {
            "x_init": x_init.tolist(),
            "y_init": y_init.tolist(),
            "v0_init": -vn,
            "angle_rad": 0.0,
            "case_name": f"lateral_yaw_vn_{vn:.3f}_psi_{math.degrees(psi_rad):.3f}",
        }
    )
    return cfg


def wall_reaction_histories_N(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-mass positive wall reaction histories in N.

    Return ``(time_s, forces_N)`` with shape ``forces_N = (n_masses, n_time)``.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty pandas DataFrame")
    if "Time_s" not in df.columns:
        raise ValueError("df must contain Time_s")
    n_masses = int(df.attrs.get("n_masses", 0) or 0)
    if n_masses <= 0:
        cols = [c for c in df.columns if c.startswith("Mass") and c.endswith("_Force_wall_x_N")]
        n_masses = len(cols)
    if n_masses <= 0:
        raise ValueError("could not determine number of masses")

    time_s = df["Time_s"].to_numpy(dtype=float)
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("Time_s must be strictly increasing")

    forces = []
    for i in range(1, n_masses + 1):
        col = f"Mass{i}_Force_wall_x_N"
        if col not in df.columns:
            raise ValueError(f"missing wall force column {col}")
        forces.append(np.maximum(df[col].to_numpy(dtype=float), 0.0))
    return time_s, np.vstack(forces)


def first_contact_velocities_x_ms(
    df: pd.DataFrame,
    forces_N: np.ndarray | None = None,
    threshold_N: float = 1.0,
) -> tuple[float | None, ...]:
    """Return x-velocity at first positive contact sample for each mass."""
    threshold = _nonnegative_float("threshold_N", threshold_N)
    if forces_N is None:
        _, forces_N = wall_reaction_histories_N(df)
    n_masses = forces_N.shape[0]
    out: list[float | None] = []
    for i in range(n_masses):
        idx = np.flatnonzero(forces_N[i] > threshold)
        if idx.size == 0:
            out.append(None)
            continue
        col = f"Mass{i + 1}_Velocity_x_m_s"
        if col not in df.columns:
            out.append(None)
        else:
            out.append(float(df[col].to_numpy(dtype=float)[int(idx[0])]))
    return tuple(out)


def count_contact_pulses(force_N: Sequence[float], threshold_N: float = 1.0) -> int:
    """Count contiguous positive-contact intervals above a threshold."""
    threshold = _nonnegative_float("threshold_N", threshold_N)
    f = np.asarray(force_N, dtype=float)
    if f.ndim != 1 or f.size == 0 or not np.all(np.isfinite(f)):
        raise ValueError("force_N must be a finite non-empty 1D sequence")
    active = f > threshold
    if not np.any(active):
        return 0
    starts = np.flatnonzero(active & np.concatenate([[True], ~active[:-1]]))
    return int(starts.size)


def equivalent_demands_from_wall_reactions(
    time_s: Sequence[float],
    reactions_N: np.ndarray,
    *,
    sdof: SDOFSettings = SDOFSettings(),
    reference_single_MN: float | None = None,
    contact_threshold_N: float = 1.0,
) -> EquivalentDemandResult:
    """Compute global/local equivalent static forces from per-mass reactions."""
    t = np.asarray(time_s, dtype=float)
    R = np.asarray(reactions_N, dtype=float)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("time_s must be a strictly increasing 1D sequence with length >= 2")
    if R.ndim != 2 or R.shape[1] != t.size or R.shape[0] == 0:
        raise ValueError("reactions_N must have shape (n_masses, len(time_s))")
    if not np.all(np.isfinite(R)):
        raise ValueError("reactions_N must be finite")
    R_pos = np.maximum(R, 0.0)

    total_N = np.sum(R_pos, axis=0)
    total_feq_N = equivalent_static_force_sdof(
        t, total_N, Tn_s=sdof.Tn_s, zeta=sdof.zeta, oscillator_mass=sdof.oscillator_mass
    )
    per_mass_feq_N = np.array(
        [
            equivalent_static_force_sdof(
                t, R_pos[i], Tn_s=sdof.Tn_s, zeta=sdof.zeta, oscillator_mass=sdof.oscillator_mass
            )
            for i in range(R_pos.shape[0])
        ],
        dtype=float,
    )
    per_mass_peak_N = np.max(R_pos, axis=1)

    per_mass_feq_MN = tuple((per_mass_feq_N / 1e6).tolist())
    eta = None
    if reference_single_MN is not None:
        ref = _positive_float("reference_single_MN", reference_single_MN)
        eta = tuple((per_mass_feq_N / 1e6 / ref).tolist())

    n_contacts = tuple(count_contact_pulses(row, threshold_N=contact_threshold_N) for row in R_pos)

    return EquivalentDemandResult(
        total_feq_MN=float(total_feq_N / 1e6),
        local_feq_MN=float(np.max(per_mass_feq_N) / 1e6),
        per_mass_feq_MN=per_mass_feq_MN,
        total_peak_MN=float(np.max(total_N) / 1e6),
        local_peak_MN=float(np.max(per_mass_peak_N) / 1e6),
        per_mass_peak_MN=tuple((per_mass_peak_N / 1e6).tolist()),
        eta=eta,
        n_contacts=n_contacts,
        first_contact_vx_m_s=None,
    )


def equivalent_demands_from_engine_df(
    df: pd.DataFrame,
    *,
    sdof: SDOFSettings = SDOFSettings(),
    reference_single_MN: float | None = None,
    contact_threshold_N: float = 1.0,
) -> EquivalentDemandResult:
    """Compute global/local equivalent demands from a core-engine DataFrame."""
    time_s, reactions = wall_reaction_histories_N(df)
    result = equivalent_demands_from_wall_reactions(
        time_s,
        reactions,
        sdof=sdof,
        reference_single_MN=reference_single_MN,
        contact_threshold_N=contact_threshold_N,
    )
    return EquivalentDemandResult(
        total_feq_MN=result.total_feq_MN,
        local_feq_MN=result.local_feq_MN,
        per_mass_feq_MN=result.per_mass_feq_MN,
        total_peak_MN=result.total_peak_MN,
        local_peak_MN=result.local_peak_MN,
        per_mass_peak_MN=result.per_mass_peak_MN,
        eta=result.eta,
        n_contacts=result.n_contacts,
        first_contact_vx_m_s=first_contact_velocities_x_ms(
            df, reactions, threshold_N=contact_threshold_N
        ),
    )


def single_mass_reference_feq_MN(
    df: pd.DataFrame,
    *,
    sdof: SDOFSettings = SDOFSettings(),
) -> float:
    """Compute the single-mass reference Feq in MN from a one-mass engine run."""
    return equivalent_static_force_sdof(
        *extract_engine_force_history(df, force_col="Impact_Force_MN", time_col="Time_s"),
        Tn_s=sdof.Tn_s,
        zeta=sdof.zeta,
        oscillator_mass=sdof.oscillator_mass,
    )


def default_velocity_families() -> tuple[VelocityFamily, ...]:
    """Return the v2.3 low/nominal/high velocity families."""
    return (
        VelocityFamily("V_L", guided=(0.1, 1.7), excursion=(1.7, 4.0), severe=(4.0, 12.0)),
        VelocityFamily("V_N", guided=(0.1, 1.7), excursion=(1.7, 6.0), severe=(6.0, 20.0)),
        VelocityFamily("V_H", guided=(0.1, 1.7), excursion=(1.7, 8.0), severe=(8.0, 20.0)),
    )


def scenario_weight_grid(
    w_s_values: Sequence[float] = (0.01, 0.03, 0.05, 0.10, 0.15, 0.25),
    rho_values: Sequence[float] = (0.10, 0.25, 0.40, 0.60, 0.80),
) -> tuple[MechanismWeights, ...]:
    """Return the explicit 30-point mechanism-weight grid."""
    grid: list[MechanismWeights] = []
    for w_s in w_s_values:
        ws = float(w_s)
        if not (0.0 <= ws < 1.0):
            raise ValueError("all w_s values must be in [0, 1)")
        for rho in rho_values:
            r = float(rho)
            if not (0.0 <= r <= 1.0):
                raise ValueError("all rho values must be in [0, 1]")
            w_e = r * (1.0 - ws)
            w_g = (1.0 - r) * (1.0 - ws)
            grid.append(MechanismWeights(w_guided=w_g, w_excursion=w_e, w_severe=ws, rho=r))
    return tuple(grid)


def scenario_grid() -> tuple[tuple[MechanismWeights, VelocityFamily], ...]:
    """Return the full 90-point weight × velocity-family grid."""
    return tuple((w, fam) for w in scenario_weight_grid() for fam in default_velocity_families())


def scenario_exceedance_from_vstar(
    *,
    v_imp_ms: float,
    v_star_ms: float,
    weights: MechanismWeights,
    family: VelocityFamily,
    v_n_max_ms: float = V_N_MAX_DEFAULT,
) -> ScenarioExceedanceResult:
    """Analytical exceedance for one scenario using a threshold normal velocity."""
    v_imp = _positive_float("v_imp_ms", v_imp_ms)
    v_star = _nonnegative_float("v_star_ms", v_star_ms)
    vmax = _positive_float("v_n_max_ms", v_n_max_ms)

    intervals: dict[str, tuple[float, float]] = {}
    feasible_raw: dict[str, float] = {}
    for name, (a_raw, b_raw) in family.intervals.items():
        a = float(a_raw)
        b = min(float(b_raw), v_imp, vmax)
        if b > a:
            intervals[name] = (a, b)
            feasible_raw[name] = weights.as_dict[name]

    denom = sum(feasible_raw.values())
    if denom <= 0.0:
        raise ValueError("all regimes infeasible for this impact speed")

    feasible_weights = {name: value / denom for name, value in feasible_raw.items()}

    def regime_prob(a: float, b: float) -> float:
        if v_star <= a:
            return 1.0
        if v_star >= b:
            return 0.0
        return float((b - v_star) / (b - a))

    pi = sum(feasible_weights[name] * regime_prob(*intervals[name]) for name in intervals)
    return ScenarioExceedanceResult(
        pi=float(pi),
        feasible_weights=feasible_weights,
        truncated_intervals=intervals,
        active_regimes=tuple(intervals),
    )


def grid_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Descriptive summary over a discrete scenario grid."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("values must be a finite non-empty 1D sequence")
    return {
        "pi_min_grid": float(np.min(arr)),
        "q5_grid": float(np.percentile(arr, 5)),
        "q50_grid": float(np.percentile(arr, 50)),
        "q95_grid": float(np.percentile(arr, 95)),
        "pi_max_grid": float(np.max(arr)),
        "n_zero": int(np.sum(np.isclose(arr, 0.0))),
        "n_active": int(np.sum(arr > 0.0)),
    }
