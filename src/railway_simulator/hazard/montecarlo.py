from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .runout import (
    impact_velocity_ms,
    normal_velocity_ms,
    ms_to_kmh,
    bounded_weibull_reach_probability,
)
from .metamodel import get_default_coeffs, power_law_feq


_WALL_ANGLE_MODELS = {"triangular_mode5", "uniform_1_45", "truncnorm_5_10"}


def _finite_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _positive_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _finite_1d_grid(name: str, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if result.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"all {name} values must be finite")
    return result


def _finite_pair(name: str, values: tuple[float, float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low = _finite_float(f"{name}[0]", values[0])
    high = _finite_float(f"{name}[1]", values[1])
    return low, high


def _finite_choices(name: str, values: list[float]) -> list[float]:
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")
    return [_finite_float(f"{name}[{index}]", value) for index, value in enumerate(values)]


@dataclass
class MCParams:
    n_samples: int = 10_000
    seed: int | None = 42
    mu_range: tuple[float, float] = (0.20, 0.40)
    beta_d_choices: list[float] = field(default_factory=lambda: [3.0, 5.0, 10.0, 20.0])
    chi_range: tuple[float, float] = (0.25, 1.0)
    k_choices: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    derail_rate_factors: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    wall_angle_model: str = "triangular_mode5"

    def __post_init__(self) -> None:
        if isinstance(self.n_samples, bool):
            raise ValueError("n_samples must be a positive integer")
        try:
            n_samples = int(self.n_samples)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_samples must be a positive integer") from exc
        if n_samples != self.n_samples or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        self.n_samples = n_samples

        mu_low, mu_high = _finite_pair("mu_range", self.mu_range)
        if mu_low <= 0.0 or mu_high < mu_low:
            raise ValueError("mu_range must satisfy mu_range[0] > 0 and mu_range[1] >= mu_range[0]")
        self.mu_range = (mu_low, mu_high)

        chi_low, chi_high = _finite_pair("chi_range", self.chi_range)
        if chi_low <= 0.0 or chi_high < chi_low or chi_high > 1.0:
            raise ValueError("chi_range must satisfy 0 < chi_range[0] <= chi_range[1] <= 1")
        self.chi_range = (chi_low, chi_high)

        self.beta_d_choices = _finite_choices("beta_d_choices", self.beta_d_choices)
        if any(beta <= 0.0 or beta > 90.0 for beta in self.beta_d_choices):
            raise ValueError("beta_d_choices values must be in (0, 90]")

        self.k_choices = _finite_choices("k_choices", self.k_choices)
        if any(shape_k <= 0.0 for shape_k in self.k_choices):
            raise ValueError("k_choices values must be > 0")

        self.derail_rate_factors = _finite_choices(
            "derail_rate_factors",
            self.derail_rate_factors,
        )
        if any(factor < 0.0 for factor in self.derail_rate_factors):
            raise ValueError("derail_rate_factors values must be >= 0")

        if self.wall_angle_model not in _WALL_ANGLE_MODELS:
            raise ValueError(f"wall_angle_model must be one of {sorted(_WALL_ANGLE_MODELS)}")


def _wall_angle_distribution(model: str):
    if model == "triangular_mode5":
        return stats.triang(c=(5.0 - 1.0) / (90.0 - 1.0), loc=1.0, scale=89.0), 1.0, 90.0
    if model == "uniform_1_45":
        return stats.uniform(loc=1.0, scale=44.0), 1.0, 45.0
    if model == "truncnorm_5_10":
        return (
            stats.truncnorm(a=(1.0 - 5.0) / 10.0, b=(90.0 - 5.0) / 10.0, loc=5.0, scale=10.0),
            1.0,
            90.0,
        )
    raise ValueError(f"unknown wall_angle_model: {model}")


def _wall_angle_probability(beta_wall_deg: float, model: str) -> float:
    """
    Integrate the model's density over the 1-degree bin:
        [beta_wall_deg - 0.5, beta_wall_deg + 0.5]

    Clip the bin to the model support.

    Models:
        triangular_mode5:
            scipy.stats.triang(c=(5-1)/(90-1), loc=1, scale=89)
            support [1, 90]

        uniform_1_45:
            scipy.stats.uniform(loc=1, scale=44)
            support [1, 45]

        truncnorm_5_10:
            scipy.stats.truncnorm(a=(1-5)/10, b=(90-5)/10, loc=5, scale=10)
            support [1, 90]

    Return 0.0 if the bin has no overlap with the support.
    Raise ValueError for unknown model or non-finite beta_wall_deg.
    """
    beta = _finite_float("beta_wall_deg", beta_wall_deg)
    distribution, support_low, support_high = _wall_angle_distribution(model)
    bin_low = max(beta - 0.5, support_low)
    bin_high = min(beta + 0.5, support_high)
    if bin_high <= bin_low:
        return 0.0
    probability = float(distribution.cdf(bin_high) - distribution.cdf(bin_low))
    return float(np.clip(probability, 0.0, 1.0))


def _validate_sample_inputs(
    v0_kmh: float,
    a_m: float,
    beta_wall_deg: float,
    Tn_ms: float,
    n_trains_year: float,
    exposure_km: float,
    derailment_rate_per_train_km: float,
) -> tuple[float, float, float, float, float, float, float]:
    v0 = _nonnegative_float("v0_kmh", v0_kmh)
    a = _nonnegative_float("a_m", a_m)
    beta_wall = _finite_float("beta_wall_deg", beta_wall_deg)
    if beta_wall < 0.0 or beta_wall > 90.0:
        raise ValueError("beta_wall_deg must be in [0, 90]")
    period = _positive_float("Tn_ms", Tn_ms)
    n_trains = _nonnegative_float("n_trains_year", n_trains_year)
    exposure = _nonnegative_float("exposure_km", exposure_km)
    derailment_rate = _nonnegative_float(
        "derailment_rate_per_train_km",
        derailment_rate_per_train_km,
    )
    return v0, a, beta_wall, period, n_trains, exposure, derailment_rate


def sample_mc_scenarios(
    v0_kmh: float,
    a_m: float,
    beta_wall_deg: float,
    mc_params: MCParams | None = None,
    feq_fn: Callable[[float], float] | None = None,
    Tn_ms: float = 100.0,
    n_trains_year: float = 100.0,
    exposure_km: float = 0.05,
    derailment_rate_per_train_km: float = 0.12e-6,
) -> pd.DataFrame:
    """
    Draw MC samples for one fixed (v0, a, beta_wall) scenario.

    Validate:
        v0_kmh >= 0
        a_m >= 0
        beta_wall_deg in [0, 90]
        Tn_ms > 0
        n_trains_year >= 0
        exposure_km >= 0
        derailment_rate_per_train_km >= 0

    Sampling:
        rng = np.random.default_rng(mc_params.seed)
        mu ~ Uniform(mu_range)
        beta_d_deg ~ uniform discrete choice
        chi ~ Uniform(chi_range)
        shape_k ~ uniform discrete choice
        derail_rate_factor ~ uniform discrete choice

    p_beta_wall is fixed for all rows:
        _wall_angle_probability(beta_wall_deg, mc_params.wall_angle_model)

    lambda_s_i must include / n_samples:
        lambda0_i =
            n_trains_year * exposure_km * derailment_rate_per_train_km * derail_rate_factor_i
        lambda_s_i = lambda0_i * p_reach_i * p_beta_wall / mc_params.n_samples

    feq_fn:
        If provided, maps vn_kmh -> Feq_MN.
        If None, use get_default_coeffs(Tn_ms) and power_law_feq.
        Tn_ms default lookup supports 30, 100, 300 ms unless more is implemented.

    Return DataFrame columns:
        mu, beta_d_deg, chi, shape_k, derail_rate_factor,
        v_imp_ms, v_n_ms, v_n_kmh,
        p_reach, p_beta_wall, lambda_s,
        Feq_MN
    """
    params = mc_params if mc_params is not None else MCParams()
    (
        v0,
        a,
        beta_wall,
        period,
        n_trains,
        exposure,
        derailment_rate,
    ) = _validate_sample_inputs(
        v0_kmh,
        a_m,
        beta_wall_deg,
        Tn_ms,
        n_trains_year,
        exposure_km,
        derailment_rate_per_train_km,
    )

    rng = np.random.default_rng(params.seed)
    n_samples = params.n_samples
    mu = rng.uniform(params.mu_range[0], params.mu_range[1], size=n_samples)
    beta_d_deg = rng.choice(params.beta_d_choices, size=n_samples)
    chi = rng.uniform(params.chi_range[0], params.chi_range[1], size=n_samples)
    shape_k = rng.choice(params.k_choices, size=n_samples)
    derail_rate_factor = rng.choice(params.derail_rate_factors, size=n_samples)

    p_beta_wall = _wall_angle_probability(beta_wall, params.wall_angle_model)

    v_imp_ms = np.array(
        [
            impact_velocity_ms(v0, a, friction, beta_d)
            for friction, beta_d in zip(mu, beta_d_deg)
        ],
        dtype=float,
    )
    v_n_ms = np.array([normal_velocity_ms(v_imp, beta_wall) for v_imp in v_imp_ms], dtype=float)
    v_n_kmh = np.array([ms_to_kmh(v_n) for v_n in v_n_ms], dtype=float)
    p_reach = np.array(
        [
            bounded_weibull_reach_probability(a, v0, friction, beta_d, chi_value, shape)
            for friction, beta_d, chi_value, shape in zip(mu, beta_d_deg, chi, shape_k)
        ],
        dtype=float,
    )

    lambda0 = n_trains * exposure * derailment_rate * derail_rate_factor
    lambda_s = lambda0 * p_reach * p_beta_wall / n_samples

    if feq_fn is None:
        coeffs = get_default_coeffs(period)
        feq_mn = np.asarray(power_law_feq(v_n_kmh, coeffs.A, coeffs.p), dtype=float)
    else:
        feq_mn = np.array([float(feq_fn(vn)) for vn in v_n_kmh], dtype=float)
    if not np.all(np.isfinite(feq_mn)):
        raise ValueError("all Feq_MN values must be finite")
    if np.any(feq_mn < 0.0):
        raise ValueError("all Feq_MN values must be >= 0")

    return pd.DataFrame(
        {
            "mu": mu,
            "beta_d_deg": beta_d_deg,
            "chi": chi,
            "shape_k": shape_k,
            "derail_rate_factor": derail_rate_factor,
            "v_imp_ms": v_imp_ms,
            "v_n_ms": v_n_ms,
            "v_n_kmh": v_n_kmh,
            "p_reach": p_reach,
            "p_beta_wall": np.full(n_samples, p_beta_wall),
            "lambda_s": lambda_s,
            "Feq_MN": feq_mn,
        }
    )


def _iso_demand_masks(
    Feq: np.ndarray,
    F_EN: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    below_mask = ((1 - delta) * F_EN <= Feq) & (Feq <= F_EN)
    above_mask = (F_EN < Feq) & (Feq <= (1 + delta) * F_EN)
    iso_mask = below_mask | above_mask
    exceed_mask = Feq > F_EN
    return below_mask, above_mask, iso_mask, exceed_mask


def forward_mc(
    v0_kmh: float,
    a_m: float,
    beta_wall_deg: float,
    F_EN: float,
    Tn_ms: float = 100.0,
    delta: float = 0.10,
    feq_fn: Callable[[float], float] | None = None,
    mc_params: MCParams | None = None,
    n_trains_year: float = 100.0,
    exposure_km: float = 0.05,
    derailment_rate_per_train_km: float = 0.12e-6,
) -> dict:
    """
    Forward MC: exceedance rate and iso-demand band rate.

    Validate:
        F_EN > 0
        Tn_ms > 0
        delta >= 0

    Must pass exposure parameters to sample_mc_scenarios.
    """
    F_EN_value = _positive_float("F_EN", F_EN)
    period = _positive_float("Tn_ms", Tn_ms)
    delta_value = _nonnegative_float("delta", delta)

    df = sample_mc_scenarios(
        v0_kmh,
        a_m,
        beta_wall_deg,
        mc_params=mc_params,
        feq_fn=feq_fn,
        Tn_ms=period,
        n_trains_year=n_trains_year,
        exposure_km=exposure_km,
        derailment_rate_per_train_km=derailment_rate_per_train_km,
    )

    Feq = df["Feq_MN"].to_numpy()
    below_mask, above_mask, _, exceed_mask = _iso_demand_masks(
        Feq,
        F_EN_value,
        delta_value,
    )

    lambda_total = float(df["lambda_s"].sum())
    lambda_exceed = float(df.loc[exceed_mask, "lambda_s"].sum())
    Z_EN_above = float(df.loc[above_mask, "lambda_s"].sum())
    Z_EN_below = float(df.loc[below_mask, "lambda_s"].sum())
    Z_EN = Z_EN_above + Z_EN_below

    active_mask = df["lambda_s"].to_numpy() > 0.0
    n_exceed = int((active_mask & exceed_mask).sum())
    n_iso_demand_above = int((active_mask & above_mask).sum())
    n_iso_demand_below = int((active_mask & below_mask).sum())
    n_iso_demand = n_iso_demand_above + n_iso_demand_below

    if not np.isclose(Z_EN, Z_EN_above + Z_EN_below, rtol=1e-12, atol=0.0):
        raise AssertionError("Z_EN must equal Z_EN_above + Z_EN_below")
    if not (0.0 <= Z_EN <= lambda_total + 1e-30):
        raise AssertionError("Z_EN must be bounded by lambda_total")
    if not (0.0 <= lambda_exceed <= lambda_total + 1e-30):
        raise AssertionError("lambda_exceed must be bounded by lambda_total")
    if n_iso_demand != n_iso_demand_above + n_iso_demand_below:
        raise AssertionError("n_iso_demand must equal above + below counts")

    return {
        "df_samples": df,
        "lambda_total": lambda_total,
        "lambda_exceed": lambda_exceed,
        "Z_EN": Z_EN,
        "Z_EN_above": Z_EN_above,
        "Z_EN_below": Z_EN_below,
        "n_samples": len(df),
        "n_exceed": n_exceed,
        "n_iso_demand": n_iso_demand,
        "n_iso_demand_above": n_iso_demand_above,
        "n_iso_demand_below": n_iso_demand_below,
        "delta": delta_value,
        "F_EN": F_EN_value,
    }


def inverse_iso_demand_region(
    v0_kmh_grid: np.ndarray,
    beta_wall_deg_grid: np.ndarray,
    a_m: float,
    F_EN: float,
    delta: float = 0.10,
    Tn_ms: float = 100.0,
    feq_fn: Callable[[float], float] | None = None,
    mc_params: MCParams | None = None,
) -> pd.DataFrame:
    """
    Construct inverse iso-demand region in (v0, beta_wall) space.

    For each (v0, beta_wall):
        df = sample_mc_scenarios(...)
        lambda_total = sum(lambda_s)
        Z_EN = sum(lambda_s over iso-demand samples)
        in_iso_demand = Z_EN > 0
        prior_density_proxy = lambda_total

        If lambda_total > 0:
            Feq_MN = lambda_s-weighted mean Feq_MN over active samples
            v_n_kmh = lambda_s-weighted mean v_n_kmh over active samples
        Else:
            Feq_MN = arithmetic mean of Feq_MN
            v_n_kmh = arithmetic mean of v_n_kmh
            prior_density_proxy = 0
            in_iso_demand = False

    Plausibility classification:
        Use prior_density_proxy among rows where in_iso_demand == True.
        If max_density == 0:
            all rows -> "low_density"
        Else:
            plausible   if density > 2/3 * max_density
            marginal    if 1/3 * max_density <= density <= 2/3 * max_density
            low_density otherwise

    Return columns:
        v0_kmh, beta_wall_deg, in_iso_demand,
        Feq_MN, v_n_kmh, prior_density_proxy, plausibility_class
    """
    v0_values = _finite_1d_grid("v0_kmh_grid", v0_kmh_grid)
    if np.any(v0_values < 0.0):
        raise ValueError("all v0_kmh_grid values must be >= 0")
    beta_values = _finite_1d_grid("beta_wall_deg_grid", beta_wall_deg_grid)
    if np.any((beta_values < 0.0) | (beta_values > 90.0)):
        raise ValueError("all beta_wall_deg_grid values must be in [0, 90]")
    a = _nonnegative_float("a_m", a_m)
    F_EN_value = _positive_float("F_EN", F_EN)
    delta_value = _nonnegative_float("delta", delta)
    period = _positive_float("Tn_ms", Tn_ms)

    rows: list[dict[str, object]] = []
    for v0 in v0_values:
        for beta_wall in beta_values:
            df = sample_mc_scenarios(
                v0,
                a,
                beta_wall,
                mc_params=mc_params,
                feq_fn=feq_fn,
                Tn_ms=period,
            )
            Feq = df["Feq_MN"].to_numpy()
            _, _, iso_mask, _ = _iso_demand_masks(Feq, F_EN_value, delta_value)
            lambda_s = df["lambda_s"].to_numpy()
            lambda_total = float(lambda_s.sum())
            Z_EN = float(df.loc[iso_mask, "lambda_s"].sum())
            in_iso_demand = Z_EN > 0.0
            prior_density_proxy = lambda_total

            if lambda_total > 0.0:
                weights = lambda_s
                Feq_MN = float(np.average(df["Feq_MN"].to_numpy(), weights=weights))
                v_n_kmh = float(np.average(df["v_n_kmh"].to_numpy(), weights=weights))
            else:
                Feq_MN = float(df["Feq_MN"].mean())
                v_n_kmh = float(df["v_n_kmh"].mean())
                prior_density_proxy = 0.0
                in_iso_demand = False

            rows.append(
                {
                    "v0_kmh": float(v0),
                    "beta_wall_deg": float(beta_wall),
                    "in_iso_demand": bool(in_iso_demand),
                    "Feq_MN": Feq_MN,
                    "v_n_kmh": v_n_kmh,
                    "prior_density_proxy": prior_density_proxy,
                }
            )

    result = pd.DataFrame(rows)
    iso_densities = result.loc[result["in_iso_demand"], "prior_density_proxy"]
    max_density = float(iso_densities.max()) if len(iso_densities) else 0.0
    if max_density == 0.0:
        result["plausibility_class"] = "low_density"
    else:
        densities = result["prior_density_proxy"].to_numpy()
        classes = np.full(len(result), "low_density", dtype=object)
        classes[densities > (2.0 / 3.0) * max_density] = "plausible"
        marginal_mask = (densities >= (1.0 / 3.0) * max_density) & (
            densities <= (2.0 / 3.0) * max_density
        )
        classes[marginal_mask] = "marginal"
        result["plausibility_class"] = classes

    return result[
        [
            "v0_kmh",
            "beta_wall_deg",
            "in_iso_demand",
            "Feq_MN",
            "v_n_kmh",
            "prior_density_proxy",
            "plausibility_class",
        ]
    ]
