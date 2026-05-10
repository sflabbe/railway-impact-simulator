"""Hazard-informed analysis package for railway impact simulator."""
from .runout import (
    kmh_to_ms,
    ms_to_kmh,
    max_lateral_reach_m,
    runout_path_length_m,
    impact_velocity_ms,
    normal_velocity_ms,
    bounded_weibull_reach_probability,
)
from .rate import (
    base_occurrence_rate_per_year,
    scenario_occurrence_rate,
    lambda_en_crit,
    iso_demand_mask,
    exceedance_mask,
)
from .metamodel import (
    get_default_coeffs,
    power_law_feq,
    fit_power_law,
    PowerLawCoeffs,
)

__all__ = [
    "kmh_to_ms",
    "ms_to_kmh",
    "max_lateral_reach_m",
    "runout_path_length_m",
    "impact_velocity_ms",
    "normal_velocity_ms",
    "bounded_weibull_reach_probability",
    "base_occurrence_rate_per_year",
    "scenario_occurrence_rate",
    "lambda_en_crit",
    "iso_demand_mask",
    "exceedance_mask",
    "get_default_coeffs",
    "power_law_feq",
    "fit_power_law",
    "PowerLawCoeffs",
]
