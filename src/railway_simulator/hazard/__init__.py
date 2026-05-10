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
from .sdof import (
    equivalent_static_force_sdof,
    compute_response_spectrum,
    termination_ratio,
    force_history_is_terminated,
)
from .engine_bridge import (
    build_one_mass_hooke_engine_params,
    extract_engine_force_history,
    equivalent_static_force_from_engine_df,
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
    "equivalent_static_force_sdof",
    "compute_response_spectrum",
    "termination_ratio",
    "force_history_is_terminated",
    "build_one_mass_hooke_engine_params",
    "extract_engine_force_history",
    "equivalent_static_force_from_engine_df",
]
