from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from railway_simulator.hazard import (
    MCParams,
    forward_mc,
    inverse_iso_demand_region,
    sample_mc_scenarios,
)
from railway_simulator.hazard.montecarlo import _wall_angle_probability


SAMPLE_COLUMNS = {
    "mu",
    "beta_d_deg",
    "chi",
    "shape_k",
    "derail_rate_factor",
    "v_imp_ms",
    "v_n_ms",
    "v_n_kmh",
    "p_reach",
    "p_beta_wall",
    "lambda_s",
    "Feq_MN",
}

FORWARD_KEYS = {
    "df_samples",
    "lambda_total",
    "lambda_exceed",
    "Z_EN",
    "Z_EN_above",
    "Z_EN_below",
    "n_samples",
    "n_exceed",
    "n_iso_demand",
    "n_iso_demand_above",
    "n_iso_demand_below",
    "delta",
    "F_EN",
}


@pytest.fixture
def mc_fast() -> MCParams:
    return MCParams(n_samples=500, seed=42)


@pytest.fixture
def mc_medium() -> MCParams:
    return MCParams(n_samples=2000, seed=0)


def test_required_columns(mc_fast: MCParams) -> None:
    df = sample_mc_scenarios(80.0, 1.0, 5.0, mc_params=mc_fast)
    assert SAMPLE_COLUMNS <= set(df.columns)


def test_row_count(mc_fast: MCParams) -> None:
    df = sample_mc_scenarios(80.0, 1.0, 5.0, mc_params=mc_fast)
    assert len(df) == mc_fast.n_samples


def test_reproducibility(mc_fast: MCParams) -> None:
    df1 = sample_mc_scenarios(80.0, 1.0, 5.0, mc_params=mc_fast)
    df2 = sample_mc_scenarios(80.0, 1.0, 5.0, mc_params=mc_fast)
    pd.testing.assert_frame_equal(df1, df2)


def test_p_reach_in_unit_interval(mc_fast: MCParams) -> None:
    df = sample_mc_scenarios(80.0, 1.0, 5.0, mc_params=mc_fast)
    assert df["p_reach"].between(0.0, 1.0).all()


def test_lambda_s_nonnegative(mc_fast: MCParams) -> None:
    df = sample_mc_scenarios(80.0, 1.0, 5.0, mc_params=mc_fast)
    assert (df["lambda_s"] >= 0.0).all()


def test_feq_nonnegative(mc_fast: MCParams) -> None:
    df = sample_mc_scenarios(80.0, 1.0, 5.0, mc_params=mc_fast)
    assert (df["Feq_MN"] >= 0.0).all()


def test_custom_feq_fn(mc_fast: MCParams) -> None:
    df = sample_mc_scenarios(
        80.0,
        1.0,
        5.0,
        mc_params=mc_fast,
        feq_fn=lambda vn: 0.5 * vn,
    )
    np.testing.assert_allclose(df["Feq_MN"], 0.5 * df["v_n_kmh"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 0},
        {"n_samples": -1},
        {"mu_range": (0.0, 0.40)},
        {"mu_range": (0.40, 0.20)},
        {"chi_range": (0.0, 1.0)},
        {"chi_range": (0.25, 1.10)},
        {"chi_range": (0.80, 0.25)},
        {"beta_d_choices": []},
        {"beta_d_choices": [0.0, 5.0]},
        {"beta_d_choices": [91.0]},
        {"k_choices": []},
        {"k_choices": [0.0, 1.0]},
        {"k_choices": [-1.0]},
        {"derail_rate_factors": []},
        {"derail_rate_factors": [-0.1, 1.0]},
        {"wall_angle_model": "unknown"},
    ],
)
def test_mcparams_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MCParams(**kwargs)


@pytest.mark.parametrize("model", ["triangular_mode5", "uniform_1_45", "truncnorm_5_10"])
def test_wall_angle_probability_in_unit_interval(model: str) -> None:
    probability = _wall_angle_probability(5.0, model)
    assert 0.0 <= probability <= 1.0


def test_wall_angle_probability_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        _wall_angle_probability(5.0, "unknown")


def test_wall_angle_probability_uniform_returns_zero_outside_support() -> None:
    assert _wall_angle_probability(80.0, "uniform_1_45") == pytest.approx(0.0)


def test_wall_angle_probability_triangular_positive_near_mode() -> None:
    assert _wall_angle_probability(5.0, "triangular_mode5") > 0.0


def test_wall_angle_probability_truncnorm_positive_near_mode() -> None:
    assert _wall_angle_probability(5.0, "truncnorm_5_10") > 0.0


def test_forward_mc_required_keys(mc_fast: MCParams) -> None:
    result = forward_mc(80.0, 1.0, 5.0, F_EN=6.0, mc_params=mc_fast)
    assert FORWARD_KEYS <= set(result.keys())


def test_Z_EN_is_NOT_subset_of_lambda_exceed(mc_fast: MCParams) -> None:
    F_EN = 6.0
    result = forward_mc(
        80.0,
        1.0,
        5.0,
        F_EN=F_EN,
        delta=0.10,
        feq_fn=lambda vn: 0.95 * F_EN,
        mc_params=mc_fast,
    )
    assert result["n_exceed"] == 0
    assert result["lambda_exceed"] == pytest.approx(0.0, abs=1e-30)
    assert result["n_iso_demand_below"] > 0
    assert result["Z_EN_below"] > 0.0
    assert result["Z_EN"] > 0.0


def test_Z_EN_identity(mc_fast: MCParams) -> None:
    result = forward_mc(80.0, 1.0, 5.0, F_EN=6.0, mc_params=mc_fast)
    assert result["Z_EN"] == pytest.approx(
        result["Z_EN_above"] + result["Z_EN_below"],
        rel=1e-9,
    )


def test_rates_bounded_by_lambda_total(mc_fast: MCParams) -> None:
    result = forward_mc(80.0, 1.0, 5.0, F_EN=6.0, mc_params=mc_fast)
    assert result["lambda_exceed"] <= result["lambda_total"] + 1e-30
    assert result["Z_EN"] <= result["lambda_total"] + 1e-30


def test_custom_feq_equals_fen_gives_zero_exceed(mc_fast: MCParams) -> None:
    F_EN = 6.0
    result = forward_mc(
        80.0,
        1.0,
        5.0,
        F_EN=F_EN,
        delta=0.10,
        feq_fn=lambda vn: F_EN,
        mc_params=mc_fast,
    )
    assert result["n_exceed"] == 0
    assert result["lambda_exceed"] == pytest.approx(0.0, abs=1e-30)
    assert result["Z_EN"] > 0.0
    n_with_reach = (result["df_samples"]["p_reach"] > 0).sum()
    assert result["n_iso_demand"] == n_with_reach


def test_lambda_exceed_uses_weighted_sum(mc_fast: MCParams) -> None:
    F_EN = 6.0
    result = forward_mc(80.0, 1.0, 20.0, F_EN=F_EN, mc_params=mc_fast)
    df = result["df_samples"]
    recomputed = df.loc[df["Feq_MN"] > F_EN, "lambda_s"].sum()
    assert result["lambda_exceed"] == pytest.approx(recomputed, rel=1e-9)


def test_rates_do_not_scale_with_n_samples() -> None:
    p1 = MCParams(n_samples=500, seed=42)
    p2 = MCParams(n_samples=5000, seed=42)
    r1 = forward_mc(80.0, 1.0, 5.0, F_EN=6.0, mc_params=p1)
    r2 = forward_mc(80.0, 1.0, 5.0, F_EN=6.0, mc_params=p2)
    ratio = r2["lambda_total"] / r1["lambda_total"]
    assert 0.5 < ratio < 2.0


def test_forward_mc_exposure_scales_rates(mc_fast: MCParams) -> None:
    r1 = forward_mc(80.0, 1.0, 5.0, F_EN=6.0, exposure_km=0.05, mc_params=mc_fast)
    r2 = forward_mc(80.0, 1.0, 5.0, F_EN=6.0, exposure_km=0.10, mc_params=mc_fast)
    assert r2["lambda_total"] == pytest.approx(2.0 * r1["lambda_total"], rel=1e-9)


def test_forward_mc_loose_order_of_magnitude_smoke(mc_medium: MCParams) -> None:
    result = forward_mc(80.0, 1.0, 20.0, F_EN=6.0, mc_params=mc_medium)
    lambda_exceed = result["lambda_exceed"]
    # Loose smoke test inspired by article rates, not a direct reproduction of the table because beta_wall is fixed in this test.
    assert 1e-9 < lambda_exceed < 1e-4


def test_inverse_output_shape(mc_fast: MCParams) -> None:
    v0_grid = np.array([60.0, 100.0])
    bw_grid = np.array([1.0, 5.0, 10.0])
    df = inverse_iso_demand_region(v0_grid, bw_grid, a_m=1.0, F_EN=6.0, mc_params=mc_fast)
    assert df.shape[0] == len(v0_grid) * len(bw_grid)


def test_inverse_required_columns(mc_fast: MCParams) -> None:
    df = inverse_iso_demand_region(
        np.array([60.0, 100.0]),
        np.array([1.0, 5.0]),
        a_m=1.0,
        F_EN=6.0,
        mc_params=mc_fast,
    )
    assert {
        "v0_kmh",
        "beta_wall_deg",
        "in_iso_demand",
        "Feq_MN",
        "v_n_kmh",
        "prior_density_proxy",
        "plausibility_class",
    } <= set(df.columns)


def test_plausibility_classes_valid(mc_fast: MCParams) -> None:
    df = inverse_iso_demand_region(
        np.array([60.0, 100.0]),
        np.array([1.0, 5.0]),
        a_m=1.0,
        F_EN=6.0,
        mc_params=mc_fast,
    )
    assert set(df["plausibility_class"]) <= {"plausible", "marginal", "low_density"}


def test_in_iso_demand_is_boolean(mc_fast: MCParams) -> None:
    df = inverse_iso_demand_region(
        np.array([60.0, 100.0]),
        np.array([1.0, 5.0]),
        a_m=1.0,
        F_EN=6.0,
        mc_params=mc_fast,
    )
    assert df["in_iso_demand"].dtype == bool


def test_result_is_region_not_point(mc_fast: MCParams) -> None:
    v0_grid = np.linspace(60.0, 200.0, 20)
    bw_grid = np.linspace(1.0, 60.0, 20)
    df = inverse_iso_demand_region(v0_grid, bw_grid, a_m=1.0, F_EN=6.0, mc_params=mc_fast)
    assert df["in_iso_demand"].sum() > 1
