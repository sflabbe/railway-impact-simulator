from __future__ import annotations

import math

import numpy as np
import pytest

from railway_simulator.hazard import (
    base_occurrence_rate_per_year,
    bounded_weibull_reach_probability,
    exceedance_mask,
    impact_velocity_ms,
    iso_demand_mask,
    kmh_to_ms,
    lambda_en_crit,
    max_lateral_reach_m,
    ms_to_kmh,
    normal_velocity_ms,
    runout_path_length_m,
    scenario_occurrence_rate,
)


def test_unit_conversion_round_trip() -> None:
    assert kmh_to_ms(36.0) == pytest.approx(10.0)
    assert ms_to_kmh(10.0) == pytest.approx(36.0)
    for speed in [0.0, 1.0, 80.0, 160.0]:
        assert ms_to_kmh(kmh_to_ms(speed)) == pytest.approx(speed)


@pytest.mark.parametrize(
    ("v0_kmh", "beta_d_deg", "expected_m"),
    [
        (80.0, 3.0, 4.39),
        (80.0, 5.0, 7.31),
        (120.0, 10.0, 32.78),
        (160.0, 20.0, 114.78),
    ],
)
def test_max_lateral_reach_against_reference_values(
    v0_kmh: float,
    beta_d_deg: float,
    expected_m: float,
) -> None:
    assert max_lateral_reach_m(v0_kmh, mu=0.30, beta_d_deg=beta_d_deg) == pytest.approx(
        expected_m,
        abs=0.05,
    )


def test_runout_path_length_matches_geometry() -> None:
    a_m = 8.5
    beta_d_deg = 7.0
    assert runout_path_length_m(a_m, beta_d_deg) == pytest.approx(
        a_m / math.sin(math.radians(beta_d_deg))
    )


def test_impact_velocity_zero_distance_and_beyond_runout() -> None:
    v0_kmh = 80.0
    mu = 0.30
    beta_d_deg = 5.0
    assert impact_velocity_ms(v0_kmh, 0.0, mu, beta_d_deg) == pytest.approx(kmh_to_ms(v0_kmh))

    a_max = max_lateral_reach_m(v0_kmh, mu, beta_d_deg)
    assert impact_velocity_ms(v0_kmh, a_max + 1.0, mu, beta_d_deg) == 0.0


def test_normal_velocity_at_key_angles() -> None:
    v_imp = 12.0
    assert normal_velocity_ms(v_imp, 0.0) == pytest.approx(0.0)
    assert normal_velocity_ms(v_imp, 45.0) == pytest.approx(v_imp / math.sqrt(2.0))
    assert normal_velocity_ms(v_imp, 90.0) == pytest.approx(v_imp)


def test_bounded_weibull_reach_probability_edges_and_monotonicity() -> None:
    v0_kmh = 80.0
    mu = 0.30
    beta_d_deg = 5.0
    chi = 0.75
    shape_k = 1.4
    a_max = max_lateral_reach_m(v0_kmh, mu, beta_d_deg)

    assert bounded_weibull_reach_probability(0.0, v0_kmh, mu, beta_d_deg, chi, shape_k) == 1.0
    assert bounded_weibull_reach_probability(a_max, v0_kmh, mu, beta_d_deg, chi, shape_k) == 0.0
    assert (
        bounded_weibull_reach_probability(a_max + 1.0, v0_kmh, mu, beta_d_deg, chi, shape_k)
        == 0.0
    )

    distances = np.linspace(0.0, a_max, 9)
    probabilities = [
        bounded_weibull_reach_probability(a, v0_kmh, mu, beta_d_deg, chi, shape_k)
        for a in distances
    ]
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
    assert all(a >= b for a, b in zip(probabilities, probabilities[1:]))


def test_bounded_weibull_matches_truncated_exponential_when_shape_is_one() -> None:
    v0_kmh = 120.0
    mu = 0.30
    beta_d_deg = 10.0
    chi = 0.8
    a_max = max_lateral_reach_m(v0_kmh, mu, beta_d_deg)
    a_m = 0.35 * a_max
    ell = chi * a_max
    expected = (math.exp(-(a_m / ell)) - math.exp(-(a_max / ell))) / (
        1.0 - math.exp(-(a_max / ell))
    )

    assert bounded_weibull_reach_probability(
        a_m,
        v0_kmh,
        mu,
        beta_d_deg,
        chi,
        shape_k=1.0,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chi": 0.0},
        {"chi": 1.1},
        {"shape_k": 0.0},
        {"a_m": -0.1},
        {"mu": 0.0},
        {"beta_d_deg": 0.0},
        {"beta_d_deg": 91.0},
    ],
)
def test_bounded_weibull_rejects_invalid_inputs(kwargs: dict[str, float]) -> None:
    params = {
        "a_m": 1.0,
        "v0_kmh": 80.0,
        "mu": 0.30,
        "beta_d_deg": 5.0,
        "chi": 0.75,
        "shape_k": 1.4,
    }
    params.update(kwargs)
    with pytest.raises(ValueError):
        bounded_weibull_reach_probability(**params)


def test_base_occurrence_rate_reference() -> None:
    assert base_occurrence_rate_per_year(100.0, 0.05, 0.12e-6) == pytest.approx(6.0e-7)


def test_scenario_occurrence_rate_product_and_invalid_probabilities() -> None:
    assert scenario_occurrence_rate(1.0e-6, 0.4, 0.25) == pytest.approx(1.0e-7)
    with pytest.raises(ValueError):
        scenario_occurrence_rate(1.0e-6, -0.1, 0.5)
    with pytest.raises(ValueError):
        scenario_occurrence_rate(1.0e-6, 0.5, 1.1)


def test_lambda_en_crit_identity_scaling_and_invalid_inputs() -> None:
    lambda_s = 2.0e-6
    fen = 6.0
    assert lambda_en_crit(lambda_s, feq=fen, fen=fen, alpha=2.0) == pytest.approx(lambda_s)
    assert lambda_en_crit(lambda_s, feq=12.0, fen=fen, alpha=2.0) == pytest.approx(
        lambda_s * 4.0
    )
    assert lambda_en_crit(lambda_s, feq=0.0, fen=fen, alpha=0.0) == 0.0
    with pytest.raises(ValueError):
        lambda_en_crit(lambda_s, feq=fen, fen=0.0, alpha=2.0)
    with pytest.raises(ValueError):
        lambda_en_crit(lambda_s, feq=fen, fen=fen, alpha=-1.0)


def test_iso_demand_is_NOT_subset_of_exceedance() -> None:
    feq = np.array([0.85, 0.95, 1.00, 1.05, 1.15]) * 6.0
    fen = 6.0
    iso = iso_demand_mask(feq, fen, delta=0.10)
    exc = exceedance_mask(feq, fen)
    assert iso.tolist() == [False, True, True, True, False]
    assert exc.tolist() == [False, False, False, True, True]


def test_masks_return_native_bool_for_scalars() -> None:
    assert type(iso_demand_mask(6.0, 6.0)) is bool
    assert type(exceedance_mask(6.1, 6.0)) is bool
    assert type(iso_demand_mask(np.float64(6.0), 6.0)) is bool
