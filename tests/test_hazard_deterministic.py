from __future__ import annotations

import math

import pytest

from railway_simulator.hazard.deterministic import (
    beta_crit_deg,
    compact_summary_table,
    deterministic_runout_grid,
    lateral_regime_label,
)


def test_beta_crit_matches_threshold_relation():
    v_imp = 80.0 / 3.6
    beta = beta_crit_deg(v_imp)
    assert beta == pytest.approx(math.degrees(math.asin(1.7 / v_imp)))
    assert beta == pytest.approx(4.386, abs=0.01)


def test_beta_crit_nan_when_threshold_unreachable():
    assert math.isnan(beta_crit_deg(1.0))
    assert math.isnan(beta_crit_deg(0.0))


def test_lateral_regime_label():
    assert lateral_regime_label(10.0, 0.0) == "no_impact"
    assert lateral_regime_label(10.0, 1.0) == "guided_compatible"
    assert lateral_regime_label(10.0, 1.7) == "beyond_substitute_guidance"


def test_deterministic_grid_contains_expected_columns_and_no_probabilistic_weights():
    df = deterministic_runout_grid(
        vehicle="test",
        speeds_kmh=[80],
        distances_m=[3],
        mu_values=[0.3, 0.5],
        beta_values_deg=[2, 5],
    )
    assert len(df) == 4
    for col in [
        "v0_kmh",
        "a_m",
        "mu",
        "beta_d_deg",
        "v_imp_ms",
        "v_n_ms",
        "beta_crit_deg",
        "regime",
    ]:
        assert col in df.columns
    assert "w_guided" not in df.columns
    assert set(df["regime"]).issubset({"guided_compatible", "beyond_substitute_guidance", "no_impact"})


def test_compact_summary_collapses_mu_range():
    df = deterministic_runout_grid(
        vehicle="test",
        speeds_kmh=[80],
        distances_m=[3],
        mu_values=[0.3, 0.5],
        beta_values_deg=[5],
    )
    compact = compact_summary_table(df)
    assert len(compact) == 1
    assert compact.loc[0, "v_imp_kmh_min"] <= compact.loc[0, "v_imp_kmh_max"]
    assert compact.loc[0, "v_n_ms_min"] <= compact.loc[0, "v_n_ms_max"]
