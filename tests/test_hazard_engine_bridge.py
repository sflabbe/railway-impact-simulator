from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pandas as pd
import pytest

from railway_simulator.hazard import (
    build_one_mass_hooke_engine_params,
    equivalent_static_force_from_engine_df,
    extract_engine_force_history,
    force_history_is_terminated,
)


pytestmark = pytest.mark.slow

MASS_KG = 1000.0
K_WALL = 1.0e6
VN_MS = 1.0
DT_S = 1e-4
T_MAX_S = 0.20


def run_engine_quiet(params):
    from railway_simulator.core.engine import run_simulation

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        try:
            return run_simulation(params, emit_peak_diagnostics=False)
        except TypeError:
            return run_simulation(params)


@pytest.fixture(scope="module")
def oracle() -> dict[str, float]:
    omega = math.sqrt(K_WALL / MASS_KG)
    return {
        "omega_rad_s": omega,
        "F_peak_N": VN_MS * math.sqrt(K_WALL * MASS_KG),
        "u_peak_m": VN_MS / omega,
        "t_contact_s": math.pi / omega,
    }


@pytest.fixture(scope="module")
def engine_params() -> dict:
    return build_one_mass_hooke_engine_params(
        mass_kg=MASS_KG,
        k_wall_N_m=K_WALL,
        v_n_ms=VN_MS,
        dt_s=DT_S,
        t_max_s=T_MAX_S,
    )


@pytest.fixture(scope="module")
def engine_df(engine_params: dict) -> pd.DataFrame:
    return run_engine_quiet(engine_params)


@pytest.fixture(scope="module")
def engine_df_2x() -> pd.DataFrame:
    params = build_one_mass_hooke_engine_params(
        mass_kg=MASS_KG,
        k_wall_N_m=K_WALL,
        v_n_ms=2.0 * VN_MS,
        dt_s=DT_S,
        t_max_s=T_MAX_S,
    )
    return run_engine_quiet(params)


def test_returns_dataframe(engine_df: pd.DataFrame) -> None:
    assert isinstance(engine_df, pd.DataFrame)
    assert not engine_df.empty


def test_required_columns_present(engine_df: pd.DataFrame) -> None:
    assert {"Time_s", "Impact_Force_MN"}.issubset(engine_df.columns)


def test_impact_force_nonnegative(engine_df: pd.DataFrame) -> None:
    force = engine_df["Impact_Force_MN"].to_numpy(dtype=float)
    assert np.all(force >= -1e-12)


def test_energy_diagnostic(engine_df: pd.DataFrame) -> None:
    if "E_num_ratio" in engine_df.columns:
        max_ratio = float(engine_df["E_num_ratio"].abs().max())
        assert max_ratio < 0.01
    elif {"E0_J", "E_num_J"}.issubset(engine_df.columns):
        E0 = float(engine_df["E0_J"].iloc[0])
        max_err = float(engine_df["E_num_J"].abs().max())
        rel_err = max_err / (abs(E0) + 1e-16)
        assert rel_err < 0.01
    elif {"E_total_initial_J", "E_balance_error_J"}.issubset(engine_df.columns):
        E0 = float(engine_df["E_total_initial_J"].iloc[0])
        max_err = float(engine_df["E_balance_error_J"].abs().max())
        rel_err = max_err / (abs(E0) + 1e-16)
        assert rel_err < 0.01
    else:
        pytest.skip(f"No recognized energy diagnostic columns: {engine_df.columns.tolist()}")


def test_peak_force_matches_analytical(engine_df: pd.DataFrame, oracle: dict[str, float]) -> None:
    F_peak_analytical_MN = oracle["F_peak_N"] / 1e6
    F_peak_engine_MN = engine_df["Impact_Force_MN"].max()
    assert F_peak_engine_MN == pytest.approx(F_peak_analytical_MN, rel=0.02)


def test_peak_penetration_matches_analytical(
    engine_df: pd.DataFrame,
    oracle: dict[str, float],
) -> None:
    if "Penetration_mm" not in engine_df.columns:
        pytest.skip("Penetration_mm column not available")
    u_peak_analytical_mm = oracle["u_peak_m"] * 1000.0
    u_peak_engine_mm = engine_df["Penetration_mm"].max()
    assert u_peak_engine_mm == pytest.approx(u_peak_analytical_mm, rel=0.02)


def test_complete_simulation_passes_termination(engine_df: pd.DataFrame) -> None:
    _, force = extract_engine_force_history(engine_df)
    assert force_history_is_terminated(force)


def test_truncated_simulation_fails_termination(oracle: dict[str, float]) -> None:
    params = build_one_mass_hooke_engine_params(
        mass_kg=MASS_KG,
        k_wall_N_m=K_WALL,
        v_n_ms=VN_MS,
        dt_s=DT_S,
        t_max_s=oracle["t_contact_s"] * 0.5,
    )
    truncated_df = run_engine_quiet(params)
    _, force = extract_engine_force_history(truncated_df)
    assert not force_history_is_terminated(force)


def test_peak_force_scales_linearly_with_velocity(
    engine_df: pd.DataFrame,
    engine_df_2x: pd.DataFrame,
) -> None:
    F_peak_engine_MN = float(engine_df["Impact_Force_MN"].max())
    F_peak_engine_2x_MN = float(engine_df_2x["Impact_Force_MN"].max())
    assert F_peak_engine_2x_MN == pytest.approx(2.0 * F_peak_engine_MN, rel=0.02)


def test_feq_positive(engine_df: pd.DataFrame) -> None:
    assert equivalent_static_force_from_engine_df(engine_df) > 0.0


def test_feq_increases_with_velocity(
    engine_df: pd.DataFrame,
    engine_df_2x: pd.DataFrame,
) -> None:
    feq = equivalent_static_force_from_engine_df(engine_df)
    feq_2x = equivalent_static_force_from_engine_df(engine_df_2x)
    assert feq_2x > feq


def test_extract_returns_arrays(engine_df: pd.DataFrame) -> None:
    time_s, force = extract_engine_force_history(engine_df)
    assert isinstance(time_s, np.ndarray)
    assert isinstance(force, np.ndarray)
    assert time_s.shape == force.shape
    assert time_s.size > 0


def test_extract_time_strictly_increasing(engine_df: pd.DataFrame) -> None:
    time_s, _ = extract_engine_force_history(engine_df)
    assert np.all(np.diff(time_s) > 0.0)


def test_extract_missing_column_raises(engine_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        extract_engine_force_history(engine_df.drop(columns=["Impact_Force_MN"]))


def test_extract_rejects_nonfinite(engine_df: pd.DataFrame) -> None:
    bad_df = engine_df.copy()
    bad_df.loc[bad_df.index[0], "Impact_Force_MN"] = np.nan
    with pytest.raises(ValueError):
        extract_engine_force_history(bad_df)
