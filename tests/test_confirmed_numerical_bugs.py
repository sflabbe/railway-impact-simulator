from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from helpers.force_reporting import wall_force_deltas
from railway_simulator.config.laws import ForceDisplacementLaw
from railway_simulator.config.loader import apply_collision_to_params, normalize_config_dict
from railway_simulator.core.contact import ContactModels
from railway_simulator.core.engine import get_default_simulation_params, run_simulation
from railway_simulator.core.integrator import HHTAlphaIntegrator
from railway_simulator.core.parametric import build_speed_scenarios


REPO_ROOT = Path(__file__).resolve().parents[1]


def _quiet_run(params: dict):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return run_simulation(params, emit_peak_diagnostics=False)


def _linear_sdof_energy_history(alpha: float, *, steps: int = 2000) -> np.ndarray:
    integrator = HHTAlphaIntegrator(alpha)
    m = 1.0
    h = 1.0
    omega = 0.5 / h
    k = omega**2 * m
    M = np.array([[m]])
    C = np.zeros((1, 1))
    zero = np.zeros(1)

    q = np.array([1.0])
    v = np.array([0.0])
    a = np.array([-k * q[0] / m])
    energies = [0.5 * m * v[0] ** 2 + 0.5 * k * q[0] ** 2]

    for _ in range(steps):
        q_old = q.copy()
        v_old = v.copy()
        a_old = a.copy()
        R_old = -k * q_old
        a_new = a_old.copy()

        for _iteration in range(50):
            q_trial, v_trial = integrator.predict(q_old, v_old, a_old, a_new, h)
            a_next = integrator.compute_acceleration(
                M,
                -k * q_trial,
                R_old,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                C,
                v_trial,
                v_old,
            )
            if np.linalg.norm(a_next - a_new) < 1.0e-13:
                a_new = a_next
                break
            a_new = a_next
        else:
            raise AssertionError("linear HHT SDOF solve did not converge")

        q, v = integrator.predict(q_old, v_old, a_old, a_new, h)
        a = a_new
        energies.append(0.5 * m * v[0] ** 2 + 0.5 * k * q[0] ** 2)

    return np.asarray(energies)


def test_hht_negative_alpha_damps_linear_sdof_energy() -> None:
    energy = _linear_sdof_energy_history(-0.15)
    assert energy[-1] / energy[0] < 1.0


def test_hht_alpha_zero_is_newmark_average_acceleration() -> None:
    integrator = HHTAlphaIntegrator(0.0)
    assert integrator.beta == pytest.approx(0.25)
    assert integrator.gamma == pytest.approx(0.5)
    info = integrator.get_stability_info()
    assert info["beta"] == pytest.approx(0.25)
    assert info["gamma"] == pytest.approx(0.5)

    energy = _linear_sdof_energy_history(0.0)
    assert energy[-1] / energy[0] == pytest.approx(1.0, rel=1.0e-9, abs=1.0e-9)


@pytest.mark.parametrize("alpha", [-0.34, 0.01])
def test_hht_alpha_outside_negative_convention_range_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"\[-1/3, 0\]"):
        HHTAlphaIntegrator(alpha)


def test_collision_preset_tabulated_law_sets_actual_contact_model() -> None:
    config_path = REPO_ROOT / "configs" / "mi_en15227_c1.yml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = apply_collision_to_params(normalize_config_dict(raw, filename=config_path.name), config_path=config_path)
    assert params["contact_model"] == "tabulated"
    assert params.get("contact_law") is not None
    assert params["collision_meta"]["contact_model_declared"] == "lankarani-nikravesh"


def test_contact_law_with_non_tabulated_model_raises() -> None:
    law = ForceDisplacementLaw(np.array([0.0, 0.1]), np.array([0.0, 1.0e6]))
    with pytest.raises(ValueError, match="contact_law.*tabulated"):
        ContactModels.compute_force(
            np.array([-0.01]),
            np.array([-1.0]),
            np.array([1.0]),
            1.0e6,
            0.7,
            "lankarani-nikravesh",
            contact_law=law,
        )


@pytest.mark.parametrize(
    "model",
    ["flores", "gonthier", "ye", "pant-wijeyewickrema", "anagnostopoulos"],
)
def test_contact_models_that_divide_by_restitution_reject_cr_zero(model: str) -> None:
    with pytest.raises(ValueError, match="requires cr_wall > 0"):
        ContactModels.compute_force(
            np.array([-0.01]),
            np.array([-1.0]),
            np.array([1.0]),
            1.0e6,
            0.0,
            model,
        )


def test_force_displacement_law_plateau_energy_is_consistent() -> None:
    law = ForceDisplacementLaw(np.array([0.0, 0.1]), np.array([0.0, 1.0e6]))
    assert law.evaluate(0.3) == pytest.approx(1.0e6)
    assert law.tangent(0.3) == pytest.approx(0.0)
    assert law.absorbed_energy(0.3) - law.absorbed_energy(0.1) == pytest.approx(1.0e6 * 0.2)


def _friction_case(model: str) -> dict:
    params = get_default_simulation_params()
    params.update(
        {
            "T_max": 5.0e-4,
            "h_init": 1.0e-4,
            "step": 5,
            "T_int": (0.0, 5.0e-4),
            "d0": 100.0,
            "v0_init": -2.0,
            "friction_model": model,
            "mu_s": 0.3,
            "mu_k": 0.2,
            "sigma_0": 1.0e5,
            "sigma_1": 1.0e3,
            "sigma_2": 0.0,
            "solver": "newton",
        }
    )
    return params


@pytest.mark.parametrize("upper, lower", [("Coulomb", "coulomb"), ("DAHL", "dahl")])
def test_friction_model_dispatch_is_case_insensitive(upper: str, lower: str) -> None:
    df_upper = _quiet_run(_friction_case(upper))
    df_lower = _quiet_run(_friction_case(lower))
    np.testing.assert_allclose(
        df_upper["Mass1_Force_friction_x_N"].to_numpy(),
        df_lower["Mass1_Force_friction_x_N"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_unknown_friction_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown friction_model"):
        _quiet_run(_friction_case("columb"))


def test_wall_force_reporting_keeps_legacy_front_and_adds_total(caplog: pytest.LogCaptureFixture) -> None:
    params = get_default_simulation_params()
    params.update(
        {
            "n_masses": 2,
            "masses": [1000.0, 1000.0],
            "x_init": [-0.01, -0.02],
            "y_init": [0.0, 0.0],
            "v0_init": 0.0,
            "fy": [1.0e6],
            "uy": [0.1],
            "k_wall": 1.0e6,
            "cr_wall": 0.0,
            "contact_model": "hooke",
            "friction_model": "none",
            "T_max": 1.0e-4,
            "h_init": 1.0e-4,
            "step": 1,
            "T_int": (0.0, 1.0e-4),
        }
    )

    with caplog.at_level("WARNING"):
        df = _quiet_run(params)

    np.testing.assert_allclose(df["Impact_Force_MN"], df["Impact_Force_front_MN"])
    assert float(df["Impact_Force_wall_total_MN"].max()) > float(df["Impact_Force_front_MN"].max())
    assert wall_force_deltas(df)["delta_Fpeak"] > 0.0
    assert "Secondary masses contacted the wall" in caplog.text


def test_picard_step_zero_uses_actual_deformation_rate() -> None:
    base = get_default_simulation_params()
    base.update(
        {
            "n_masses": 2,
            "masses": [1000.0, 1000.0],
            "x_init": [1.5, 2.5],
            "y_init": [0.0, 0.0],
            "v0_init": -1.0,
            "fy": [1.0e6],
            "uy": [0.1],
            "bw_a": 0.0,
            "bw_A": 1.0,
            "bw_beta": 0.5,
            "bw_gamma": 0.5,
            "bw_n": 2,
            "contact_model": "hooke",
            "friction_model": "none",
            "T_max": 1.0e-4,
            "h_init": 1.0e-4,
            "step": 1,
            "T_int": (0.0, 1.0e-4),
            "newton_tol": 1.0e-10,
            "picard_tol": 1.0e-10,
            "max_iter": 20,
            "picard_max_iters": 20,
        }
    )
    picard = dict(base, solver="picard")
    newton = dict(base, solver="newton")

    df_picard = _quiet_run(picard)
    df_newton = _quiet_run(newton)

    assert abs(float(df_picard.loc[1, "Spring1_Force_N"])) > 0.0
    assert df_picard.loc[1, "Spring1_Force_N"] == pytest.approx(
        df_newton.loc[1, "Spring1_Force_N"],
        rel=1e-3,
        abs=1e-3,
    )


def test_run_simulation_records_actual_final_time_metadata() -> None:
    df = _quiet_run({"T_max": 2.5e-4, "h_init": 1.0e-4})
    assert df.attrs["T_max_requested"] == pytest.approx(2.5e-4)
    assert df.attrs["t_final_actual"] == pytest.approx(3.0e-4)


def test_speed_scenarios_deep_copy_nested_params() -> None:
    scenarios = build_speed_scenarios({"nested": {"values": [1.0]}}, [10.0, 20.0])
    scenarios[0].params["nested"]["values"][0] = 99.0
    assert scenarios[1].params["nested"]["values"][0] == 1.0
