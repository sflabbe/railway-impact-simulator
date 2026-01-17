import unittest
import numpy as np
import pandas as pd

from railway_simulator.studies.numerics_sensitivity import run_numerics_sensitivity


def fake_simulation(cfg):
    dt = float(cfg.get("h_init", 1e-4))
    alpha = float(cfg.get("alpha_hht", -0.15))
    tol = float(cfg.get("newton_tol", 1e-5))
    t = np.linspace(0.0, 0.4, 5)
    # Peak depends on all three in a predictable way
    peak = 2.0 + 0.05 * (1e-4 / dt) + 0.1 * (-alpha) + 0.0 * tol
    pulse = np.array([0.0, 0.5, 1.0, 0.5, 0.0]) * peak
    df = pd.DataFrame({"Time_s": t, "Impact_Force_MN": pulse, "Penetration_mm": np.zeros_like(t)})
    df.attrs["n_lu"] = 1
    df.attrs["n_nonlinear_iters"] = 2
    return df


class TestNumericsSensitivity(unittest.TestCase):
    def test_cartesian_sweep(self):
        cfg = {"T_max": 0.4}
        summary = run_numerics_sensitivity(
            cfg,
            dt_values=[2e-4, 1e-4],
            alpha_values=[-0.1, -0.15],
            tol_values=[1e-4],
            simulate_func=fake_simulation,
        )
        self.assertEqual(len(summary), 2 * 2 * 1)
        baseline = summary[
            (summary["dt_requested"] == 1e-4)
            & (summary["alpha_hht"] == -0.15)
            & (summary["newton_tol"] == 1e-4)
        ].iloc[0]
        rel = baseline["peak_force_rel_to_baseline_pct"]
        self.assertTrue(rel == 0.0 or np.isclose(rel, 0.0))

if __name__ == "__main__":
    unittest.main()
