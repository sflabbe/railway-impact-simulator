import unittest
import numpy as np
import pandas as pd

from railway_simulator.studies.convergence import run_convergence_study


def fake_simulation(cfg):
    dt = float(cfg.get("h_init", 1e-4))
    t = np.linspace(0.0, 0.4, 5)
    # Make peak depend weakly on dt (smaller dt -> slightly higher accuracy -> slightly different peak)
    peak = 1.0 + 0.1 * (1e-4 / dt)
    pulse = np.array([0.0, 0.5, 1.0, 0.5, 0.0]) * peak
    df = pd.DataFrame(
        {
            "Time_s": t,
            "Impact_Force_MN": pulse,
            "Penetration_mm": np.linspace(0, 100, 5),
            "Acceleration_g": np.linspace(0, 10, 5),
            "E_balance_error_J": np.linspace(0, 0, 5),
        }
    )
    df.attrs["n_lu"] = 3
    df.attrs["n_nonlinear_iters"] = 9
    return df


class TestConvergenceStudy(unittest.TestCase):
    def test_convergence_summary(self):
        cfg = {"T_max": 0.4, "h_init": 2e-4}
        dt_values = [2e-4, 1e-4]
        summary = run_convergence_study(cfg, dt_values, simulate_func=fake_simulation)
        self.assertEqual(len(summary), 2)
        # Sorted descending dt in implementation
        self.assertAlmostEqual(float(summary.loc[0, "dt_s"]), 2e-4)
        self.assertAlmostEqual(float(summary.loc[1, "dt_s"]), 1e-4)
        # relative change exists for second row
        self.assertTrue(np.isnan(summary.loc[0, "relative_change_peak_pct"]) or summary.loc[0, "relative_change_peak_pct"] is None)
        self.assertIsNotNone(summary.loc[1, "relative_change_peak_pct"])

if __name__ == "__main__":
    unittest.main()
