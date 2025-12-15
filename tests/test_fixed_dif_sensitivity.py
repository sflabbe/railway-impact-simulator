import unittest
import numpy as np
import pandas as pd

from railway_simulator.studies.strain_rate_sensitivity import run_fixed_dif_sensitivity


def fake_simulation(cfg):
    """
    Deterministic fake simulation.

    Peak force scales linearly with k_wall (or whichever stiffness is used).
    """
    t = np.linspace(0.0, 0.4, 5)
    k = float(cfg.get("k_wall", 1.0))
    # produce a simple triangular pulse with peak proportional to k
    pulse = np.array([0.0, 0.5, 1.0, 0.5, 0.0]) * (k * 1e-8)
    df = pd.DataFrame(
        {
            "Time_s": t,
            "Impact_Force_MN": pulse,
            "Penetration_mm": np.linspace(0, 100, 5),
            "Acceleration_g": np.linspace(0, 10, 5),
            "E_balance_error_J": np.linspace(0, -100, 5),
        }
    )
    df.attrs["n_lu"] = 7
    df.attrs["n_nonlinear_iters"] = 42
    return df


class TestFixedDIFSensitivity(unittest.TestCase):
    def test_scaling_and_output(self):
        cfg = {"k_wall": 6e7, "T_max": 0.4, "h_init": 1e-4}
        difs = [1.0, 1.1, 1.2]
        summary = run_fixed_dif_sensitivity(
            cfg, difs, k_path="k_wall", out_dir=None, simulate_func=fake_simulation
        )
        self.assertEqual(len(summary), 3)
        self.assertTrue(all(summary["k0_N_m"] == 6e7))
        self.assertAlmostEqual(float(summary.loc[0, "peak_force_MN"]), 6e7 * 1e-8, places=12)
        self.assertAlmostEqual(float(summary.loc[1, "k_scaled_N_m"]), 6e7 * 1.1, places=6)

if __name__ == "__main__":
    unittest.main()
