import unittest

from railway_simulator.materials.strain_rate import (
    concrete_dif_compression,
    concrete_dif_tension,
    steel_dif_cowper_symonds,
)


class TestStrainRateModels(unittest.TestCase):
    def test_concrete_dif_reasonable(self):
        # These checks are intentionally broad: we only want to catch "exploding DIF" bugs.
        self.assertAlmostEqual(concrete_dif_compression(0.0), 1.0)
        self.assertTrue(1.05 <= concrete_dif_compression(1.0) <= 1.30)
        self.assertTrue(1.05 <= concrete_dif_compression(10.0) <= 1.50)

        self.assertAlmostEqual(concrete_dif_tension(0.0), 1.0)
        self.assertTrue(1.10 <= concrete_dif_tension(1.0) <= 1.50)
        self.assertTrue(1.10 <= concrete_dif_tension(10.0) <= 1.80)

    def test_steel_cowper_symonds(self):
        self.assertAlmostEqual(steel_dif_cowper_symonds(0.0), 1.0)
        self.assertTrue(steel_dif_cowper_symonds(10.0) > steel_dif_cowper_symonds(1.0))


if __name__ == "__main__":
    unittest.main()
