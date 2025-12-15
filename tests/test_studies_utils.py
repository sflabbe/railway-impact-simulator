import unittest

from railway_simulator.studies import _parse_path_tokens, get_by_path, set_by_path


class TestPathUtils(unittest.TestCase):
    def test_parse_tokens(self):
        self.assertEqual(_parse_path_tokens("k_wall"), [("k_wall", None)])
        self.assertEqual(_parse_path_tokens("fy[0]"), [("fy", 0)])
        self.assertEqual(_parse_path_tokens("train.fy[2]"), [("train", None), ("fy", 2)])

    def test_get_set_roundtrip(self):
        cfg = {"a": {"b": [10, 20, 30]}, "k_wall": 1.0}
        self.assertEqual(get_by_path(cfg, "a.b[1]"), 20)
        cfg2 = set_by_path(cfg, "a.b[1]", 99)
        self.assertEqual(get_by_path(cfg2, "a.b[1]"), 99)
        # original unchanged
        self.assertEqual(get_by_path(cfg, "a.b[1]"), 20)

    def test_invalid_token(self):
        with self.assertRaises(ValueError):
            _parse_path_tokens("bad-token")

if __name__ == "__main__":
    unittest.main()
