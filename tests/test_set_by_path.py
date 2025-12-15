from railway_simulator.studies import set_by_path, get_by_path

def test_set_get_scalar():
    cfg = {"k_wall": 10.0}
    cfg2 = set_by_path(cfg, "k_wall", 12.0)
    assert cfg["k_wall"] == 10.0
    assert get_by_path(cfg2, "k_wall") == 12.0

def test_set_get_indexed_list():
    cfg = {"fy": [1.0, 2.0, 3.0]}
    cfg2 = set_by_path(cfg, "fy[1]", 9.0)
    assert get_by_path(cfg2, "fy[1]") == 9.0
