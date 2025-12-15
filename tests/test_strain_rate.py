from railway_simulator.materials.strain_rate import concrete_dif_compression, concrete_dif_tension

def test_dif_reasonable_ranges():
    assert 1.0 <= concrete_dif_compression(0.0) <= 1.001
    assert 1.05 <= concrete_dif_compression(1.0) <= 1.3
    assert 1.2 <= concrete_dif_compression(10.0) <= 1.6

    assert 1.0 <= concrete_dif_tension(0.0) <= 1.001
    assert 1.1 <= concrete_dif_tension(1.0) <= 1.6
