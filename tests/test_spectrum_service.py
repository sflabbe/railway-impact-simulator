from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from railway_simulator.domain.spectrum import SRSSettings
from railway_simulator.spectrum.service import SpectrumService


def test_spectrum_service_compute_envelope_and_ratio() -> None:
    t = np.linspace(0.0, 0.2, 200)
    df1 = pd.DataFrame({"Time_s": t, "Impact_Force_MN": np.sin(np.pi * t / 0.2).clip(min=0.0)})
    df2 = pd.DataFrame({"Time_s": t, "Impact_Force_MN": 2.0 * df1["Impact_Force_MN"]})
    service = SpectrumService()
    settings = SRSSettings(Tn_grid_ms=(20.0, 100.0, 500.0), zeta=0.05)

    c1 = service.compute_srs(df1, settings)
    c2 = service.compute_srs(df2, settings)

    assert list(c1["Tn_ms"]) == [20.0, 100.0, 500.0]
    assert "Feq_MN" in c1.columns

    env = service.envelope([c1, c2])
    assert np.all(env["Feq_MN_envelope"].to_numpy() >= c1["Feq_MN"].to_numpy())

    ratio = service.ratio(c2, c1)
    assert np.allclose(ratio["Feq_MN_ratio"].to_numpy(), 2.0, rtol=1e-9, atol=1e-9)


def test_spectrum_envelope_interpolates_misaligned_overlapping_periods() -> None:
    c1 = pd.DataFrame({"Tn_ms": [100.0, 200.0, 300.0], "Feq_MN": [1.0, 2.0, 1.0]})
    c2 = pd.DataFrame({"Tn_ms": [200.0, 300.0, 400.0], "Feq_MN": [3.0, 1.0, 1.0]})

    env = SpectrumService.envelope([c1, c2])

    assert env["Tn_ms"].tolist() == [200.0, 300.0]
    assert env["Feq_MN_envelope"].tolist() == pytest.approx([3.0, 1.0])


def test_spectrum_envelope_partial_overlap_is_explicit() -> None:
    c1 = pd.DataFrame({"Tn_ms": [100.0, 200.0, 300.0], "Feq_MN": [1.0, 2.0, 1.0]})
    c2 = pd.DataFrame({"Tn_ms": [200.0, 300.0, 400.0], "Feq_MN": [3.0, 1.0, 1.0]})

    with pytest.warns(RuntimeWarning, match="partial period overlap"):
        env = SpectrumService.envelope([c1, c2], allow_partial_overlap=True)

    assert env["Tn_ms"].tolist() == [100.0, 200.0, 300.0, 400.0]
    assert env["n_contributing_curves"].tolist() == [1, 2, 2, 1]


def test_spectrum_envelope_rejects_non_overlapping_period_grids() -> None:
    c1 = pd.DataFrame({"Tn_ms": [100.0, 200.0], "Feq_MN": [1.0, 2.0]})
    c2 = pd.DataFrame({"Tn_ms": [300.0, 400.0], "Feq_MN": [3.0, 4.0]})

    with pytest.raises(ValueError, match="period grids do not overlap"):
        SpectrumService.envelope([c1, c2])


def test_spectrum_ratio_same_grid_is_elementwise() -> None:
    numerator = pd.DataFrame({"Tn_ms": [100.0, 200.0, 300.0], "Feq_MN": [2.0, 6.0, 12.0]})
    denominator = pd.DataFrame({"Tn_ms": [100.0, 200.0, 300.0], "Feq_MN": [1.0, 2.0, 3.0]})

    ratio = SpectrumService.ratio(numerator, denominator)

    assert ratio["Tn_ms"].tolist() == [100.0, 200.0, 300.0]
    assert ratio["Feq_MN_ratio"].tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_spectrum_ratio_partial_overlap_uses_only_valid_numerator_periods() -> None:
    numerator = pd.DataFrame({"Tn_ms": [100.0, 200.0, 300.0], "Feq_MN": [9.0, 6.0, 12.0]})
    denominator = pd.DataFrame({"Tn_ms": [200.0, 300.0, 400.0], "Feq_MN": [2.0, 3.0, 4.0]})

    ratio = SpectrumService.ratio(numerator, denominator)

    assert ratio["Tn_ms"].tolist() == [200.0, 300.0]
    assert ratio["Feq_MN_ratio"].tolist() == pytest.approx([3.0, 4.0])


def test_spectrum_ratio_rejects_empty_effective_overlap() -> None:
    numerator = pd.DataFrame({"Tn_ms": [100.0, 400.0], "Feq_MN": [2.0, 8.0]})
    denominator = pd.DataFrame({"Tn_ms": [200.0, 300.0], "Feq_MN": [4.0, 6.0]})

    with pytest.raises(ValueError, match="no numerator periods lie inside the common overlap"):
        SpectrumService.ratio(numerator, denominator)


def test_spectrum_ratio_rejects_non_overlapping_period_grids() -> None:
    numerator = pd.DataFrame({"Tn_ms": [100.0, 150.0], "Feq_MN": [2.0, 3.0]})
    denominator = pd.DataFrame({"Tn_ms": [300.0, 400.0], "Feq_MN": [4.0, 5.0]})

    with pytest.raises(ValueError, match="period grids do not overlap"):
        SpectrumService.ratio(numerator, denominator)


def test_spectrum_ratio_does_not_use_endpoint_clamping_outside_overlap() -> None:
    numerator = pd.DataFrame({"Tn_ms": [100.0, 200.0, 300.0], "Feq_MN": [999.0, 6.0, 12.0]})
    denominator = pd.DataFrame({"Tn_ms": [200.0, 300.0, 400.0], "Feq_MN": [0.5, 3.0, 4.0]})

    ratio = SpectrumService.ratio(numerator, denominator)

    assert 100.0 not in ratio["Tn_ms"].tolist()
    assert ratio["Feq_MN_ratio"].tolist() == pytest.approx([12.0, 4.0])
