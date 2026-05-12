from __future__ import annotations

import numpy as np
import pandas as pd

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
