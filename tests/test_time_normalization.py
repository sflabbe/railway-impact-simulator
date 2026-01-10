from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from railway_simulator.core.engine import (
    _coerce_scalar_types_for_simulation,
    get_default_simulation_params,
    normalize_simulation_params,
)


def test_tmax_override_without_tint() -> None:
    defaults = get_default_simulation_params()
    raw = dict(defaults)
    raw["T_max"] = 3.0

    flags = {"T_int": False, "T_max": True, "step": False, "h_init": False}
    coerced = _coerce_scalar_types_for_simulation(raw)
    normalized = normalize_simulation_params(coerced, defaults, flags)

    assert normalized["T_max"] == 3.0
    assert normalized["T_int"] == (0.0, 3.0)
    assert normalized["step"] == int(np.ceil(3.0 / defaults["h_init"]))
