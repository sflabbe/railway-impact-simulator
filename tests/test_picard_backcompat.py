from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, "src")

import numpy as np
import yaml

from railway_simulator.core.engine import run_simulation


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _apply_short_run(params: dict) -> dict:
    T_max = 0.01
    h_init = 1.0e-4
    params.update(
        {
            "T_max": T_max,
            "h_init": h_init,
            "step": int(np.ceil(T_max / h_init)),
            "T_int": (0.0, T_max),
        }
    )
    return params


def test_max_iter_without_picard_fields_is_ok() -> None:
    params = _load_config(Path("configs/ice1_aluminum.yml"))
    params = _apply_short_run(params)

    df = run_simulation(params)

    assert "Time_s" in df.columns
    assert len(df) > 1


def test_picard_config_with_explicit_limits_is_ok() -> None:
    params = _load_config(Path("configs/ice1_full_dissertation.yml"))
    params = _apply_short_run(params)

    df = run_simulation(params)

    assert "Time_s" in df.columns
    assert len(df) > 1
