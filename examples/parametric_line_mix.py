"""
Example: line-specific traffic mix (TGV / IC / Cargo) using the Python API.

This script shows how to:

- Define a traffic mix in a pandas.DataFrame
- Build `ScenarioDefinition` objects for each speed / train type
- Run a parametric study and compute an envelope of `Impact_Force_MN`
- Save envelope and per-scenario histories to CSV

It is intended as a more "programmatic" counterpart to the CLI command:

    railway-sim parametric \
      --base-config configs/ice1_80kmh.yml \
      --speeds "320:0.2,200:0.4,120:0.4" \
      --quantity Impact_Force_MN \
      --output-dir results_parametric/track_mix \
      --prefix track_mix
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from railway_simulator.core.parametric import ScenarioDefinition, run_parametric_envelope
# `run_simulation` / `SimulationParams` are not strictly needed here, but you can
# use them if you prefer to build strongly-typed parameters instead of dicts.
# from railway_simulator.core.engine import SimulationParams, run_simulation


# ----------------------------------------------------------------------
# 1. Load a base parameter set
# ----------------------------------------------------------------------

# For a minimal, runnable example we simply re-use the ICE-1 config as "base train".
# In a real application you would likely have one YAML per train family, e.g.:
#   configs/tgv_base.yml, configs/ic_base.yml, configs/cargo_base.yml, ...
BASE_CONFIG_PATH = Path("configs/ice1_80kmh.yml")


def load_base_params(path: Path) -> dict:
    """Load a YAML config into a plain dict."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} did not yield a mapping.")
    return data


# In this simple example we re-use the same base parameters for all train types.
# Physically you would adapt masses, spring layout, Bouc–Wen parameters, etc.
base_params = load_base_params(BASE_CONFIG_PATH)

base_by_type: dict[str, dict] = {
    "TGV": base_params,
    "IC": base_params,
    "Cargo": base_params,
}

# Example traffic mix for a given line section
traffic_data = pd.DataFrame(
    {
        "name": ["TGV_320", "IC_200", "Cargo_120"],
        "train_type": ["TGV", "IC", "Cargo"],
        "speed_kmh": [320.0, 200.0, 120.0],
        "share": [0.20, 0.40, 0.40],  # 20 % / 40 % / 40 %
    }
)


# ----------------------------------------------------------------------
# 2. Build ScenarioDefinition objects
# ----------------------------------------------------------------------

def build_scenarios(df: pd.DataFrame) -> list[ScenarioDefinition]:
    """Convert the traffic mix table into a list of ScenarioDefinition objects."""
    scenarios: list[ScenarioDefinition] = []

    for row in df.itertuples(index=False):
        base = deepcopy(base_by_type[row.train_type])

        # Override impact velocity [m/s], assuming your params dict uses v0_init
        # Sign convention: negative velocity = towards the rigid wall (as in the ICE-1 configs)
        base["v0_init"] = -row.speed_kmh / 3.6

        scen = ScenarioDefinition(
            name=row.name,
            params=base,
            weight=float(row.share),
            meta={
                "train_type": row.train_type,
                "speed_kmh": float(row.speed_kmh),
            },
        )
        scenarios.append(scen)

    return scenarios


# ----------------------------------------------------------------------
# 3. Run study and compute envelope + weighted mean
# ----------------------------------------------------------------------

def main() -> None:
    scenarios = build_scenarios(traffic_data)

    envelope_df, summary_df, per_scenario = run_parametric_envelope(
        scenarios,
        quantity="Impact_Force_MN",
    )

    # Save to disk
    output_dir = Path("results/line_mix_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    envelope_df.to_csv(output_dir / "envelope_force.csv", index=False)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)

    for name, df in per_scenario.items():
        df.to_csv(output_dir / f"{name}_history.csv", index=False)

    print(f"Written envelope and summaries to: {output_dir}")


if __name__ == "__main__":
    main()
