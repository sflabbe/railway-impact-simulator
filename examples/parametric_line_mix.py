from pathlib import Path
from copy import deepcopy

import pandas as pd

from railway_simulator.core.parametric import ScenarioDefinition, run_parametric_envelope
from railway_simulator.core.engine import SimulationParams, run_simulation  # if you want to build params here

# ----------------------------------------------------------------------
# 1. Define / load base parameter sets for each train type
#    (here just placeholders – you will plug in your real configs)
# ----------------------------------------------------------------------

# Example: you might already have YAML files per train type and nominal speed
# and a small loader that returns a dict, then SimulationParams(**dict)
# For illustration we assume you already have fully prepared dicts:
base_tgv_params: dict = {...}     # TGV-like geometry, Bouc–Wen, contact, etc.
base_ic_params: dict = {...}      # Intercity / passenger
base_cargo_params: dict = {...}   # Freight

base_by_type = {
    "TGV": base_tgv_params,
    "IC": base_ic_params,
    "Cargo": base_cargo_params,
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

scenarios: list[ScenarioDefinition] = []

for row in traffic_data.itertuples(index=False):
    base = deepcopy(base_by_type[row.train_type])
    # Override impact velocity [m/s], assuming your params dict uses v0_init
    base["v0_init"] = -row.speed_kmh / 3.6

    scen = ScenarioDefinition(
        name=row.name,
        params=base,
        weight=row.share,
        meta={
            "train_type": row.train_type,
            "speed_kmh": row.speed_kmh,
        },
    )
    scenarios.append(scen)

# ----------------------------------------------------------------------
# 3. Run study and compute envelope + weighted mean
# ----------------------------------------------------------------------

envelope_df, summary_df, per_scenario = run_parametric_envelope(
    scenarios,
    quantity="Impact_Force_MN",
)

# Save to disk or plot with matplotlib/plotly
output_dir = Path("results/line_mix_example")
output_dir.mkdir(parents=True, exist_ok=True)

envelope_df.to_csv(output_dir / "envelope_force.csv", index=False)
summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)

for name, df in per_scenario.items():
    df.to_csv(output_dir / f"{name}_history.csv", index=False)
