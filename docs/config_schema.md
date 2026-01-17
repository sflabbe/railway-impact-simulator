# Configuration Schema (YAML)

This document describes the **EN15227-inspired** configuration structure used by
`railway-sim`. It is **not a compliance claim**; presets are data-only
approximations intended for convenience.

## Top-level structure

```yaml
units: "SI"           # optional, defaults to SI
collision:
  partner:
    type: rigid_wall  # default
  interface:
    type: linear
    k_N_per_m: 4.5e7
  scenario: null      # optional preset label
```

All existing train/material/solver fields remain valid at the top level (e.g.,
`n_masses`, `masses`, `fy`, `uy`, `v0_init`, `k_wall`, `contact_model`, ...). If
`collision` is omitted, the loader migrates legacy `k_wall` / `contact_model`
information into a `collision` section at runtime and preserves the original
keys under `legacy:`.

## Collision partner

```yaml
collision:
  partner:
    type: rigid_wall                # default
```

```yaml
collision:
  partner:
    type: reference_mass_1d
    mass_kg: 80000
    dofs: [x]
```

```yaml
collision:
  partner:
    type: en15227_preset
    preset_id: EN15227_2011_C2_regional_129t
```

## Interface laws

```yaml
collision:
  interface:
    type: linear
    k_N_per_m: 4.5e7
    gap_m: 0.0
```

```yaml
collision:
  interface:
    type: piecewise_linear
    points:
      - [0.0, 0.0]
      - [0.245, 1.05e6]
      - [0.475, 1.35e6]
```

```yaml
collision:
  interface:
    type: plateau
    ramp_to_force_N: 5.0e5
    ramp_disp_m: 0.05
    plateau_force_N: 1.05e6
    plateau_disp_m: 0.245
    final_force_N: 1.35e6
    final_disp_m: 0.475
```

```yaml
collision:
  interface:
    type: tabulated_curve
    csv_path: curves/coupler_curve.csv
```

## Presets (EN15227-inspired, data-only)

Presets live in `data/en15227/`:

* `partners.yaml`
* `interfaces.yaml`
* `scenarios.yaml`

Use them either directly in `collision.partner`/`collision.interface` or via
`collision.scenario`.

## Migration

Use the migration script to write the new format and keep the original keys in
`legacy`:

```bash
python scripts/migrate_yaml_configs.py configs/ice1_aluminum.yml --output-dir migrated/
```

## Example (EN15227-inspired preset)

```yaml
units: "SI"
collision:
  partner:
    type: en15227_preset
    preset_id: EN15227_2011_C2_regional_129t
  interface:
    type: en15227_preset
    preset_id: EN15227_2011_C2_central_coupler
  scenario: null

# ... train, solver, material fields remain unchanged below ...
n_masses: 7
masses: [4000, 10000, 4000, 4000, 4000, 10000, 4000]
fy: [8.0e6, 8.0e6, 8.0e6, 8.0e6, 8.0e6, 8.0e6]
uy: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
```
