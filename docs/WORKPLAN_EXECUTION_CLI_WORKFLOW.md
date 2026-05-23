# Workplan execution — persistent CLI workflow

This increment adds a shell-first workflow equivalent to the Streamlit Project Workbench.
The goal is reproducible thesis/CI execution: create a project, run the full-train
parametric study, and export/compare persisted Shock Response Spectrum (SRS) curves
without opening the UI.

## Added commands

```bash
railway-sim project create --name impact_workbench --root projects/impact_workbench
railway-sim project list --db projects/impact_workbench/project.sqlite
```

```bash
railway-sim study run-full-train \
  --db projects/impact_workbench/project.sqlite \
  --base-config configs/traxx_freight.yml \
  --name train_consist_comparison
```

```bash
railway-sim study list --db projects/impact_workbench/project.sqlite
railway-sim study runs --db projects/impact_workbench/project.sqlite --study-id <study_id>
```

```bash
railway-sim srs list --db projects/impact_workbench/project.sqlite --study-id <study_id>
railway-sim srs export \
  --db projects/impact_workbench/project.sqlite \
  --study-id <study_id> \
  --output results/consist_srs_long.csv
```

```bash
railway-sim srs compare \
  --db projects/impact_workbench/project.sqlite \
  --study-id <study_id> \
  --output-dir results/consist_srs_comparison \
  --zeta 0.05
```

## Dry-run before expensive studies

Use dry-run to verify the scenario count and resolved study parameters without
writing a study or running the solver:

```bash
railway-sim study run-full-train \
  --db projects/impact_workbench/project.sqlite \
  --base-config configs/traxx_freight.yml \
  --name train_consist_comparison \
  --dry-run
```

CLI overrides can narrow the grid:

```bash
railway-sim study run-full-train \
  --db projects/impact_workbench/project.sqlite \
  --base-config configs/traxx_freight.yml \
  --name train_consist_comparison \
  --speeds 10,20,30 \
  --modes lok_solo,zug_full \
  --contact-models anagnostopoulos \
  --mu-values 0.30 \
  --zeta-values 0.05 \
  --tn-grid-ms 10,30,100,300,1000 \
  --dry-run
```

## Outputs

The persistent study command stores:

- run histories in `<project>/runs/*.csv`
- SRS curves in `<project>/spectra/*.csv`
- study/scenario/run/SRS metadata in `<project>/project.sqlite`

The SRS comparison command writes:

- `srs_curves_long.csv`
- `srs_envelope.csv`
- `srs_overlay.png`
- `srs_envelope.png`
- `srs_full_train_vs_lok_ratio.csv` when matching `lok_solo` and `zug_full` curves exist
- `srs_full_train_vs_lok_ratio.png` when ratio curves exist

## Design decision

The CLI does not duplicate the Streamlit logic. It uses the same domain,
persistence, simulation and spectrum services introduced for the Project Workbench.
This keeps UI exploration and terminal production runs on the same data model.

## Validation performed for this increment

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q -m 'not slow'
# 162 passed, 17 deselected

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q -m slow
# 17 passed, 162 deselected

PYTHONPATH=src python -m compileall -q src tests scripts examples
# ok
```
