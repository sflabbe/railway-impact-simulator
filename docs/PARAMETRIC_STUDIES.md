# Parametric studies

This document describes the generic parametric study workflow used by `railway-impact-simulator`.
It covers the YAML study format, the CLI, the Project Workbench UI, SQLite persistence, and export of data and plots.

The workflow is intentionally split into layers:

1. **YAML study definition**: declares the base configuration and the sweep dimensions.
2. **Pure grid builder**: expands the YAML into deterministic scenarios without running the solver.
3. **Runner**: applies each scenario to the base configuration and executes the existing simulation engine.
4. **Persistence**: optionally stores the project, study, scenarios, runs and result CSV paths in SQLite.
5. **UI/export**: shows summaries, warnings and basic plots without re-running saved studies.

The generic study type is called `parametric_grid`.

---

## 1. Minimal YAML study

Example file:

```text
configs/studies/impact_parametric_mini.yml
```

A typical structure is:

```yaml
study:
  name: impact_parametric_mini
  type: parametric_grid
  description: Minimal parametric impact study for smoke tests and UI previews.

base:
  config: ../en15227/traxx_en15227.yml

dimensions:
  - name: impact_velocity_mps
    kind: parameter
    path: v0_init
    values: [-5.55556, -11.11111]

  - name: contact_law
    kind: parameter
    path: contact_model
    values: [hooke, lankarani-nikravesh]

outputs:
  quantities:
    - Impact_Force_MN
    - Penetration_mm
    - Acceleration_g

srs:
  enabled: false
```

The `dimensions` block is the central part. Each dimension has:

| Field | Meaning |
|---|---|
| `name` | Human-readable dimension name. It also becomes a metadata column in summaries. |
| `kind` | Dimension category. For ordinary config edits use `parameter`. |
| `path` | Path inside the normalized simulation config. This path must exist. |
| `values` | Values used to build the cartesian scenario grid. |

For the example above, the grid has `2 x 2 = 4` scenarios.

Important unit convention: values are applied exactly as written to the config path. If a config uses `v0_init` in m/s with sign, the YAML must also use m/s with sign. Do not label such values as km/h unless an explicit conversion dimension has been implemented.

---

## 2. CLI usage

### Preview only

Use dry-run mode to inspect the generated scenarios without running the solver:

```bash
uv run railway-sim study run-grid \
  --spec configs/studies/impact_parametric_mini.yml \
  --dry-run
```

Limit the preview to the first scenario:

```bash
uv run railway-sim study run-grid \
  --spec configs/studies/impact_parametric_mini.yml \
  --dry-run \
  --limit 1
```

Export the preview table:

```bash
uv run railway-sim study run-grid \
  --spec configs/studies/impact_parametric_mini.yml \
  --dry-run \
  --out build/parametric_preview.csv
```

### Run in memory and export a summary

```bash
uv run railway-sim study run-grid \
  --spec configs/studies/impact_parametric_mini.yml \
  --limit 1 \
  --out build/parametric_summary.csv
```

This executes the first scenario and writes a summary table. It does not persist a project database unless `--db` is passed.

### Run and persist to SQLite

```bash
uv run railway-sim study run-grid \
  --spec configs/studies/impact_parametric_mini.yml \
  --db projects/impact_workbench/project.sqlite \
  --limit 1 \
  --out build/parametric_persistent_summary.csv
```

With `--db`, the command stores:

| SQLite table | Content |
|---|---|
| `projects` | Project workspace, default `impact_workbench`. |
| `config_snapshots` | Normalized base configuration used by the run. |
| `studies` | Study definition with `study_type = "parametric_grid"`. |
| `scenarios` | One row per generated scenario, including metadata and applied config. |
| `runs` | One row per executed scenario, including status and result CSV path. |
| `run_metrics` | Scalar metrics extracted from the time history. |

`--limit` works in persistent and non-persistent mode. Use it for smoke tests before launching a larger grid.

`--strict` stops at the first failed scenario. Without `--strict`, failed scenarios are reported with `status=failed` and the run continues.

---

## 3. Project Workbench UI

Launch the UI with the optional UI dependencies:

```bash
uv run --extra ui streamlit run src/railway_simulator/core/app.py
```

Open **Project Workbench** and use the **Parametric studies** tab.

The **Custom parametric grid** section supports:

1. Loading a YAML spec.
2. Previewing scenarios without running the solver.
3. Running a limited or full grid.
4. Persisting results to SQLite.
5. Displaying summary tables.
6. Displaying solver warnings prominently.
7. Plotting peak impact force by scenario when `Impact_Force_MN` is available.
8. Downloading CSV and HTML exports.

The default UI values are conservative:

| UI input | Default |
|---|---|
| Spec path | `configs/studies/impact_parametric_mini.yml` |
| DB path | `projects/impact_workbench/project.sqlite` |
| Project name | `impact_workbench` |
| Limit scenarios | `1` |
| Strict mode | `false` |
| Persist results | `true` |

Use `Limit scenarios = 0` only when you intentionally want to run all scenarios in the YAML grid.

---

## 4. Outputs and exports

### Summary table

The main output is a summary table with stable front columns:

| Column | Meaning |
|---|---|
| `scenario_index` | Deterministic zero-based scenario index. |
| `scenario_label` | Deterministic label assembled from dimension values. |
| `status` | `ok`, `failed`, or other runner status. |
| metadata columns | One column per scenario dimension, for example `impact_velocity_mps` and `contact_law`. |
| `peak_Impact_Force_MN` | Peak value extracted from `Impact_Force_MN`, if available. |
| `peak_Penetration_mm` | Peak value extracted from `Penetration_mm`, if available. |
| `peak_Acceleration_g` | Peak value extracted from `Acceleration_g`, if available. |
| `t_end` | End time of the time history, if available. |
| `n_steps` | Number of time increments, if available. |
| `warnings` | Captured solver warnings or structured warnings returned by a custom runner. |
| `error` | Error message for failed scenarios. |

The UI download button **Download summary CSV** exports this table.

### Peak-force plot

If `peak_Impact_Force_MN` is available, the UI creates a bar chart of peak force by scenario label.

The plot can be exported as:

| Export | File name | Notes |
|---|---|---|
| Plot data CSV | `parametric_grid_peak_force_data.csv` | Small table with scenario label and peak force. |
| Plot HTML | `parametric_grid_peak_force.html` | Standalone Plotly HTML suitable for browser viewing and report checks. |

The HTML export is meant as a review artifact, not as the final thesis figure. For a final chapter figure, prefer regenerating publication-quality plots from the exported CSV or from the persisted run CSV files.

### Time-history CSV files

For persistent runs, each scenario time history is written as CSV under the project run folder, typically:

```text
projects/impact_workbench/runs/<scenario_id>.csv
```

The SQLite `runs.result_csv_path` column points to this file. The database stores metadata and file paths; large time histories remain file artifacts.

### Saved studies

Saved `parametric_grid` studies can be selected again in the Project Workbench. The UI reconstructs the summary from:

1. Scenario metadata in SQLite.
2. Run status and result CSV paths in SQLite.
3. Persisted result CSV files.
4. Solver warnings stored in scenario metadata when available.

This means a saved study can be inspected again without re-running the solver.

---

## 5. Solver warnings

Solver warnings must not be hidden by an `ok` status. The current workflow exposes warnings in the summary table and in a dedicated warning table in the UI.

Implementation detail:

- In-memory/custom runners may return warnings directly through result dictionaries or DataFrame attributes.
- Persistent runs capture warning logs emitted by `railway_simulator.core` during each scenario.
- Warnings are stored in scenario metadata as `solver_warnings` when available.

There is currently no separate `solver_warnings` SQL table. If warning classification becomes important for thesis-grade QA, add a dedicated table or artifact registry later.

---

## 6. Recommended workflow for thesis studies

For development and debugging:

```bash
uv run railway-sim study run-grid \
  --spec configs/studies/impact_parametric_mini.yml \
  --dry-run
```

Then run one scenario:

```bash
uv run railway-sim study run-grid \
  --spec configs/studies/impact_parametric_mini.yml \
  --db projects/impact_workbench/project.sqlite \
  --limit 1 \
  --out build/smoke_summary.csv
```

Only after the mini run is clean, increase `--limit` or remove it.

For chapter production, use the same pattern with a larger academic spec, for example a future:

```text
configs/studies/trackside_clearance_demand.yml
```

The rule is: tables and figures should come from exported CSV files or persisted run artifacts, not from manually copied values.

---

## 7. Current limitations

- The generic workflow does not yet create a ZIP bundle for parametric grid studies.
- The UI exports a basic peak-force chart, not publication-ready thesis figures.
- Solver warnings are stored in scenario metadata, not in a dedicated warnings table.
- Heavy grids still run synchronously. Use the CLI for larger studies.
- The future lateral yaw/contact-progressive engine is not part of this workflow yet.
