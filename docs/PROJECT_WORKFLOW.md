# Project workflow and persistence MVP

This repository now contains a project-oriented layer around the validated engine.
The engine remains the numerical source of truth; the new layer provides traceable
project, study, scenario, run, metric and SRS persistence.

## Layout

```text
project_root/
  project.sqlite
  configs/
  runs/
  studies/
  spectra/
  artifacts/
```

SQLite stores metadata, scalar metrics and paths. Time histories and spectrum
curves remain file artifacts so the database does not become a blob store.

## Core objects

- `Project`: reproducible workspace.
- `StudyDefinition`: parametric study definition.
- `Scenario`: one executable engine configuration.
- `SimulationRun`: result of one scenario.
- `SRSCurve`: persisted response spectrum curve.

## Contact patch traceability

The package runner writes `contact_patch_version = contact_state_per_mass_v1` into
scenario metadata. The actual contact state logic lives in
`railway_simulator.core.contact_state.ContactState`.

## SRS comparison

Use `railway_simulator.spectrum.SpectrumService` to compute `Feq(Tn)` curves from
saved time histories. The service delegates to `hazard.sdof.compute_response_spectrum`
so UI, scripts and study runners share one implementation.


## Persistent CLI workflow

The project database can now be driven from the shell as well as from Streamlit.
See `docs/WORKPLAN_EXECUTION_CLI_WORKFLOW.md` for the full command sequence.

Minimal reproducible flow:

```bash
railway-sim project create --name stempi --root projects/stempi
railway-sim study run-full-train --db projects/stempi/project.sqlite --spec configs/studies/stempi_full_train.yml
railway-sim srs compare --db projects/stempi/project.sqlite --study-id <study_id> --output-dir results/stempi_srs_compare --zeta 0.05
```

## Streamlit report bundle workflow

The Project Workbench now includes a **Report bundle** tab. After a persisted
study has been run, the tab can generate the same LaTeX chapter bundle as the
CLI command:

```bash
railway-sim report build-chapter \
  --db projects/stempi/project.sqlite \
  --study-id <study_id> \
  --output-dir projects/stempi/reports/chapter_stempi
```

The UI stores the bundle under `project_root/reports/` by default and offers a
ZIP download containing the LaTeX source, bibliography, figures, tables and
metadata.
