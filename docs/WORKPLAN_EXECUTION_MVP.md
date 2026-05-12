# Workplan execution MVP

Implemented in this package:

## 1. Domain layer

New package `railway_simulator.domain`:

- `Project`
- `StudyDefinition`
- `Scenario`
- `SimulationRun`
- `RunMetric`
- `SRSSettings`
- `SRSCurve`
- `VehicleConsist`, `MassPoint`, `CrushLink`, `Coupler`

## 2. SQLite project persistence

New package `railway_simulator.persistence`:

- `schema.sql`
- `ProjectDatabase`
- project/config/study/scenario/run/metric/SRS repositories

The database stores metadata and file paths. Time histories and SRS curves remain CSV artifacts.

## 3. Service layer

New package `railway_simulator.services`:

- `ProjectService`: create/open project folders with `project.sqlite`
- `SimulationService`: run a `Scenario`, save CSV, persist `SimulationRun`, extract scalar metrics

## 4. SRS service

New package `railway_simulator.spectrum`:

- `SpectrumService.compute_srs`
- `compute_and_store`
- `envelope`
- `ratio`

This wraps `hazard.sdof.compute_response_spectrum` so UI, CLI and scripts can share one numerical SRS implementation.

## 5. Contact patch hardening

New module `core/contact_state.py`:

- per mass `active` state
- per mass `v0_contact`
- explicit engine sign convention
- shared kinematics helper

`engine.py` now uses `ContactState` in the wall contact update path.

`ContactModels.compute_force` now raises `ValueError` for unknown model names instead of silently falling back to `anagnostopoulos`.

## 6. Full train study runner

New module `studies/full_train.py`:

- `FullTrainStudySpec`
- `FullTrainStudyRunner`
- TRAXX locomotive solo vs full freight proxy scenario builder
- optional SRS computation after each successful run

Also added sample spec:

- `configs/studies/stempi_full_train.yml`

## 7. Documentation

Added:

- `docs/PROJECT_WORKFLOW.md`
- this execution summary

Extended:

- `docs/CONTACT_MODEL_VERIFICATION.md`

## 8. Tests

Added tests for:

- contact state behavior and unknown contact model errors
- SQLite project persistence
- SRS computation/envelope/ratio
- full train scenario construction

Validation run:

```bash
PYTHONPATH=src pytest -q -m 'not slow'
# 156 passed, 17 deselected, 1 warning

PYTHONPATH=src pytest -q -m slow
# 17 passed, 156 deselected

PYTHONPATH=src python -m compileall -q src tests scripts examples
# passed
```

## Not yet implemented

The Streamlit UI was not surgically rewired in this MVP. The new services and study runner are now available for the next PR, where the existing `Parametric Studies` tab can call `ProjectService`, `FullTrainStudyRunner` and `SpectrumService` instead of duplicating study logic in the UI.
