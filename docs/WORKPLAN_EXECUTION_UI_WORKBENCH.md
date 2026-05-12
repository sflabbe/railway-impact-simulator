# Workplan execution — UI project workbench

This increment connects the new project/study/SRS architecture to the Streamlit
interface without replacing the legacy simulator and parametric tabs.

## Implemented

- Added a Streamlit **Project Workbench** tab.
- Added `railway_simulator.ui.app` as the preferred Streamlit entry point.
- Updated the CLI launcher to run `railway_simulator/ui/app.py`.
- Kept `railway_simulator.core.app` as a compatibility wrapper/implementation.
- Added UI workflow to create/open a project SQLite database.
- Added UI workflow to run a persisted full-train study from the current sidebar
  configuration.
- Added SRS comparison from persisted `srs_curves` records.
- Added SRS overlay, envelope, and `Zug_full / Lok_solo` ratio plots.
- Added CSV export for selected SRS curves.
- Added repository read models for joined run/scenario and curve/scenario rows.
- Made `railway_simulator.ui.__init__` lazy so importing lightweight UI helpers
  does not require the optional `streamlit` dependency.

## User workflow

```text
railway-sim ui
  -> Project Workbench
     -> Create / initialize project
     -> Launch full-train study
     -> SRS comparison
```

The persisted project folder contains:

```text
project.sqlite
runs/*.csv
spectra/*.csv
configs/
studies/
artifacts/
```

SQLite stores metadata, hashes, study definitions, run records, scalar metrics,
and SRS curve references. Time histories and SRS arrays remain CSV artifacts.

## Verification commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q -m 'not slow'
PYTHONPATH=src pytest -q -m slow
PYTHONPATH=src python -m compileall -q src tests scripts examples
```

Current result in the sandbox:

```text
160 passed, 17 deselected, 1 warning
17 passed, 160 deselected
compileall ok
```

`ruff` was not available in the sandbox environment.

## Notes

The full-train launcher is synchronous in Streamlit. This is intentional for the
MVP. Heavy sweeps should later be moved to a CLI/background worker or a job queue,
but the current implementation already persists completed runs and can compare
SRS curves without re-running simulations.
