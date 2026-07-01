# Patch Report

Date: 2026-07-02

## Scope

Final cleanup before merge on `bugfix/final-cleanup-before-merge`.

Intentionally changed:
- Contact-law override workflow: `src/railway_simulator/studies/__init__.py`, `studies/parametric_grid.py`, `studies/full_train.py`.
- Contact initial-speed state: `src/railway_simulator/core/contact_state.py`, `core/engine.py`.
- Spectrum envelope overlap policy: `src/railway_simulator/spectrum/service.py`.
- Regression/path tests and helpers under `tests/`.
- `.gitignore` now excludes local `projects/`, `*.sqlite`, `*.sqlite3`, and `data.zip`.

Intentionally not changed:
- Bouc-Wen evolution equation.
- Legacy `Impact_Force_MN` definition; it remains equal to `Impact_Force_front_MN`.
- Strict `contact_law` vs non-`tabulated` contact-model validation in `ContactModels.compute_force`.

Removed local artifacts:
- `projects/` runtime workspace with SQLite/run CSV/spectra outputs.
- `docs/chapter_impact_current_engine/` generated chapter output.
- `.pytest_cache/`, `build/pytest-tmp/`, and `build/pytest_tmp/` cache/temp folders.
- Untracked exploratory `configs/studies/trackside_clearance_demand_traxx_mu030.yml`.
- Untracked exploratory `tools/build_chapter_impact_current_engine_assets.py`.
- `data.zip` was already absent at cleanup time.

## Verification Commands

```text
git status --short
git diff --stat
git diff --ignore-space-at-eol --stat
git diff --name-only
git diff --check
git ls-files --deleted
uv run pytest tests/test_parametric_grid.py tests/test_contact_state.py tests/test_confirmed_numerical_bugs.py tests/test_spectrum_service.py -q
uv run pytest tests/test_full_train_study.py tests/test_config_loader.py tests/test_energy_balance.py tests/test_hazard_montecarlo.py tests/test_hazard_sdof.py tests/test_picard_backcompat.py tests/test_set_by_path.py tests/test_contact_state_regression.py tests/test_parametric_grid_cli.py tests/test_parametric_grid_io.py tests/test_parametric_grid_persistence.py tests/test_project_workbench.py -q
uv run pytest -q
uv run --project C:\Users\sflab\railway-impact-simulator pytest C:\Users\sflab\railway-impact-simulator\tests\test_parametric_grid_io.py C:\Users\sflab\railway-impact-simulator\tests\test_parametric_grid_cli.py C:\Users\sflab\railway-impact-simulator\tests\test_parametric_grid_persistence.py C:\Users\sflab\railway-impact-simulator\tests\test_project_workbench.py C:\Users\sflab\railway-impact-simulator\tests\test_picard_backcompat.py -q
```

Results:
- Focused regression slice: `44 passed in 2.39s`.
- Expanded modified-module slice: `137 passed, 1 warning in 11.89s`.
- Full suite: `257 passed, 2 warnings in 28.57s`.
- Outside-CWD path-sensitive slice: `36 passed in 4.12s`.

Remaining warnings:
- `DeprecationWarning`: legacy configs without `collision` are auto-migrated.
- Git reports LF-to-CRLF checkout-normalization warnings, but `git diff --ignore-space-at-eol --stat` matches `git diff --stat`; no line-ending-only patch was found.
- `git diff --check` reported no whitespace errors.
- `git ls-files --deleted` is empty.

## Numerical Spot Checks

```text
hht_alpha_-0.15_energy_ratio=0.053553700120
hht_alpha_0_energy_ratio=0.999999999991
contact_override_model=hooke
contact_override_law_is_none=True
contact_override_flag=True
phantom_force_N=-100000.000000
phantom_backbone_N=-100000.000000
srs_envelope_Tn_ms=[200.0, 300.0]
srs_envelope_Feq_MN=[3.0, 1.0]
impact_front_peak_MN=0.010000000
impact_total_peak_MN=0.030000000
impact_delta_Fpeak=2.000000000
```

Additional covered checks:
- Direct manual `contact_law` plus non-`tabulated` model still raises.
- Friction typo `columb` raises `Unknown friction_model`.
- No generated SQLite database, runtime project workspace, local result CSV tree, or generated chapter output remains in `git status`.
