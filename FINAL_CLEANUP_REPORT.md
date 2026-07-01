# Final Cleanup Report

Date: 2026-07-02

## Files Intentionally Changed

- `src/railway_simulator/spectrum/service.py`: added an explicit `ValueError` when period grids overlap but no numerator periods remain inside the common overlap.
- `tests/test_spectrum_service.py`: added regression coverage for the empty effective-overlap ratio case.
- `FINAL_CLEANUP_REPORT.md`: refreshed this final cleanup report.

`.gitignore` already covered `.pytest-tmp/`, `.pytest_cache/`, `build/`, `__pycache__/`, Python bytecode, and SQLite files, so it was not changed.

## Commands Run

```text
git status --short
rg -n "class SpectrumService|def ratio|ratio\(" src tests
Get-Content .gitignore
rg --files tests
Get-Content src\railway_simulator\spectrum\service.py
Get-Content tests\test_spectrum_service.py
git ls-files
git clean -ndX
git clean -fdX -e .venv/
python -m pytest tests/test_spectrum_service.py -q
python -m pytest tests/test_full_train_study.py -q
python -m pytest tests/test_parametric_grid.py -q
uv sync
uv run python -m pytest tests/test_spectrum_service.py -q
uv run python -m pytest tests/test_full_train_study.py -q
uv run python -m pytest tests/test_parametric_grid.py -q
uv run python -m pytest tests/test_confirmed_numerical_bugs.py -q
uv run python -m pytest tests/test_contact_state.py -q
uv run python -m pytest tests/test_hazard_sdof.py -q
uv run python -m pytest -q
$env:MPLBACKEND = 'Agg'; .\.venv\Scripts\python.exe -m pytest tests/test_workbench_cli.py::test_srs_cli_export_and_compare_from_persisted_curves -q --basetemp="$env:TEMP\ris-pytest-clean"
$env:MPLBACKEND = 'Agg'; .\.venv\Scripts\python.exe -m pytest -q --basetemp="$env:TEMP\ris-pytest-full" -o cache_dir="$env:TEMP\ris-pytest-cache"
$env:MPLBACKEND = 'Agg'; Set-Location $env:TEMP; C:\Users\sflab\railway-impact-simulator\.venv\Scripts\python.exe -m pytest C:\Users\sflab\railway-impact-simulator\tests\test_spectrum_service.py -q --basetemp="$env:TEMP\ris-pytest-clean-cwd" -o cache_dir="$env:TEMP\ris-pytest-cache"
git diff --check
git diff --stat
git diff --ignore-space-at-eol --stat
git diff -- src\railway_simulator\spectrum\service.py tests\test_spectrum_service.py
git clean -fdX -e .pytest-tmp/
git clean -ndX
git diff --cached --name-only
git diff --name-only --diff-filter=D
git ls-files -o --exclude-standard
PowerShell zip creation from git ls-files to C:\Users\sflab\AppData\Local\Temp\railway-impact-simulator-clean-final2.zip
tar -tf C:\Users\sflab\AppData\Local\Temp\railway-impact-simulator-clean-final2.zip | Select-String -Pattern "pytest-tmp|pytest_cache|__pycache__|(^|/)build/|\.sqlite$|\.pyc$|srs_.*\.(csv|png)$|(^|/)chapter/.*\.tex$"
```

## Test Results

- `python -m pytest tests/test_spectrum_service.py -q`: `9 passed`.
- `python -m pytest tests/test_full_train_study.py -q`: `2 passed`.
- `python -m pytest tests/test_parametric_grid.py -q`: failed under global Python because `pydantic` was not installed after the local venv cleanup.
- `uv sync`: rebuilt the project virtual environment.
- `uv run python -m pytest tests/test_spectrum_service.py -q`: `9 passed`.
- `uv run python -m pytest tests/test_full_train_study.py -q`: `2 passed`.
- `uv run python -m pytest tests/test_parametric_grid.py -q`: `18 passed`.
- `uv run python -m pytest tests/test_confirmed_numerical_bugs.py -q`: `19 passed`.
- `uv run python -m pytest tests/test_contact_state.py -q`: `6 passed`.
- `uv run python -m pytest tests/test_hazard_sdof.py -q`: `44 passed`.
- Initial full suite `uv run python -m pytest -q`: `263 passed, 2 skipped, 1 failed`; the failure was `test_srs_cli_export_and_compare_from_persisted_curves` due to the environment's missing/broken Tcl/Tk install for Matplotlib.
- Retest of that CLI SRS case with `MPLBACKEND=Agg` and temp files outside the repo: `1 passed`.
- Full suite with `MPLBACKEND=Agg`, `--basetemp` outside the repo, and pytest cache outside the repo: `264 passed, 2 skipped, 2 warnings`.
- Clean-CWD spectrum check from `%TEMP%`: `9 passed`.

The remaining warnings were the existing legacy-config auto-migration `DeprecationWarning`s.

## Cleanup Status

- Removed ignored `.pytest_cache/`, `build/`, `.venv/`, egg-info, and `__pycache__/` artifacts with `git clean`.
- Post-clean `git clean -ndX` reports only `.pytest-tmp/`.
- No `*.sqlite` or `*.pyc` artifacts were found outside the ACL-protected `.pytest-tmp/` directory.
- `.pytest-tmp/` is not staged and will not be included in the final package, but local deletion is blocked by Windows access control (`WinError 5`). `git clean` also reports `warning: failed to remove .pytest-tmp/: Directory not empty`.

## Behavior Confirmations

- `SpectrumService.ratio()` now raises `ValueError` for empty effective overlap and still uses `np.interp(..., left=np.nan, right=np.nan)`.
- Partial-overlap ratio tests confirm no endpoint clamping outside the denominator range.
- Same-grid ratio behavior remains elementwise.
- No physics/model behavior was changed:
  - Bouc-Wen evolution equation unchanged.
  - Legacy `Impact_Force_MN` behavior unchanged.
  - Strict `contact_law` plus non-tabulated `contact_model` validation unchanged.

## Package

Final package command uses `git ls-files` so the zip is generated from tracked working-tree source only, including this final report and excluding ignored/generated artifacts:

```text
C:\Users\sflab\AppData\Local\Temp\railway-impact-simulator-clean-final2.zip
```

Package validation:

```text
tar -tf C:\Users\sflab\AppData\Local\Temp\railway-impact-simulator-clean-final2.zip | Select-String -Pattern "pytest-tmp|pytest_cache|__pycache__|(^|/)build/|\.sqlite$|\.pyc$|srs_.*\.(csv|png)$|(^|/)chapter/.*\.tex$"
```

Result: no matches.
