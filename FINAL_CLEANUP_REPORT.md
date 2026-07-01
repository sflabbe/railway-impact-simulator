# Final Cleanup Report

Date: 2026-07-02

Branch: `bugfix/final-merge-mojones`

## Files Changed Intentionally

- `.codex`: removed zero-byte local-agent artifact.
- `.gitignore`: added `.pytest-tmp/`.
- `pyproject.toml`: changed pytest basetemp to `.pytest-tmp` and added `pythonpath = ["src"]`.
- `src/railway_simulator/studies/__init__.py`: made contact-law normalization defensive.
- `src/railway_simulator/studies/full_train.py`: deep-copied base/scenario params.
- `src/railway_simulator/studies/parametric_grid.py`: only normalizes contact law on actual contact-model overrides.
- `src/railway_simulator/spectrum/service.py`: removed silent endpoint clamping from `ratio()`.
- `tests/test_parametric_grid.py`, `tests/test_full_train_study.py`, `tests/test_spectrum_service.py`: regression coverage.

## Verification Commands

```text
python -m pytest tests/test_spectrum_service.py -q
python -m pytest tests/test_parametric_grid.py -q
python -m pytest -q
uv run python -m pytest tests/test_parametric_grid.py -q
uv run python -m pytest tests/test_spectrum_service.py -q
uv run python -m pytest tests/test_full_train*.py -q
uv run python -m pytest tests/test_contact*.py -q
uv run python -m pytest tests/test_full_train_study.py -q
uv run python -m pytest tests/test_contact_state.py tests/test_contact_state_regression.py -q
uv run python -m pytest -q
uv run --project C:\Users\sflab\railway-impact-simulator python -m pytest C:\Users\sflab\railway-impact-simulator\tests -q
git status --short
git diff --stat
git diff --ignore-space-at-eol --stat
git diff --check
git diff --cached --name-only
git diff --name-only --diff-filter=D
git ls-files -o --exclude-standard
```

## Results

- `python -m pytest tests/test_spectrum_service.py -q`: `8 passed in 1.13s`.
- Plain system `python -m pytest ...` now uses `src/` and `.pytest-tmp`; broader runs stop on missing system dependencies (`pydantic`, `typer`, `plotly`), not on the old missing `build/` parent.
- `uv run python -m pytest tests/test_parametric_grid.py -q`: `18 passed`.
- `uv run python -m pytest tests/test_spectrum_service.py -q`: `8 passed`.
- PowerShell passed wildcard paths literally for `tests/test_full_train*.py` and `tests/test_contact*.py`; concrete equivalents were run.
- `uv run python -m pytest tests/test_full_train_study.py -q`: `2 passed`.
- `uv run python -m pytest tests/test_contact_state.py tests/test_contact_state_regression.py -q`: `9 passed`.
- `uv run python -m pytest -q`: `265 passed, 3 warnings`.
- Outside-CWD project-environment run from `C:\Users\sflab\AppData\Local\Temp`: `265 passed, 3 warnings`.

Warnings observed:
- Legacy config auto-migration `DeprecationWarning`.
- Existing local `.pytest_cache` is unwritable in this sandbox, causing a pytest cache warning only.

## Confirmations

- Pytest no longer depends on a missing `build/` parent; basetemp is `.pytest-tmp`.
- Full-train scenarios no longer share or contaminate `collision_meta`.
- Contact-law normalization returns an isolated config and metadata dict.
- `SpectrumService.ratio()` restricts work to overlapping periods and uses `left=np.nan, right=np.nan`.
- The tracked zero-byte `.codex` artifact was removed.
- No SQLite/local database is staged; no files are staged at all.
- No generated runtime artifacts are staged; the only untracked non-ignored path is this report.
- No unrelated deleted files remain; the only deleted tracked path is `.codex`.
- `git diff --stat` and `git diff --ignore-space-at-eol --stat` match; `git diff --check` reported no whitespace errors.
