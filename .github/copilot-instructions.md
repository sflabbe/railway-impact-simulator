## Quick context

This repo implements an HHT-α railway impact simulator (CLI + optional Streamlit UI). Key entry points:

- CLI script: `railway-sim` → implemented in `src/railway_simulator/cli.py` (see CLI helpers and speed parsing).
- Streamlit UI: `src/railway_simulator/core/app.py` (launched with `railway-sim ui` or `streamlit run src/railway_simulator/core/app.py`).
- Core solver: `src/railway_simulator/core/engine.py` (numerical integration and simulation outputs).
- Parametric/study helpers: `src/railway_simulator/core/parametric.py` and `src/railway_simulator/studies/`.

Developer notes for AI agents
- Target Python: `>=3.10` (see `pyproject.toml`). Install core deps or use editable install (`pip install -e .`) when modifying behavior.
- UI extras: `.[ui]` installs Streamlit and heavier deps; protect those changes behind feature flags.

Project-specific patterns and conventions
- Configs are YAML/JSON and merged into defaults. Example config examples live in `configs/` and are authoritative.
- CLI speed spec format: a string like `"320:0.2,200:0.4,120:0.4"` → use `_parse_speeds_spec` in `cli.py` for parsing.
- Speed→velocity conversion: `_speed_kmh_to_v0_init` returns a negative value (velocity towards barrier). Preserve this sign convention.
- Output/logging: per-run logs are written to `<output_dir>/<runname>.log` (see `_setup_logger` in `cli.py`).
- Time columns: results use `Time_s` or `Time_ms`. UI/helpers prefer `Time_ms` for plotting; when reading DataFrames check both.
- DataFrame attrs: runtime metadata like `n_lu` and `n_nonlinear_iters` may be stored in `df.attrs` (tests mock these attributes).

Testing & expected behaviors
- Tests commonly inject a `simulate_func` (see `tests/test_convergence_study.py`) for deterministic behavior — follow the same pattern when adding unit-testable logic.
- Convergence/sensitivity helpers expect sorted dt behavior (the tests assert descending `dt_s` order). Preserve ordering semantics when modifying those functions.

Build / run workflows
- Install and run the CLI locally:
  - `python -m pip install -e .`
  - `railway-sim run --config configs/ice1_80kmh.yml --output-dir results/ice1_80`
- Run the Streamlit UI (dev):
  - `pip install "[ui]"`
  - `streamlit run src/railway_simulator/core/app.py --server.address 127.0.0.1 --server.port 8501`
- Portable Windows bundles: use `BUILD_PORTABLE_CLI.cmd` and `BUILD_PORTABLE_UI.cmd` (top-level scripts).

What to look for when changing solver / numerics
- Numerical parameters (dt, alpha, newton_tol) are sensitive. Study code in `src/railway_simulator/studies/` and replicate the `simulate_func` hook used in tests when adding regressions.
- Performance metrics are computed from `Time_s` arrays and `df.attrs`. If adding instrumentation, add to `df.attrs` so metrics functions pick them up.

Style & small implementation notes
- Keep public CLI behavior stable: flags and config keys are part of user-facing API. If you add a config key, update examples in `configs/` and `README.md`.
- Use the existing logging pattern (`_setup_logger`) so CI/dev runs produce per-run logs in `results/` directories.
- For plotting/UI, prefer `Time_ms` axis when possible; UI code contains helpers `_get_time_axis` and `make_envelope_figure`.

Files to inspect for examples
- `README.md` — usage, install, and example CLI calls.
- `pyproject.toml` — dependencies and `railway-sim` entry point.
- `src/railway_simulator/cli.py` — parsing, config loading, logging, and run helpers.
- `src/railway_simulator/core/app.py` — Streamlit app patterns and `execute_simulation` usage.
- `tests/test_convergence_study.py` — typical unit test style (function injection, small synthetic DataFrame, checks on ordering/columns).

If uncertain
- Prefer reproducing an existing CLI invocation (see README examples) and run a small synthetic scenario rather than large UI runs.
- Ask for the specific area (CLI, core solver, UI, or packaging) if multiple subsystems would be touched.

Please review any unclear areas (tests to preserve, config keys to keep backwards-compatible, and desired CI workflows) so I can refine this guidance.
