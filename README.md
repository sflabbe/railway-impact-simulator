# Railway Impact Simulator

HHT-α railway impact simulator with Bouc–Wen hysteresis, inspired by the dynamic load models discussed in
**DZSF Bericht 53 (2024)** (“Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr”).

It provides:

- A command line interface (**`railway-sim`**) for single runs and studies (speed mixes + envelopes, sensitivity, convergence)
- YAML-defined custom parametric grids for reproducible mini studies and chapter studies
- CSV / JSON output for post‑processing, plus SQLite-backed Project Workbench runs
- Optional ASCII plots for headless terminals (SSH, Termux, containers)
- Optional Streamlit-based UI/dashboard with Project Workbench and parametric-study launcher (extra deps)

> ⚠️ Python: **≥ 3.10** (project metadata).  
> ⚠️ On very new Python versions (e.g. 3.13), some scientific wheels may not be available on all platforms yet.

---

## 1. Development setup

This repository is managed with **uv**. The source of truth for dependencies is:

- `pyproject.toml` for declared runtime, optional UI, and development dependencies.
- `uv.lock` for the resolved dependency set.

There is currently no `requirements.txt` source of truth in this repository. If a legacy requirements export is added later, it should be treated as a generated compatibility artifact, not as the primary dependency declaration.

### 1.1 Install uv

Linux / macOS / WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 1.2 Clone and synchronize the development environment

```bash
git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator
uv sync --all-extras --dev
```

This creates and synchronizes the local `.venv` from `pyproject.toml` and `uv.lock`.

### 1.3 Run basic checks

```bash
uv run railway-sim --help
uv run pytest
uv run ruff check .
```

Equivalent Make targets are provided:

```bash
make sync
make test
make lint
make smoke
```

`make typecheck` is intentionally only a notice target at the moment: `mypy` is not configured for this repo, so no strong typing gate is claimed.

### 1.4 Run the CLI

```bash
uv run railway-sim run \
  --config configs/ice1_aluminum.yml \
  --speed-kmh 80 \
  --output-dir results/ice1_80 \
  --ascii-plot
```

### 1.5 Run the Streamlit UI

The UI dependencies are installed by `uv sync --all-extras --dev` because the `ui` extra is included. Start the UI with:

```bash
uv run railway-sim ui
```

If you synchronized only the default environment, run with the UI extra explicitly:

```bash
uv run --extra ui streamlit run src/railway_simulator/core/app.py
```

Open `http://127.0.0.1:8501` in your browser. The Project Workbench includes a **Parametric studies** section for loading YAML study specs, previewing scenarios, running limited mini grids, inspecting warnings, plotting peak-force summaries, and downloading CSV/HTML outputs.

### 1.6 Update dependencies

Add runtime dependency:

```bash
uv add package-name
```

Add development dependency:

```bash
uv add --dev package-name
```

Update or regenerate the lockfile:

```bash
uv lock
uv lock --check
```

Portable Windows bundle note: `tools/windows-portable/Build_Portable_Bundle.ps1` still bootstraps the Python embeddable distribution with pip. That path is retained as a legacy packaging fallback and is not the dependency source of truth.

## 2. Quickstart

Example configs live in `configs/`.

### 2.1 Windows PowerShell quickstart

```powershell
railway-sim run `
  --config configs/ice1_aluminum.yml `
  --speed-kmh 80 `
  --output-dir results/ice1_80 `
  --ascii-plot
```

### 2.2 Linux / macOS / WSL quickstart

```bash
railway-sim run \
  --config configs/ice1_aluminum.yml \
  --speed-kmh 80 \
  --output-dir results/ice1_80 \
  --ascii-plot
```

Output files typically include:

- `results.csv` (time history)
- `run.log` (log file)

---

## 3. Common CLI workflows

### 3.1 Single run

```bash
railway-sim run \
  --config configs/ice1_steel.yml \
  --speed-kmh 120 \
  --output-dir results/ice1_steel_120 \
  --plot
```

### 3.2 Parametric study (speed mix envelope)

Example: speed mix **320 / 200 / 120 km/h** with weights **0.2 / 0.4 / 0.4** for the envelope of `Impact_Force_MN`:

```bash
railway-sim parametric \
  --base-config configs/ice1_aluminum.yml \
  --speeds "320:0.2,200:0.4,120:0.4" \
  --quantity Impact_Force_MN \
  --output-dir results_parametric/track_mix \
  --prefix track_mix \
  --ascii-plot
```

### 3.3 Custom parametric grid from YAML

Custom studies are defined in `configs/studies/*.yml` and can be previewed before running. The minimal smoke-test study is:

```bash
uv run railway-sim study run-grid \
  --spec configs/studies/impact_parametric_mini.yml \
  --dry-run
```

Run only the first scenario and export a summary:

```bash
uv run railway-sim study run-grid \
  --spec configs/studies/impact_parametric_mini.yml \
  --limit 1 \
  --out build/parametric_summary.csv
```

Persist the same run into a Project Workbench SQLite database:

```bash
uv run railway-sim study run-grid \
  --spec configs/studies/impact_parametric_mini.yml \
  --db projects/impact_workbench/project.sqlite \
  --limit 1 \
  --out build/parametric_persistent_summary.csv
```

The summary contains one row per scenario, scenario metadata, extracted peak metrics when available, and a `warnings` column. Solver warnings are intentionally kept visible even when the scenario status is `ok`.

For the full workflow, YAML format, SQLite mapping, UI usage, and CSV/HTML exports, see `docs/PARAMETRIC_STUDIES.md`.

### 3.4 Convergence study

```bash
railway-sim convergence \
  --config configs/traxx_freight.yml \
  --dts "2e-4,1e-4,5e-5" \
  --quantity Impact_Force_MN \
  --out results_parametric/convergence
```

### 3.5 Live terminal monitor

The live terminal monitor provides a full-screen progress view with live plots while a simulation runs.
It is designed for headless terminals, including SSH sessions.

Requirements:

- A TTY for stdin and stdout.
- Python curses support (available on Linux, macOS, and WSL; Windows Python may not ship with curses).

Run it by adding `--live-monitor`:

```bash
railway-sim run \
  --config configs/ice1_aluminum.yml \
  --speed-kmh 80 \
  --output-dir results/ice1_80_live \
  --live-monitor
```

Useful flags:

- `--live-refresh-s 0.05` controls how often the UI refreshes.
- `--live-sample-s 0.02` controls how often the solver emits progress samples.
- `--live-hold` keeps the monitor open until you press `q`.
- `--live-auto-close` exits the monitor automatically on completion.

Expected output (first lines of the full screen view):

```
Railway impact simulator  status running
step 120/400  30.0%   t   0.0120 s   dt 1.00e-04   wall     1.2 s
solver newton   iters 3   max resid 1.20e-06   contact 1
F    12.345 MN   pen     1.234 mm   v   -22.220 m/s   a     0.500 g   Eb 1.234e-05
```

Troubleshooting:

If you request `--live-monitor` in a non-interactive shell, the CLI runs without the monitor and logs a warning.
If your terminal does not support Unicode, the UI falls back to ASCII sparklines.
Press `q` to quit, `p` to pause, `v` to switch views, and `s` to save a text snapshot in the output directory.

---

## 4. Solver knobs (YAML keys)

These keys are commonly used when tuning the nonlinear solver:

- `solver`: `"newton"` (default) or `"picard"` (legacy fixed-point).
- `max_iter`: Newton–Raphson max iterations per step.
- `newton_tol`: Newton–Raphson residual tolerance.
- `picard_max_iters`: Picard max iterations per step.
- `picard_tol`: Picard residual tolerance.

> Backward compatibility: if a YAML sets `max_iter` or `newton_tol` but omits Picard controls,
> the simulator mirrors those values into `picard_max_iters` / `picard_tol`.

---

## 5. Configuration files (YAML / JSON)

Configs are loaded from `--config` / `--base-config` and merged into internal defaults.

- Descriptive metadata keys like `case_name` / `notes` are accepted and ignored by the solver.
- Other unknown keys are ignored with a warning (helps catch typos).
- The example configs under `configs/` are the best reference for valid keys and units.

Minimal override example:

```yaml
# quickstart.yml
v0_init: -22.22   # [m/s] towards the barrier (sign convention)
T_max: 0.40       # [s] total simulated time
h_init: 1.0e-4    # [s] timestep
```

> Note: if you set `T_max` and `h_init` but do not set `step`, the simulator derives a consistent
> `step ≈ T_max / h_init`.

---

## 6. Repo layout

```
.
├── configs/   # YAML example configs (ICE1, TRAXX, etc.) and configs/studies/*.yml
├── docs/      # Documentation and reports
├── examples/  # Scripts and worked examples
├── src/       # Library + CLI implementation
├── tests/     # Pytest regression tests
└── tools/     # Helper scripts (portable builds, utilities)
```

---

## 7. Streamlit UI

The Streamlit app entry point is:

- `src/railway_simulator/core/app.py`

Start the UI after synchronizing the `ui` extra with uv:

```bash
uv run railway-sim ui
```

Or explicitly request the UI extra:

```bash
uv run --extra ui streamlit run src/railway_simulator/core/app.py
```

Open `http://127.0.0.1:8501` (default) in your browser. In **Project Workbench → Parametric studies**, use `configs/studies/impact_parametric_mini.yml` for the first smoke test. The UI supports scenario preview, limited execution, SQLite persistence, summary tables, solver-warning display, peak-force chart data, and CSV/HTML downloads.

---

## 8. More docs in this repo

- `PROJECT_SUMMARY.md` – overview / roadmap
- `ARCHITECTURE.md` – system design notes
- `DEVELOPER_GUIDE.md` – developer notes and local workflows
- `docs/PARAMETRIC_STUDIES.md` – YAML parametric grids, CLI/UI workflows, SQLite persistence, and exports
- `VALIDATION_Pioneer.md` – validation notes
- `CITATION_REFERENCE.md` – citation/reference notes

---

## 9. License

MIT License — see `LICENSE`.

---

## 10. Citation

If you use this simulator in academic work, cite the repository and the DZSF research report it is based on
(see `CITATION_REFERENCE.md`).
