# Railway Impact Simulator

HHT-α railway impact simulator with Bouc–Wen hysteresis, inspired by the dynamic load models discussed in
**DZSF Bericht 53 (2024)** (“Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr”).

It provides:

- A command line interface (**`railway-sim`**) for single runs and studies (speed mixes + envelopes, sensitivity, convergence)
- CSV output for post‑processing
- Optional ASCII plots for headless terminals (SSH, Termux, containers)
- Optional Streamlit-based UI/dashboard (extra deps)

> ⚠️ Python: **≥ 3.10** (project metadata).  
> ⚠️ On very new Python versions (e.g. 3.13), some scientific wheels may not be available on all platforms yet.

---

## 1. Installation

### 1.1 Clone the repository

```bash
git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator
```

### 1.2 Create a virtual environment (recommended)

Linux / macOS / WSL:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 1.3 Install the core package

```bash
python -m pip install .
```

Check that the entry point is available:

```bash
railway-sim --help
```

### 1.4 Install with UI extras

The UI dependencies are bundled as an extra named `ui`:

```bash
python -m pip install ".[ui]"
```

---

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

### 3.3 Convergence study

```bash
railway-sim convergence \
  --config configs/traxx_freight.yml \
  --dts "2e-4,1e-4,5e-5" \
  --quantity Impact_Force_MN \
  --out results_parametric/convergence
```

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
├── configs/   # YAML example configs (ICE1, TRAXX, etc.)
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

Start the UI after installing `.[ui]`:

```bash
railway-sim ui
```

Open `http://127.0.0.1:8501` (default) in your browser.

---

## 8. More docs in this repo

- `PROJECT_SUMMARY.md` – overview / roadmap
- `ARCHITECTURE.md` – system design notes
- `DEVELOPER_GUIDE.md` – developer notes and local workflows
- `VALIDATION_Pioneer.md` – validation notes
- `CITATION_REFERENCE.md` – citation/reference notes

---

## 9. License

MIT License — see `LICENSE`.

---

## 10. Citation

If you use this simulator in academic work, cite the repository and the DZSF research report it is based on
(see `CITATION_REFERENCE.md`).
