# Railway Impact Simulator

Hilber–Hughes–Taylor-α (HHT-α) implicit time integration with Bouc–Wen hysteresis for simulating railway vehicle impacts on rigid barriers and bridge/abutment piers.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DZSF%20Report-10.48755/dzsf.240006.01-blue.svg)](https://doi.org/10.48755/dzsf.240006.01)

The code implements the discrete train model, contact laws and parameter sets developed in:

> Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024).  
> *Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr* (DZSF Bericht 53).  
> Deutsches Zentrum für Schienenverkehrsforschung beim Eisenbahn-Bundesamt.  
> https://doi.org/10.48755/dzsf.240006.01

A short project overview is given in [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md).  
Detailed citation formats are collected in [`CITATION_REFERENCE.md`](CITATION_REFERENCE.md).

---

## 1. Features

- Discrete multi-mass train model (locomotive + wagons, configurable geometry and masses)
- HHT-α implicit time integration with Bouc–Wen hysteresis for vehicle crushing
- Linear and Hertz-type contact laws (Anagnostopoulos, Hunt–Crossley, Lankarani–Nikravesh, etc.)
- Several friction formulations (LuGre, Dahl, Coulomb–Stribeck, Brown–McPhee)
- Optional SDOF building/pier model with either linear or Takeda-type degrading hysteresis
- Energy bookkeeping (kinetic, spring, contact, damping and friction losses)
- Streamlit web interface for interactive studies
- Command-line interface (CLI) for batch runs based on YAML/JSON configuration files
- Speed-based parametric studies with envelopes (“Umhüllende”) and weighted mean histories
- ASCII-style plots and Matplotlib pop-up plots directly from the CLI
- Performance metrics (wall-clock time, real-time factor, LU solves, estimated FLOPs)
- Optional PDF reports and log files for documenting simulation runs

Validated against the Pioneer wagon crash test (FRA, 1999); see [`VALIDATION_Pioneer.md`](VALIDATION_Pioneer.md).

---

## 2. Installation

Requirements:

- Python 3.10 or newer
- A recent `pip`

From a clean environment:

```bash
git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator

# Install as a package (recommended)
pip install .

# or, for development
pip install -e .
```

This installs the `railway_simulator` package and the `railway-sim` CLI entry point.

---

## 3. Command-line usage

Show top-level help:

```bash
railway-sim --help
```

### 3.1 Single-run simulation

Run a simulation from a configuration file:

```bash
railway-sim \
  --config configs/ice1_80kmh.yml \
  --output-dir results/ice1_80
```

- `--config` points to a YAML or JSON file describing:
  - train geometry and masses
  - Bouc–Wen material parameters
  - contact and friction model
  - integration settings (time step, duration, tolerances)
- `--output-dir` is the directory where time histories and logs are written.

The CLI writes:

- `results.csv` – full time history (contact force, penetration, acceleration, energies, etc.)
- a console summary with performance metrics (wall time, real-time factor, LU solves, FLOPs)
- a log file with the same information and additional details (file name depends on prefix/output directory)

Example with a filename prefix:

```bash
railway-sim \
  --config configs/ice1_80kmh.yml \
  --output-dir results/ice1_80 \
  --prefix ice1_80
```

### 3.2 Parametric studies & envelopes

The `parametric` subcommand runs a family of simulations at different speeds and builds an envelope and (optionally) a weighted mean history.

Example: a line with 20% high-speed (TGV), 40% IC passenger traffic and 40% cargo:

```bash
railway-sim parametric \
  --base-config configs/ice1_80kmh.yml \
  --speeds "320:0.2,200:0.4,120:0.4" \
  --quantity Impact_Force_MN \
  --output-dir results_parametric/track_mix \
  --prefix track_mix
```

- `--base-config` is a YAML/JSON config with the common train/contact/material setup.
- `--speeds` defines a set of speeds in km/h and optional weights:

  - `"320:0.2,200:0.4,120:0.4"` → speeds 320, 200, 120 km/h with weights 0.2, 0.4, 0.4  
  - `"80,120,160"` → speeds 80, 120, 160 km/h, all with weight 1.0

- `--quantity` selects which result column to envelope (e.g. `Impact_Force_MN`, `Acceleration_g`, …).

The parametric command writes:

- `track_mix_Impact_Force_MN_envelope.csv` – time histories of:
  - `Impact_Force_MN_envelope` (pointwise max over scenarios)
  - `Impact_Force_MN_weighted_mean` (weighted average over scenarios)
- `track_mix_Impact_Force_MN_summary.csv` – per-scenario summary (peak, time of peak, LU solves, etc.)
- a console summary with parametric performance metrics

### 3.3 CLI plotting options

Both `run` and `parametric` support simple plotting directly from the CLI:

- ASCII plot (printed in terminal)
- Matplotlib window (interactive popup)

Single run example:

```bash
railway-sim \
  --config configs/ice1_80kmh.yml \
  --output-dir results/ice1_80 \
  --ascii-plot \
  --plot
```

Parametric example (envelope of `Impact_Force_MN`):

```bash
railway-sim parametric \
  --base-config configs/ice1_80kmh.yml \
  --speeds "320:0.2,200:0.4,120:0.4" \
  --quantity Impact_Force_MN \
  --output-dir results_parametric/track_mix \
  --prefix track_mix \
  --ascii-plot \
  --plot
```

The CLI ensures that both axes start at zero (`x >= 0`, `y >= 0`) for these quick-look plots.

### 3.4 Logging & performance metrics

Each CLI run prints performance metrics such as:

- wall-clock time
- simulated time span
- number of time steps
- min/mean/max Δt
- real-time factor (simulated time / wall time)
- approximate number of linear solves and DOFs
- estimated LU FLOPs and FLOP/s

The same information is written into a log file in the chosen `--output-dir`, together with additional metadata about the configuration and (for parametric runs) the scenario list.

### 3.5 PDF report generation

For documentation and reporting, you can generate a compact PDF report that summarises:

- key input parameters
- envelope or single-run curves
- performance metrics
- basic plots of the selected quantity

Enable it with `--pdf-report`:

```bash
railway-sim parametric \
  --base-config configs/ice1_80kmh.yml \
  --speeds "320:0.2,200:0.4,120:0.4" \
  --quantity Impact_Force_MN \
  --output-dir results_parametric/track_mix \
  --prefix track_mix \
  --pdf-report
```

The CLI prints the path of the generated PDF report on success.  
On desktop systems, the default behaviour is to try to open the report with the system PDF viewer.

---

## 4. Streamlit application

For interactive studies, launch the Streamlit app from the repository root:

```bash
streamlit run src/railway_simulator/core/app.py
```

The app provides:

**Sidebar controls for:**

- train configuration (research locomotive model and generic passenger/freight presets)
- Bouc–Wen material parameters and presets (e.g. aluminium vs steel)
- contact and friction models
- HHT-α time integration settings
- optional SDOF building/pier model
- basic parametric settings for quick speed sweeps and envelopes

**Plots for:**

- impact force, penetration and acceleration over time
- force–penetration hysteresis, including contact backbone
- energy components and overall energy balance
- (optionally) envelopes over a set of speeds

**Optional outputs:**

- SDOF building response (displacement, velocity, acceleration)
- building hysteresis (linear or Takeda-type)
- force-based response spectra for different damping ratios
- export of the full time history as CSV, TXT or XLSX

---

## 5. Repository layout

Main elements:

- `pyproject.toml`  
  Project metadata, dependencies and CLI entry point (`railway-sim`).

- `configs/`  
  Example configuration files (YAML), e.g.:
  - `ice1_80kmh.yml` – ICE-1 / Pioneer-type impact around 80 km/h.

- `examples/`  
  Small scripts demonstrating parametric usage, e.g.:
  - `parametric_line_mix.py` – build a simple line mix (TGV / IC / cargo) and call the CLI.

- `src/railway_simulator/` – Python package:
  - `__init__.py` – package metadata and public API
  - `cli.py` – command-line interface (`railway-sim`)
  - `core/engine.py` – numerical engine (HHT-α + Bouc–Wen + contact + friction + energy bookkeeping)
  - `core/app.py` – Streamlit application
  - `core/parametric.py` – helpers for parametric studies and envelopes
  - `core/report.py` – helper for generating PDF reports
  - `core/__init__.py` – core exports

- `VALIDATION_Pioneer.md`  
  Validation against the FRA Pioneer wagon crash test.

- `LICENSE`  
  MIT license for the software.

---

## 6. Validation

The default configuration is based on a single passenger wagon impacting a rigid wall at approximately 80 km/h (Pioneer wagon crash test).

Validation covers:

- peak and plateau impact forces  
- impact duration  
- energy dissipation  
- force–displacement hysteresis  

Details, references and acceptance criteria are given in `VALIDATION_Pioneer.md`.

---

## 7. Citing

If you use this code or the underlying methodology, please cite both the software and the research report.

### Software

```text
Labbé, S. (2025). Railway Impact Simulator: HHT-α implicit integration
with Bouc–Wen hysteresis [Computer software].
https://github.com/sflabbe/railway-impact-simulator
```

### Research report

```text
Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024).
Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr
(DZSF Bericht No. 53). Deutsches Zentrum für Schienenverkehrsforschung
beim Eisenbahn-Bundesamt. https://doi.org/10.48755/dzsf.240006.01
```

---

## 8. Licenses and disclaimer

### Software

The code in this repository is:

> © 2025 Sebastián Labbé  
> Licensed under the MIT License. See `LICENSE`.

### Research report

The underlying DZSF research report is licensed under:

> Creative Commons Attribution 4.0 International (CC BY 4.0)  
> https://creativecommons.org/licenses/by/4.0/

This implementation is provided without warranty and does not replace a full structural design according to the applicable standards.  
For safety-critical applications, use the tool only together with professional engineering judgement and the relevant design codes.
