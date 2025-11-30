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

Show help:

```bash
railway-sim --help
```

Run a simulation from a configuration file:

```bash
railway-sim \
  --config configs/ice1_80kmh.yml \
  --output-dir results/
```

- `--config` points to a YAML or JSON file describing:
  - train geometry and masses
  - Bouc–Wen material parameters
  - contact and friction model
  - integration settings (time step, duration, tolerances)
- `--output-dir` is the directory where time histories and summaries are written.

Use `configs/ice1_80kmh.yml` as a template and adapt parameters as needed.

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

**Plots for:**

- impact force, penetration and acceleration over time
- force–penetration hysteresis, including contact backbone
- energy components and overall energy balance

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

- `src/railway_simulator/` – Python package:
  - `__init__.py` – package metadata and public API
  - `cli.py` – command-line interface
  - `core/engine.py` – numerical engine (HHT-α + Bouc–Wen + contact + friction)
  - `core/app.py` – Streamlit application
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
