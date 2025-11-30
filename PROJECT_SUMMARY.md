# Railway Impact Simulator – Project summary

This repo bundles the numerical engine, CLI and simple UI that grew out of the DZSF project

> *Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr*  
> (DZSF-Bericht 53, 2024)

The goal is **not** to be a full-blown FE code, but a small, transparent tool to explore impact loads from railway vehicles and play with the assumptions behind design values.

---

## 1. What this project does

- Models a **train as a discrete multi-mass system** (lumps + nonlinear springs).
- Uses **HHT-α implicit integration** with:
  - Bouc–Wen hysteresis for crushing / plastic deformation
  - Optional SDOF “building / pier” DOF with linear or Takeda-type hysteresis
- Supports several **contact laws**:
  - linear penalty
  - Hunt–Crossley / Lankarani–Nikravesh style nonlinear contact
  - Anagnostopoulos gap element
- Includes a few **friction models**:
  - simple Coulomb / Stribeck
  - LuGre / Dahl / Brown–McPhee style law (for sensitivity work)
- Tracks **energy consistently**:
  - kinetic
  - spring / contact
  - Rayleigh damping and friction losses
  - accumulated dissipation and energy balance error

Everything revolves around short, reproducible impact scenarios:
- single vehicle on a rigid wall
- vehicle against a stiff pier / SDOF
- parametric speed mixes for “line mixtures” (track categories, etc.)

---

## 2. Main components

- **Numerical engine** – `src/railway_simulator/core/engine.py`  
  HHT-α stepper + Bouc–Wen springs + contact / friction routines + energy bookkeeping.

- **CLI** – `src/railway_simulator/cli.py`
  - `railway-sim run` – single scenario from YAML/JSON config  
  - `railway-sim parametric` – speed-based parametric study with envelopes and weighted means  
  - Optional ASCII plots, matplotlib plots, log files and PDF reports.

- **Parametric helper** – `src/railway_simulator/core/parametric.py`  
  Builds scenarios from a base config (e.g. speed mixes) and computes envelopes.

- **Streamlit UI** – `src/railway_simulator/core/app.py`  
  Lightweight UI for trying out parameter changes without editing YAML by hand.

- **Reporting** – `src/railway_simulator/core/report.py`  
  Small PDF report generator for documenting single runs or parametric studies.

- **Configs and examples**
  - `configs/ice1_80kmh.yml` – ICE-1 / Pioneer-type reference case
  - `examples/parametric_line_mix.py` – minimal script for speed mixtures

---

## 3. Typical workflows

**Single impact run**

- Start from `configs/ice1_80kmh.yml`
- Adjust:
  - initial speed `v0_init`
  - Bouc–Wen parameters (`fy`, `uy`, `bw_*`)
  - contact model and wall stiffness
  - optional building DOF
- Run via CLI:

  ```bash
  railway-sim run \
    --config configs/ice1_80kmh.yml \
    --output-dir results/ice1_80 \
    --ascii-plot \
    --plot \
    --pdf-report
  ```

**Parametric line mixture**

- Keep a base config for the vehicle / track.
- Use `railway-sim parametric` with a speed/weight specification:

  ```bash
  railway-sim parametric \
    --base-config configs/ice1_80kmh.yml \
    --speeds "320:0.2,200:0.4,120:0.4" \
    --quantity Impact_Force_MN \
    --output-dir results_parametric/track_mix \
    --prefix track_mix \
    --ascii-plot \
    --plot \
    --pdf-report
  ```

This writes:
- time history (`*_envelope.csv`)
- scenario summary (`*_summary.csv`)
- a compact PDF report
- a log file with performance metrics (time step statistics, LU counts, FLOP estimates).

---

## 4. Intended use (and non-goals)

The code is meant for:

- research and teaching on train–structure impact behaviour
- exploring the sensitivity of impact loads to:
  - vehicle layout
  - crush characteristics
  - contact law choices
  - line mixtures and speed distributions
- quick parametric “what if” checks before building more complex FE models

It is **not**:

- a replacement for full structural design according to Eurocode / Ril / etc.
- a general-purpose FE solver
- a black-box design tool

Always combine results with proper engineering judgement and the relevant design codes.
