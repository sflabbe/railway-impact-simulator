# Railway Impact Simulator

HHT-α railway impact simulator with Bouc–Wen hysteresis, based on the dynamic load models from DZSF Bericht 53 (2024).  
The code implements a configurable, vectorised mass–spring model for train impacts on rigid obstacles and provides:

- A command line interface (`railway-sim`) for single runs and parametric studies
- CSV/NumPy output for post‑processing
- Optional ASCII plots for headless terminals (SSH, Termux, etc.)
- (Optional) extra dependencies for UI/visualisation based on Streamlit and Plotly

> ⚠️ The repository currently targets **Python ≥ 3.10**.

---

## 1. Installation

### 1.1. Clone the repository

```bash
git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator
```

### 1.2. Create a virtual environment (recommended)

On Linux / macOS / WSL:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### 1.3. Install the core package

The *core* installation includes only the numerical engine, CLI and plotting:

```bash
pip install --no-build-isolation .
```

You can then check that the entry point is available:

```bash
railway-sim --help
```

You should see the CLI help with the available sub‑commands (e.g. `run`, `parametric`).

---

## 2. Optional UI / heavy dependencies

The project defines an extra dependency group called `ui` that pulls in heavier libraries such as **Streamlit** and **pyarrow** (used for dashboards / richer visualisation).

To install the package **with** these optional UI dependencies:

```bash
pip install --no-build-isolation ".[ui]"
```

> 💡 On typical desktop Python distributions (Linux, Windows, macOS) this will use pre‑built wheels for `pyarrow`.  
> ⚠️ On Android / Termux this is **not recommended**: `pyarrow` tries to compile the Apache Arrow C++ stack and will most likely fail or take a very long time. On Termux, stick to the *core* installation without `[ui]`.

---

## 3. Quickstart (single run)

The repository ships with example configuration files in `configs/`.  
A good starting point is an **ICE‑1 middle car at 80 km/h** impacting a rigid obstacle.

From the repository root, with the virtual environment activated:

```bash
railway-sim run \
  --config configs/ice1_80kmh.yml \
  --output-dir results/ice1_80
```

This will:

- Load the YAML configuration `configs/ice1_80kmh.yml`
- Run a single HHT‑α + Bouc–Wen simulation
- Write a time history to `results/ice1_80/results.csv`
- Create a detailed log file at `results/ice1_80/run.log`

The CSV typically contains columns such as

- `Time_s`, `Time_ms`
- Kinematic quantities for each mass
- `Impact_Force_MN` (or similar) for the contact force

You can post‑process the CSV with your tool of choice (Pandas, Excel, MATLAB, etc.).

---

## 4. Parametric study example

The CLI also supports parametric studies where a base configuration is re‑run for multiple speeds and combined into an envelope.

Example: a **track mix** with 320 / 200 / 120 km/h, with usage weights 0.2 / 0.4 / 0.4, for the envelope of `Impact_Force_MN`:

```bash
railway-sim parametric \
  --base-config configs/ice1_80kmh.yml \
  --speeds "320:0.2,200:0.4,120:0.4" \
  --quantity Impact_Force_MN \
  --output-dir results_parametric/track_mix \
  --prefix track_mix
```

This will:

- Build three scenarios:
  - `v320`: 320 km/h, weight 0.2
  - `v200`: 200 km/h, weight 0.4
  - `v120`: 120 km/h, weight 0.4
- Run all three simulations
- Write the **envelope time history** to  
  `results_parametric/track_mix/track_mix_Impact_Force_MN_envelope.csv`
- Write a **scenario summary** (peaks, etc.) to  
  `results_parametric/track_mix/track_mix_Impact_Force_MN_summary.csv`
- Log performance statistics (wall‑clock time, time‑step info, FLOP estimates, …)

---

## 5. ASCII plots (headless / Termux / SSH)

For headless environments you can enable an ASCII plot directly in the terminal.  
This is useful on:

- SSH sessions without X forwarding
- Android / Termux
- Minimal containers

Add the `--ascii-plot` flag:

```bash
railway-sim run \
  --config configs/ice1_80kmh.yml \
  --output-dir results/ice1_80_ascii \
  --ascii-plot
```

and for a parametric envelope:

```bash
railway-sim parametric \
  --base-config configs/ice1_80kmh.yml \
  --speeds "320:0.2,200:0.4,120:0.4" \
  --quantity Impact_Force_MN \
  --output-dir results_parametric/track_mix_ascii \
  --prefix track_mix \
  --ascii-plot
```

At the end of the run, the CLI prints an ASCII envelope such as:

```text
ASCII envelope plot (Impact_Force_MN vs Time_ms):
# Impact_Force_MN_envelope envelope
            ***
           *  *
          *    *
        **      *
      ***        *
    ***           *
***                ***************
# Time [ms] (0 – 400)
```

(Shape and scale depend on the configuration, this is just a schematic example.)

---

## 6. Configuration files

Configuration is provided via **YAML** (or JSON) files passed to `--config` / `--base-config`.

Internally, the CLI does roughly:

1. Load a default parameter dictionary from `get_default_simulation_params()`
2. Load your config file into a flat dictionary
3. Override default values with those found in the config
4. Create a `SimulationParams` object and run the solver

Some important notes:

- Keys in your YAML must match the fields of `SimulationParams`.  
  If you add unknown keys, you will get errors like  
  `SimulationParams.__init__() got an unexpected keyword argument 'foo'`.
- The provided example configs under `configs/` are the best reference for valid keys and units.
- You normally **do not** wrap parameters in extra levels like `scenario:` or `output:` in your own configs, unless you wired your own loader around the core engine.

A minimal example that only overrides a couple of defaults could look like:

```yaml
# quickstart.yml
# Only keys listed here override the internal defaults.

v0_init: -22.22     # [m/s] initial train velocity (sign convention from the model)
T_end: 0.40         # total simulated time [s], exact name depends on SimulationParams
```

> ⚠️ The exact parameter names (`T_end`, `v0_init`, etc.) should follow the ones used in the existing example configs and the `SimulationParams` definition.

---

## 7. Termux / Android notes

It is possible to run the simulator on Android via **Termux**.  
The recommended approach is:

1. Install base packages in Termux:

   ```bash
   pkg update
   pkg upgrade

   pkg install python python-pip \
               numpy scipy pandas matplotlib \
               clang cmake pkg-config
   ```

2. Clone the repository and create a virtual environment that can see the system‑wide scientific stack:

   ```bash
   git clone https://github.com/sflabbe/railway-impact-simulator.git
   cd railway-impact-simulator

   python -m venv .venv --system-site-packages
   source .venv/bin/activate
   ```

3. Install the **core** package only (no UI extras):

   ```bash
   pip install --no-build-isolation .
   ```

4. Test the CLI:

   ```bash
   railway-sim run \
     --config configs/ice1_80kmh.yml \
     --output-dir results/ice1_80
   ```

   and optionally:

   ```bash
   railway-sim parametric \
     --base-config configs/ice1_80kmh.yml \
     --speeds "320:0.2,200:0.4,120:0.4" \
     --quantity Impact_Force_MN \
     --output-dir results_parametric/track_mix \
     --prefix track_mix \
     --ascii-plot
   ```

> ⚠️ Do **not** try to install the `[ui]` extra on Termux: `pyarrow` will attempt to build the full Apache Arrow C++ stack and will most likely fail or be extremely slow on a phone.

---

## 8. Development

To work on the code base:

```bash
git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install in editable mode with core dependencies
pip install --no-build-isolation -e .
```

You can then:

- Modify files under `src/railway_simulator/`
- Run the CLI (`railway-sim`) and your changes will be picked up automatically
- Add / update example configs under `configs/`

---

## 9. License

This project is released under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for full text.

---

## 10. Citation

If you use this simulator in academic work, you may cite it along with the DZSF research report on which the model is based.  
(Insert your preferred citation / DOI here once available.)
