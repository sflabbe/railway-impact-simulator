# Railway Impact Simulator

HHT-α railway impact simulator with Bouc–Wen hysteresis, based on the dynamic load models from **DZSF Bericht 53 (2024)**.

The code implements a configurable, vectorized mass–spring model for train impacts on rigid obstacles and provides:

- A command line interface (`railway-sim`) for single runs and parametric studies
- CSV/NumPy output for post‑processing
- Optional ASCII plots for headless terminals (SSH, Termux, etc.)
- Optional UI / visualization extras based on Streamlit and Plotly

> ⚠️ The repository currently targets **Python ≥ 3.10**.  
> ⚠️ If you run very new Python versions (e.g. 3.13), some scientific wheels may not be available on all platforms yet.

---

## 1. Installation

### 1.1. Clone the repository

```bash
git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator
```

### 1.2. Create a virtual environment (recommended)

Linux / macOS / WSL:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 1.3. Install the core package

The **core** installation includes the numerical engine, CLI, and lightweight terminal plotting.

Recommended (build isolation on):

```bash
python -m pip install .
```

If you intentionally use `--no-build-isolation` (e.g., offline builds), make sure `setuptools` is installed in the environment (see above):

```bash
python -m pip install --no-build-isolation .
```

Check that the entry point is available:

```bash
railway-sim --help
```

You should see the CLI help with the available sub‑commands (e.g. `run`, `parametric`).

---

## 2. Optional UI / heavy dependencies

The project defines an extra dependency group called `ui` that pulls in heavier libraries such as **Streamlit** and **pyarrow** (used for dashboards / richer visualization).

Install the package **with** these optional UI dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install ".[ui]"
```

> 💡 On typical desktop Python distributions (Linux, Windows, macOS) this usually uses pre‑built wheels for `pyarrow`.  
> ⚠️ On Android / Termux this is **not recommended**: `pyarrow` may attempt to compile the Apache Arrow C++ stack and will likely fail or be very slow. On Termux, stick to the **core** installation (no `[ui]`).

### 2.1. Run the Streamlit UI server (Linux / macOS / Windows)

If you installed the `[ui]` extras, you can start the Streamlit dashboard locally.

1. Activate your environment

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Locate the Streamlit app entrypoint

A Streamlit app is a Python file you run with `streamlit run ...`. Common filenames are:

- `streamlit_app.py`
- `app.py`
- `ui/app.py`

If you are unsure, search for `import streamlit` in the repository:

```bash
python - <<'PY'
import pathlib
hits = []
for p in pathlib.Path(".").rglob("*.py"):
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if "import streamlit" in t or "from streamlit" in t:
        hits.append(str(p))
print("\n".join(hits) if hits else "No Streamlit entrypoint found.")
PY
```

3. Start the server

Replace `<path-to-streamlit-app>.py` with the file you found:

```bash
streamlit run <path-to-streamlit-app>.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501` in your browser.

If you want to access it from another device on your LAN, bind to all interfaces:

```bash
streamlit run <path-to-streamlit-app>.py --server.address 0.0.0.0 --server.port 8501
```

> ⚠️ Security note: binding to `0.0.0.0` exposes the server on your local network. Use this only on trusted networks.


---

## 3. Quickstart (single run)

The repository ships with example configuration files in `configs/`.  
A good starting point is an **ICE‑1 middle car at 80 km/h** impacting a rigid obstacle.

From the repository root, with the virtual environment activated:

```bash
railway-sim run   --config configs/ice1_80kmh.yml   --output-dir results/ice1_80
```

This will typically:

- Load the YAML configuration `configs/ice1_80kmh.yml`
- Run a single HHT‑α + Bouc–Wen simulation
- Write a time history to `results/ice1_80/results.csv`
- Create a log file at `results/ice1_80/run.log`

The CSV typically contains columns such as:

- `Time_s`, `Time_ms`
- kinematic quantities for each mass
- contact/impact force (e.g. `Impact_Force_MN`, depending on the config/output settings)

You can post‑process the CSV with your tool of choice (Pandas, Excel, MATLAB, etc.).

---

## 4. Parametric study example

The CLI supports parametric studies where a base configuration is re‑run for multiple speeds and combined into an envelope.

Example: a **track mix** with 320 / 200 / 120 km/h, with usage weights 0.2 / 0.4 / 0.4, for the envelope of `Impact_Force_MN`:

```bash
railway-sim parametric   --base-config configs/ice1_80kmh.yml   --speeds "320:0.2,200:0.4,120:0.4"   --quantity Impact_Force_MN   --output-dir results_parametric/track_mix   --prefix track_mix
```

This will typically:

- build three scenarios:
  - `v320`: 320 km/h, weight 0.2
  - `v200`: 200 km/h, weight 0.4
  - `v120`: 120 km/h, weight 0.4
- run all simulations
- write the envelope time history to  
  `results_parametric/track_mix/track_mix_Impact_Force_MN_envelope.csv`
- write a scenario summary (peaks, etc.) to  
  `results_parametric/track_mix/track_mix_Impact_Force_MN_summary.csv`

---

## 5. ASCII plots (headless / Termux / SSH)

For headless environments you can enable an ASCII plot directly in the terminal.  
This is useful on:

- SSH sessions without X forwarding
- Android / Termux
- minimal containers

Add the `--ascii-plot` flag:

```bash
railway-sim run   --config configs/ice1_80kmh.yml   --output-dir results/ice1_80_ascii   --ascii-plot
```

…and for a parametric envelope:

```bash
railway-sim parametric   --base-config configs/ice1_80kmh.yml   --speeds "320:0.2,200:0.4,120:0.4"   --quantity Impact_Force_MN   --output-dir results_parametric/track_mix_ascii   --prefix track_mix   --ascii-plot
```

At the end of the run, the CLI prints an ASCII envelope such as:

```text
ASCII envelope plot (Impact_Force_MN vs Time_ms):
            ***
           *  *
          *    *
        **      *
      ***        *
    ***           *
***                ***************
# Time [ms] (0 – 400)
```

(Shape and scale depend on the configuration; this is a schematic example.)

---

## 6. Configuration files

Configuration is provided via **YAML** (or JSON) files passed to `--config` / `--base-config`.

Internally, the CLI does roughly:

1. load default parameters (e.g. via `get_default_simulation_params()`)
2. load your config file into a flat dictionary
3. override defaults with values found in the config
4. create a `SimulationParams` object and run the solver

Notes:

- Keys in your YAML must match the fields of `SimulationParams`.  
  Unknown keys typically raise errors like  
  `SimulationParams.__init__() got an unexpected keyword argument 'foo'`.
- The example configs under `configs/` are the best reference for valid keys and units.
- Avoid wrapping parameters in extra levels like `scenario:` or `output:` unless your loader explicitly supports it.

Minimal example (illustrative only — follow the names used in `configs/`):

```yaml
# quickstart.yml
v0_init: -22.22   # [m/s] initial train velocity (sign convention from the model)
T_end: 0.40       # [s] total simulated time (exact key name depends on SimulationParams)
```

---

## 7. Termux / Android notes

It is possible to run the simulator on Android via **Termux**.

Recommended approach:

1. Install Python and a scientific stack via Termux packages **where possible** (varies by device/repo), then create a venv:

   ```bash
   git clone https://github.com/sflabbe/railway-impact-simulator.git
   cd railway-impact-simulator

   python -m venv .venv --system-site-packages
   source .venv/bin/activate
   ```

2. Install the **core** package only (no UI extras):

   ```bash
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install --no-build-isolation .
   ```

> ⚠️ Do **not** try to install the `[ui]` extra on Termux: `pyarrow` may attempt a native build and will likely fail or be extremely slow on a phone.

### 7.1. Running the UI server via proot-distro (Debian on Termux)

If you want to run the **UI / dashboard server** on Android, a practical approach is to use a full Debian userland via **proot-distro** (often called “proot-debian”). This can improve compatibility for heavier Python packages compared to running everything directly in Termux.

1. Install and set up Debian:

   ```bash
   pkg update
   pkg install -y proot-distro
   proot-distro install debian
   proot-distro login debian
   ```

2. Inside Debian, install dependencies:

   ```bash
   apt update
   apt install -y git python3 python3-venv python3-pip build-essential
   ```

3. Clone the repo (either in Debian, or into your Termux home and access it from Debian):

   ```bash
   git clone https://github.com/sflabbe/railway-impact-simulator.git
   cd railway-impact-simulator
   ```

4. Create a venv and install the UI extras:

   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install ".[ui]"
   ```

5. Start the server.

   - If the repository includes a Streamlit app entrypoint (look for a file you run with `streamlit run ...`), start it like this:

     ```bash
     streamlit run <path-to-streamlit-app>.py --server.address 127.0.0.1 --server.port 8501
     ```

   - If you want to access it from other devices on the same Wi‑Fi (LAN), bind to all interfaces:

     ```bash
     streamlit run <path-to-streamlit-app>.py --server.address 0.0.0.0 --server.port 8501
     ```

   Then open `http://127.0.0.1:8501` on the phone (or `http://<phone-ip>:8501` from another device on your LAN).

> ⚠️ Security note: binding to `0.0.0.0` exposes the server on your local network. Use this only on trusted networks.


---

## 8. Development

To work on the code base:

```bash
git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install in editable mode (core)
python -m pip install --no-build-isolation -e .
```

You can then:

- modify files under `src/railway_simulator/`
- run the CLI (`railway-sim`) and your changes will be picked up automatically
- add/update example configs under `configs/`

---

## 9. License

This project is released under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for full text.

---

## 10. Citation

If you use this simulator in academic work, you may cite it along with the DZSF research report on which the model is based.

(Insert your preferred citation / DOI here once available.)
