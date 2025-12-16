# Railway Impact Simulator

HHT-α railway impact simulator with Bouc–Wen hysteresis, inspired by the dynamic load models discussed in **DZSF Bericht 53 (2024)** (“Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr”).

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

**HTTPS (recommended):**

```bash
git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator
```

**SSH (only if your SSH keys are set up):**

```bash
git clone git@github.com:sflabbe/railway-impact-simulator.git
cd railway-impact-simulator
```

If you get “**port 22: Network is unreachable**”, your network blocks SSH. Use HTTPS, or configure **SSH over port 443**:

```sshconfig
# ~/.ssh/config
Host github.com
  Hostname ssh.github.com
  Port 443
  User git
```

### 1.2 Create a virtual environment (recommended)

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

### 1.3 Install the core package

Recommended (build isolation on):

```bash
python -m pip install .
```

If you intentionally use `--no-build-isolation` (offline builds, controlled build env), **make sure `setuptools` is installed**:

```bash
python -m pip install --upgrade setuptools wheel
python -m pip install --no-build-isolation .
```

Check that the entry point is available:

```bash
railway-sim --help
```

---

## 2. Optional UI / heavy dependencies

The project defines an extra dependency group called `ui` (Streamlit + Plotly + friends).

Install the package with UI dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install ".[ui]"
```

> ⚠️ On Android / Termux this is usually **not recommended**: some packages may attempt to compile native components and can be slow or fail.

---

## 3. Quickstart

The repo ships example configuration files in `configs/`.

### 3.1 Single run

Run an included example config:

```bash
railway-sim run \
  --config configs/ice1_80kmh.yml \
  --output-dir results/ice1_80
```

Optional convenience flags:

- Override the impact speed (km/h): `--speed-kmh 120`
- Override initial velocity directly (m/s): `--v0-init -33.33`
- ASCII plot in terminal: `--ascii-plot`
- Pop up matplotlib plot: `--plot`
- Generate a PDF report (if enabled by the code): `--pdf-report`

Example:

```bash
railway-sim run \
  --config configs/ice1_80kmh.yml \
  --speed-kmh 80 \
  --output-dir results/ice1_80 \
  --ascii-plot
```

Output files typically include:

- `results.csv` (time history)
- `run.log` (log file)

### 3.2 Parametric study (speed mix envelope)

Example: speed mix **320 / 200 / 120 km/h** with weights **0.2 / 0.4 / 0.4** for the envelope of `Impact_Force_MN`:

```bash
railway-sim parametric \
  --base-config configs/ice1_80kmh.yml \
  --speeds "320:0.2,200:0.4,120:0.4" \
  --quantity Impact_Force_MN \
  --output-dir results_parametric/track_mix \
  --prefix track_mix \
  --ascii-plot
```

This will typically write:

- `..._envelope.csv` (envelope time history)
- `..._summary.csv` (peaks, scenario stats, …)
- log files with performance information

### 3.3 Studies (convergence / sensitivity)

The CLI also exposes study commands (run `railway-sim --help` for the full list).

**Time-step convergence:**

```bash
railway-sim convergence \
  --config configs/ice1_80kmh.yml \
  --dts "2e-4,1e-4,5e-5" \
  --quantity Impact_Force_MN \
  --out results_parametric/convergence
```

**Numerical sensitivity sweep (dt / alpha / Newton tolerance):**

```bash
railway-sim numerics-sensitivity \
  --config configs/ice1_80kmh.yml \
  --dts "2e-4,1e-4" \
  --alphas "0.05,0.10" \
  --tols "1e-6,1e-8" \
  --quantity Impact_Force_MN \
  --out results_parametric/numerics_sensitivity
```

**Fixed DIF (strain-rate proxy) by scaling a stiffness-like parameter (default: `k_wall`):**

```bash
railway-sim strain-rate-sensitivity \
  --config configs/ice1_80kmh.yml \
  --difs "1.0,1.1,1.2" \
  --k-path k_wall \
  --quantity Impact_Force_MN \
  --out results_parametric/strain_rate_fixed_dif
```

---

## 4. Streamlit server (Linux / macOS / Windows)

The Streamlit app entry point is:

- `src/railway_simulator/core/app.py`

1) Install UI extras:

```bash
python -m pip install ".[ui]"
```

2) Start the server:

```bash
streamlit run src/railway_simulator/core/app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501` in your browser.

To access it from another device on your LAN:

```bash
streamlit run src/railway_simulator/core/app.py --server.address 0.0.0.0 --server.port 8501
```

> ⚠️ Security note: binding to `0.0.0.0` exposes the server on your local network. Use this only on trusted networks.

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

> Note: if you set `T_max` and `h_init` but do not set `step`, the simulator derives a consistent `step ≈ T_max / h_init`.

---

## 6. Termux / Android notes

### 6.1 Core CLI on Termux (no UI extras)

A typical approach is to use Termux packages for the scientific stack (availability varies), then create a venv:

```bash
git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator

python -m venv .venv --system-site-packages
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation .
```

> ⚠️ Avoid installing `.[ui]` on pure Termux unless you know your device can build required native deps.

### 6.2 UI server via proot-distro (Debian on Termux)

If you want a better chance of running the Streamlit UI on Android, use a full Debian userland via **proot-distro**:

```bash
pkg update
pkg install -y proot-distro
proot-distro install debian
proot-distro login debian
```

Inside Debian:

```bash
apt update
apt install -y git python3 python3-venv python3-pip build-essential

git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install ".[ui]"
```

Then run Streamlit (same as desktop):

```bash
streamlit run src/railway_simulator/core/app.py --server.address 127.0.0.1 --server.port 8501
```

---

## 7. Development

Editable install (core):

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Then:

- edit code under `src/railway_simulator/`
- run `railway-sim ...` and your changes will be picked up

---

## 8. More docs in this repo

- `PROJECT_SUMMARY.md` – overview / roadmap
- `VALIDATION_Pioneer.md` – validation notes
- `CITATION_REFERENCE.md` – citation/reference notes

---

## 9. License

MIT License — see `LICENSE`.

---

## 10. Citation

If you use this simulator in academic work, cite the repository and the DZSF research report it is based on (see `CITATION_REFERENCE.md`).
