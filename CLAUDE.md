# CLAUDE.md - AI Assistant Guide for Railway Impact Simulator

**Last Updated**: 2025-12-01
**Repository**: railway-impact-simulator
**Version**: 0.2.0
**Python**: ≥3.10

---

## 1. Repository Overview

### Purpose
This is an **HHT-α railway impact simulator** with Bouc-Wen hysteresis, implementing dynamic load models from DZSF Bericht 53 (2024). It simulates train impacts on rigid obstacles using a configurable, vectorized mass-spring model.

### Key Capabilities
- **CLI-based**: `railway-sim` command with `run` and `parametric` subcommands
- **Parametric studies**: Multi-speed envelope analysis
- **Multiple physics models**: Contact models, friction laws, hysteresis
- **Headless-friendly**: ASCII plotting for SSH/Termux environments
- **Optional UI**: Streamlit-based web interface
- **Research-grade**: Validated against Pioneer full-scale crash test

### Academic Context
- **Methodology**: Based on DZSF Bericht 53 (2024) - German railway research
- **License**: MIT (software) + CC BY 4.0 (methodology attribution)
- **DOI**: 10.48755/dzsf.240006.01
- **Use Case**: Research/parametric studies, NOT production design or Eurocode compliance

---

## 2. Codebase Structure

### Directory Layout
```
railway-impact-simulator/
├── src/railway_simulator/          # Main package
│   ├── __init__.py                 # Version export only
│   ├── cli.py                      # Typer-based CLI (838 lines)
│   └── core/                       # Core simulation engine
│       ├── engine.py               # Numerical solver (1364 lines) ⭐ KEY
│       ├── parametric.py           # Multi-scenario orchestration
│       ├── report.py               # PDF generation
│       └── app.py                  # Streamlit UI (optional dependency)
│
├── configs/                        # YAML configuration examples
│   ├── ice1_80kmh.yml             # Minimal override (11 lines)
│   ├── ice1_aluminum.yml          # Soft crushing (fy=8 MN)
│   ├── ice1_steel.yml             # Stiff crushing (fy=18 MN)
│   ├── generic_passenger.yml      # Generic European car
│   └── traxx_freight.yml          # Freight locomotive
│
├── examples/                       # Python API examples
│   ├── ice1_aluminum_minimal.py   # Single-run example
│   └── parametric_line_mix.py     # Parametric API example
│
├── README.md                       # User documentation
├── PROJECT_SUMMARY.md              # Developer overview
├── VALIDATION_Pioneer.md           # Validation methodology
├── CITATION_REFERENCE.md           # Academic citation guide
├── LICENSE                         # MIT License
└── pyproject.toml                  # Build config + dependencies
```

### Key Files Priority
When working on this codebase, read in this order:
1. **`README.md`**: User-facing documentation, CLI usage
2. **`src/railway_simulator/core/engine.py`**: Core numerical implementation
3. **`src/railway_simulator/cli.py`**: CLI interface and orchestration
4. **`configs/ice1_80kmh.yml`**: Minimal config example
5. **`examples/ice1_aluminum_minimal.py`**: Simplest API usage
6. **`PROJECT_SUMMARY.md`**: High-level architecture
7. **`VALIDATION_Pioneer.md`**: Physics validation context

---

## 3. Core Modules and Responsibilities

### 3.1 `src/railway_simulator/core/engine.py` (⭐ MOST IMPORTANT)

**Purpose**: Pure numerical computation engine, UI-agnostic

**Key Components**:

#### Data Structures
- **`SimulationParams`** (dataclass): 40+ fields defining entire simulation
  - Geometry: `n_masses`, `masses`, `x_init`, `y_init`
  - Kinematics: `v0_init`, `angle_rad`, `d0`
  - Material: `fy` (yield forces), `uy` (yield deformations)
  - Contact: `k_wall`, `cr_wall`, `contact_model`
  - Integration: `alpha_hht`, `h_init`, `T_max`, `newton_tol`, `max_iter`
  - Bouc-Wen: `bw_a`, `bw_A`, `bw_beta`, `bw_gamma`, `bw_n`
  - Friction: `friction_model`, `mu_s`, `mu_k`, `sigma_0`, `sigma_1`, `sigma_2`
  - Building SDOF: `building_enable`, `building_mass`, `building_zeta`, etc.

- **`TrainConfig`** (dataclass): Helper for building train configurations

#### Physics Models
- **`BoucWenModel`**: Bouc-Wen hysteresis with RK4 integration
- **`FrictionModels`**: Coulomb, Stribeck, LuGre, Dahl, Brown-McPhee
- **`ContactModels`**: Linear penalty, Hunt-Crossley, Lankarani-Nikravesh, Anagnostopoulos
- **`StructuralDynamics`**: Building/pier SDOF with Takeda hysteresis

#### Numerical Solver
- **`HHTAlphaIntegrator`**: HHT-α implicit time integration
  - **Method**: Picard fixed-point iteration (NOT full Newton-Raphson)
  - **Convergence**: `newton_tol` (default: 1e-4), `max_iter` (default: 50)
  - **Damping**: `alpha_hht` typically in [-0.3, 0] for numerical stability

#### Main Simulator
- **`ImpactSimulator`**: Orchestrates entire simulation
  - Assembles mass matrix, damping matrix, stiffness matrix
  - Handles contact force computation
  - Computes friction forces
  - Tracks energy balance (kinetic, spring, contact, damping, friction)
  - Returns comprehensive DataFrame with 20+ columns

#### Public API Functions
```python
get_default_simulation_params() -> dict
    # Returns baseline 40t passenger car at 80 km/h

run_simulation(params: dict | SimulationParams) -> pd.DataFrame
    # Main entry point - accepts dict or dataclass
    # Returns DataFrame with time history and energy tracking

_coerce_scalar_types_for_simulation(params_dict: dict) -> dict
    # Type normalization for YAML inputs
    # Handles scientific notation, underscored numbers, arrays
```

### 3.2 `src/railway_simulator/cli.py`

**Purpose**: Command-line interface using Typer framework

**Commands**:

#### `railway-sim run`
Single impact simulation
- **Input**: `--config` (YAML/JSON path)
- **Overrides**: `--speed-kmh`, `--v0-init`
- **Output**: CSV, log, optional PDF
- **Visualization**: `--ascii-plot`, `--plot`, `--pdf-report`

#### `railway-sim parametric`
Speed-based parametric study
- **Input**: `--base-config`, `--speeds "320:0.2,200:0.4,120:0.4"`
- **Quantity**: `--quantity Impact_Force_MN` (column to envelope)
- **Output**: Envelope CSV, summary CSV, log, optional PDF

**Helper Functions**:
- `_load_config()`: YAML/JSON loading
- `_parse_speeds_spec()`: Parse "speed:weight" strings
- `_ascii_plot()`: 70×20 ASCII terminal plotting
- `_compute_single_run_performance()`: FLOP estimates, time step stats
- `_compute_parametric_performance()`: Aggregate metrics

### 3.3 `src/railway_simulator/core/parametric.py`

**Purpose**: Multi-scenario orchestration

**Key Components**:
- **`ScenarioDefinition`** (dataclass): Name, params, weight, metadata
- **`run_parametric_envelope()`**:
  - Runs multiple scenarios with different speeds
  - Computes envelope (max at each time point)
  - Computes weighted mean history
  - Validates time grid consistency
  - Returns: envelope DataFrame, summary DataFrame, metadata

### 3.4 `src/railway_simulator/core/report.py`

**Purpose**: PDF generation using matplotlib PdfPages backend

**Functions**:
- `generate_single_run_report()`: 3-page PDF with config, performance, force/penetration, energies
- `generate_parametric_report()`: 3-page PDF with performance, envelope plot, scenario table
- Cross-platform file opening (macOS/Windows/Linux)

### 3.5 `src/railway_simulator/core/app.py`

**Purpose**: Streamlit-based interactive UI (optional dependency)

**Features**:
- Header with institutional logos (KIT, EBA, DZSF)
- Research background and citation information
- Interactive parameter controls
- Plotly-based visualizations
- Real-time simulation and plotting

**Dependencies**: streamlit, plotly, pyarrow (NOT available on Termux)

---

## 4. Configuration System

### Philosophy
1. **Defaults**: `get_default_simulation_params()` provides baseline
2. **YAML Override**: Config files override specific parameters only
3. **Type Coercion**: Automatic normalization of types from YAML
4. **CLI Override**: Command-line flags can override config

### Configuration File Structure

**Minimal Example** (`configs/ice1_80kmh.yml`):
```yaml
v0_init: -22.22        # ≈ 80 km/h towards wall (negative = towards barrier)
T_max: 0.40            # simulation duration [s]
alpha_hht: -0.15       # HHT-α parameter (numerical damping)
contact_model: "lankarani-nikravesh"
k_wall: 60_000_000.0   # 60 MN/m
cr_wall: 0.8           # coefficient of restitution
```

**Full Example** (`configs/ice1_aluminum.yml` - 99 lines):
- Complete train geometry (7 masses)
- Full Bouc-Wen parameters
- Friction model configuration
- Integration parameters

### Available Configs
1. **`ice1_80kmh.yml`**: Minimal override (11 lines) - start here
2. **`ice1_aluminum.yml`**: Soft aluminum-like crushing (fy=8 MN, uy=100 mm)
3. **`ice1_steel.yml`**: Stiff steel S355-like crushing (fy=18 MN, uy=40 mm)
4. **`generic_passenger.yml`**: Generic European car (~60t)
5. **`traxx_freight.yml`**: Freight locomotive (~90t)

### Critical Parameters

**Sign Conventions** (IMPORTANT):
- `v0_init < 0`: Velocity towards the wall (barrier)
- Example: `-22.22 m/s` ≈ 80 km/h impact
- `angle_rad = 0`: Perpendicular impact

**Units** (SI + practical):
- Masses: kg
- Positions: m
- Forces: N (output often in MN)
- Time: s (output often in ms)
- Velocity: m/s
- Acceleration: m/s² (output often in g)

**Contact Models**:
- `"linear"`: Simple penalty
- `"hunt-crossley"`: Nonlinear with velocity damping
- `"lankarani-nikravesh"`: Hunt-Crossley variant (DEFAULT)
- `"anagnostopoulos"`: Gap element

**Friction Models**:
- `"coulomb"`: Simple Coulomb friction
- `"stribeck"`: Stribeck effect
- `"lugre"`: LuGre dynamic friction (DEFAULT)
- `"dahl"`: Dahl model
- `"brown-mcphee"`: Brown-McPhee variant

**HHT-α Integration**:
- `alpha_hht`: Typically in [-0.3, 0] for numerical damping
- Default: -0.10 to -0.15
- More negative = more damping

### Type Handling
YAML supports flexible numeric notation:
- Scientific: `8.0e6` → `8000000.0`
- Underscored: `60_000_000` → `60000000`
- Arrays: Lists → `np.ndarray`

---

## 5. Development Workflows

### 5.1 Environment Setup

**Standard Installation**:
```bash
git clone https://github.com/sflabbe/railway-impact-simulator.git
cd railway-impact-simulator
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install --no-build-isolation -e .  # Editable mode
```

**With UI Dependencies**:
```bash
pip install --no-build-isolation -e ".[ui]"
```

**Termux/Android** (core only):
```bash
pkg install python python-pip numpy scipy pandas matplotlib
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --no-build-isolation .  # NOT editable, NO [ui]
```

### 5.2 Common Development Tasks

#### Running Single Simulation
```bash
# Using CLI
railway-sim run --config configs/ice1_80kmh.yml --output-dir results/test

# Using Python API
python examples/ice1_aluminum_minimal.py
```

#### Running Parametric Study
```bash
railway-sim parametric \
  --base-config configs/ice1_80kmh.yml \
  --speeds "320:0.2,200:0.4,120:0.4" \
  --quantity Impact_Force_MN \
  --output-dir results/parametric
```

#### Modifying Core Engine
1. Edit `src/railway_simulator/core/engine.py`
2. Changes are immediately available (editable install)
3. Test with CLI: `railway-sim run --config configs/ice1_80kmh.yml`

#### Adding New Configuration
1. Create new YAML in `configs/`
2. Start from `configs/ice1_80kmh.yml` (minimal)
3. Override only necessary parameters
4. Test: `railway-sim run --config configs/your_new_config.yml`

#### Debugging
- **Log files**: Check `{output_dir}/{prefix}_run.log`
- **Energy balance**: Check `E_balance_error_J` column in output CSV
- **Convergence**: Look for "Max iterations reached" warnings in log
- **ASCII plot**: Add `--ascii-plot` to CLI for quick visual check

### 5.3 Output Files

**Single Run**:
- `{prefix}_results.csv`: Time history with 20+ columns
- `{prefix}_run.log`: Detailed execution log
- `{prefix}_report.pdf`: Optional 3-page report

**Parametric Study**:
- `{prefix}_{quantity}_envelope.csv`: Envelope time history
- `{prefix}_{quantity}_summary.csv`: Per-scenario summary
- `{prefix}_{quantity}_parametric.log`: Execution log
- `{prefix}_{quantity}_report.pdf`: Optional report

### 5.4 Git Workflow

**Current Branch**: `claude/claude-md-mimz4fgex6gkn0oy-01RuwDtdJ75vo4W5J7umrkED`

**Committing Changes**:
1. Stage changes: `git add <files>`
2. Commit with clear message: `git commit -m "Description"`
3. Push to feature branch: `git push -u origin <branch-name>`

**Important**: Always push to branches starting with `claude/` and ending with matching session ID.

---

## 6. Testing and Validation

### Current Approach
**NO formal test suite** - validation-based QA instead

**Validation Strategy**:
1. **Physics validation**: `VALIDATION_Pioneer.md` documents comparison with full-scale Pioneer crash test
2. **Energy balance checking**: All simulations track energy conservation error
3. **Example scripts**: Serve as smoke tests
4. **Parametric consistency**: Envelope validation checks time grid alignment

### Quality Checks
When modifying code:
1. Run example scripts: `python examples/ice1_aluminum_minimal.py`
2. Check CLI: `railway-sim run --config configs/ice1_80kmh.yml`
3. Verify energy balance: `E_balance_error_J` should be small (< 1% of total)
4. Check convergence: No "Max iterations reached" warnings

### Future Testing
If adding tests:
- Framework: pytest (already in `.gitignore`)
- Location: `tests/` directory (create if needed)
- Focus areas:
  - Configuration loading and validation
  - Type coercion logic
  - Physics model implementations
  - Energy conservation

---

## 7. Dependencies and Environment

### Core Dependencies (Always Required)
```toml
numpy>=1.24          # Numerical arrays
scipy>=1.10          # Scientific computing, constants
matplotlib>=3.7      # Plotting, PDF backend
pandas>=2.0          # DataFrames for results
typer[all]>=0.12     # CLI framework
PyYAML>=6.0          # Config parsing
plotly>=5.0          # Interactive plots
```

### Optional Dependencies (UI Extra)
```toml
streamlit>=1.35      # Web UI
pyarrow<22,>=7.0     # Arrow data format (Streamlit dep)
```

### Platform Constraints
- **Linux/macOS/WSL**: Full support
- **Windows**: Full support
- **Android/Termux**: Core only (pyarrow compilation fails)

### Python Version
- **Minimum**: Python 3.10
- **Reason**: Dataclass features, type hints

---

## 8. Coding Conventions and Patterns

### 8.1 Code Organization Principles

**Separation of Concerns**:
- `engine.py`: Pure numerical computation, no I/O, no UI
- `cli.py`: Command-line interface, orchestration
- `parametric.py`: Multi-scenario logic
- `report.py`: PDF generation
- `app.py`: Web UI (completely isolated)

**Dataclass-Heavy Design**:
- Use dataclasses for all parameter structures
- Type hints everywhere
- Explicit field defaults

### 8.2 Numerical Conventions

**Energy Tracking** (critical for validation):
All simulations track:
- `E_kin_J`: Kinetic energy
- `E_spring_J`: Bouc-Wen spring energy
- `E_contact_J`: Contact/wall energy
- `E_damp_rayleigh_J`: Rayleigh damping dissipation
- `E_friction_J`: Friction dissipation
- `E_balance_error_J`: Conservation error (should be small)

**DataFrame Output Structure**:
- Time columns: `Time_s`, `Time_ms`
- Impact columns: `Impact_Force_MN`, `Penetration_mm`
- Per-mass kinematics: `M{i}_X_m`, `M{i}_V_ms`, `M{i}_A_ms2`, `M{i}_A_g`
- Energy columns: Various `E_*_J`
- Building (if enabled): `Building_X_m`, `Building_V_ms`, etc.

**DataFrame Metadata** (stored in `df.attrs`):
- `n_lu`: Number of LU decompositions
- `n_dof`: Degrees of freedom

### 8.3 Documentation Standards

**Docstring Style**:
- Module-level: Purpose and high-level usage
- Class-level: Responsibilities and key attributes
- Function-level: Args, Returns, Examples (where helpful)

**Comments**:
- Explain WHY, not WHAT
- Physics equations: Include references to DZSF report
- Numerical stability: Document assumptions

### 8.4 Performance Patterns

**Vectorization**:
- Use NumPy operations over loops where possible
- Mass matrix, damping matrix operations are vectorized

**Logging Performance**:
Always log:
- Wall-clock time
- Time steps and Δt statistics
- Real-time factor (simulated time / wall time)
- LU solve count estimates
- FLOP estimates: `(2/3) * n_dof³ * n_solves`

**ASCII Plotting**:
- Use for headless environments (SSH, Termux)
- 70×20 grid is standard
- Helps catch obvious errors without opening files

---

## 9. Common Pitfalls and Best Practices

### 9.1 Configuration Pitfalls

**DON'T**:
- Add unknown keys to YAML (will raise TypeError)
- Wrap parameters in extra levels (e.g., `scenario:`, `output:`)
- Use positive `v0_init` when you mean towards the wall

**DO**:
- Start from existing configs (`ice1_80kmh.yml` is minimal)
- Override only necessary parameters
- Check units carefully (SI base units)
- Use underscores in large numbers for readability: `60_000_000`

### 9.2 Development Pitfalls

**DON'T**:
- Install `[ui]` extra on Termux (pyarrow will fail)
- Modify `engine.py` to add I/O or UI code (keep it pure)
- Skip energy balance checking after changes
- Use `git commit --amend` without checking authorship

**DO**:
- Use editable install: `pip install -e .`
- Test CLI after engine changes
- Check log files for convergence issues
- Verify energy balance: `E_balance_error_J` column
- Use `--ascii-plot` for quick visual validation

### 9.3 Numerical Pitfalls

**Convergence Issues**:
- If "Max iterations reached" warnings appear:
  - Reduce `h_init` (time step)
  - Increase `max_iter` (default: 50)
  - Adjust `newton_tol` (default: 1e-4)
  - Check for unrealistic parameters (e.g., very stiff walls with soft trains)

**Energy Balance**:
- `E_balance_error_J` should be < 1% of initial energy
- Large errors indicate:
  - Convergence failures
  - Numerical instability
  - Bug in physics model

**Contact Models**:
- `"lankarani-nikravesh"` is default and well-tested
- Other models may need different `k_wall` values
- Always validate against known results when switching models

### 9.4 File Management

**Output Organization**:
- Use `--output-dir` to organize results
- Use `--prefix` for meaningful filenames
- Log files are critical for debugging

**Don't Commit**:
- Result CSV files (`results/`)
- PDF reports
- Log files
- `.pytest_cache/`
- `__pycache__/`
- `.ipynb_checkpoints/`

---

## 10. Key Decision Points for AI Assistants

### When Should You Use the CLI vs. Python API?

**Use CLI** (`railway-sim`) when:
- Running standard single/parametric studies
- User wants quick results without coding
- Output to CSV/PDF is sufficient
- Headless environment (SSH, Termux)

**Use Python API** (`run_simulation()`) when:
- Custom post-processing needed
- Integration into larger workflow
- Programmatic parameter sweeps
- Custom plotting or analysis

### When Should You Modify Which Module?

**Modify `engine.py`** when:
- Adding new physics models (friction, contact, hysteresis)
- Changing numerical solver
- Adding new output quantities
- Fixing numerical bugs

**Modify `cli.py`** when:
- Adding new CLI commands or flags
- Changing output file formats
- Adding new plotting options
- Improving error messages

**Modify `parametric.py`** when:
- Changing envelope computation logic
- Adding new scenario types (beyond speed variations)
- Modifying summary statistics

**Modify `report.py`** when:
- Changing PDF layout or content
- Adding new plots to reports

**DON'T modify `app.py`** unless:
- User explicitly requests Streamlit UI changes
- And they have `[ui]` dependencies installed

### When Should You Create New Files?

**Create new config** when:
- User wants new train/impact scenario
- Different material properties needed
- Location: `configs/{descriptive_name}.yml`

**Create new example** when:
- Demonstrating new API usage pattern
- Location: `examples/{descriptive_name}.py`

**DON'T create**:
- New top-level Python modules (keep architecture simple)
- Documentation files (update existing README.md, etc.)
- Test files (unless user explicitly wants test suite)

---

## 11. Quick Reference

### Most Common Commands
```bash
# Single run
railway-sim run --config configs/ice1_80kmh.yml --output-dir results/test

# Parametric study
railway-sim parametric \
  --base-config configs/ice1_80kmh.yml \
  --speeds "320:0.2,200:0.4,120:0.4" \
  --quantity Impact_Force_MN \
  --output-dir results/parametric

# With ASCII plot (headless)
railway-sim run --config configs/ice1_80kmh.yml --ascii-plot

# Override speed
railway-sim run --config configs/ice1_80kmh.yml --speed-kmh 120
```

### Most Common Python API
```python
from railway_simulator.core.engine import run_simulation, get_default_simulation_params

# Load defaults and override
params = get_default_simulation_params()
params['v0_init'] = -33.33  # 120 km/h
params['T_max'] = 0.50

# Run simulation
df = run_simulation(params)

# Check energy balance
print(f"Energy error: {df['E_balance_error_J'].iloc[-1]:.2f} J")

# Plot
import matplotlib.pyplot as plt
plt.plot(df['Time_ms'], df['Impact_Force_MN'])
plt.show()
```

### Critical Files to Check First
1. `README.md` - User documentation
2. `src/railway_simulator/core/engine.py` - Core implementation
3. `configs/ice1_80kmh.yml` - Simplest config example
4. `examples/ice1_aluminum_minimal.py` - Simplest API usage

### Key Columns in Output DataFrame
- **Time**: `Time_s`, `Time_ms`
- **Impact**: `Impact_Force_MN`, `Penetration_mm`
- **Kinematics**: `M{i}_X_m`, `M{i}_V_ms`, `M{i}_A_g`
- **Energy**: `E_kin_J`, `E_spring_J`, `E_contact_J`, `E_balance_error_J`

### Debugging Checklist
1. Check log file: `{output_dir}/{prefix}_run.log`
2. Check energy balance: `E_balance_error_J` column
3. Look for convergence warnings: "Max iterations reached"
4. Use ASCII plot: `--ascii-plot` flag
5. Reduce time step if unstable: `h_init: 1.0e-5`

---

## 12. Academic and Research Context

### Citation Requirements
- **Software**: MIT License (Sebastián Labbé, 2025)
- **Methodology**: DZSF Bericht 53 (Stempniewski et al., 2024)
- **License**: Methodology under CC BY 4.0
- **DOI**: 10.48755/dzsf.240006.01

### Use Case Boundaries
**Appropriate**:
- Parametric studies
- Research analysis
- Model validation
- Academic publications

**NOT appropriate**:
- Production design without validation
- Eurocode compliance certification
- Safety-critical decisions without expert review

### Validation Status
- Validated against Pioneer full-scale crash test
- See `VALIDATION_Pioneer.md` for methodology
- Energy balance provides internal consistency check

---

## 13. For AI Assistants: Communication Guidelines

### When Explaining Code
1. Reference line numbers: `engine.py:712`
2. Cite specific functions/classes
3. Explain physics context (this is research code)
4. Link to DZSF report concepts when relevant

### When Making Changes
1. ALWAYS read files before modifying
2. Test with `railway-sim run` after engine changes
3. Check energy balance after numerical modifications
4. Maintain separation of concerns (engine vs. CLI vs. UI)
5. Don't add features beyond user request (avoid over-engineering)

### When Debugging
1. Check log files first
2. Look for energy balance errors
3. Use ASCII plot for quick validation
4. Verify convergence (no "Max iterations" warnings)
5. Test with minimal config (`ice1_80kmh.yml`)

### When User Requests Are Unclear
Ask specifically about:
- Single run or parametric study?
- Which config to use as base?
- What quantity to track/envelope?
- Output format preferences (CSV, PDF, plot)?
- Platform (desktop, SSH, Termux)?

---

## 14. Appendix: Performance Expectations

### Typical Run Times (Desktop)
- **Single run** (0.4s simulation, 1e-4 time step): ~1-5 seconds wall time
- **Parametric (3 speeds)**: ~5-15 seconds wall time
- **Real-time factor**: 10-100× (100ms wall time per 1s simulated)

### Memory Usage
- **Single run**: < 100 MB
- **Parametric**: < 500 MB
- **Streamlit UI**: ~200-500 MB

### FLOP Estimates
- **Formula**: `(2/3) * n_dof³ * n_lu_solves`
- **Typical**: 10-100 MFLOP/s throughput
- Logged in performance output

### Termux Performance
- Expect 2-10× slower than desktop
- Use `--ascii-plot` instead of matplotlib GUI
- Core dependencies only (no UI)
- Suitable for parameter exploration, not heavy parametric studies

---

**END OF CLAUDE.MD**

For questions or issues, refer to the main repository: https://github.com/sflabbe/railway-impact-simulator
