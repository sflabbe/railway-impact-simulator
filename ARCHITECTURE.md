# Architecture Documentation

This document describes the software architecture of the Railway Impact Simulator, including system design, component interactions, and data flow.

## Table of Contents

- [System Overview](#system-overview)
- [Architecture Principles](#architecture-principles)
- [Component Diagram](#component-diagram)
- [Module Architecture](#module-architecture)
- [Data Flow](#data-flow)
- [Physics Engine](#physics-engine)
- [Extensibility](#extensibility)

---

## System Overview

The Railway Impact Simulator is a scientific computing application for analyzing railway vehicle collisions with deformable structures. It implements implicit time integration (HHT-α method) with nonlinear contact, friction, and hysteresis models.

### Key Features

- **Implicit Time Integration:** HHT-α method with Newton-Raphson solver
- **Multiple Contact Models:** 9 validated contact force formulations
- **Advanced Friction:** LuGre, Dahl, Coulomb-Stribeck, Brown-McPhee models
- **Hysteresis:** Bouc-Wen model for crushing structures
- **Energy Tracking:** Comprehensive energy balance verification
- **Multi-Interface:** CLI, Web UI (Streamlit), Python API

### Target Users

- Railway safety engineers
- Crashworthiness researchers
- Structural dynamics analysts
- Academic researchers

---

## Architecture Principles

### 1. Separation of Concerns

```
┌─────────────────────────────────────────────────┐
│                  Interfaces                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   CLI    │  │   Web UI │  │  Python  │      │
│  │  (Typer) │  │(Streamlit)│  │   API    │      │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘      │
└────────┼─────────────┼─────────────┼────────────┘
         │             │             │
         └─────────────┼─────────────┘
                       │
         ┌─────────────▼─────────────┐
         │      Core Engine          │
         │  (UI-agnostic logic)      │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼─────────────┐
         │   Physics Models          │
         │ (Contact, Friction, etc.)  │
         └───────────────────────────┘
```

**Key Points:**
- Core engine is completely UI-agnostic
- All interfaces (CLI, UI, API) use the same core
- Physics models are modular and swappable

### 2. Modularity

Each physics model is in a separate module:

```
src/railway_simulator/core/
├── engine.py         # Main orchestrator
├── contact.py        # Contact models (independent)
├── friction.py       # Friction models (independent)
├── integrator.py     # Time integration (independent)
└── parametric.py     # Multi-scenario studies
```

### 3. Testability

- Pure functions where possible
- Dependency injection for configurable components
- Comprehensive unit and integration tests
- Energy balance verification in every simulation

### 4. Performance

- NumPy vectorization for all array operations
- Parallel execution for parametric studies
- Efficient sparse matrix operations (where applicable)
- Performance counter tracking (`n_lu`, FLOPS estimation)

---

## Component Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      Railway Impact Simulator                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    User Interfaces                        │  │
│  │                                                            │  │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │  CLI    │  │ Streamlit│  │  Jupyter │  │  Python  │  │  │
│  │  │ (Typer) │  │  Web UI  │  │ Notebooks│  │   API    │  │  │
│  │  └────┬────┘  └─────┬────┘  └────┬─────┘  └────┬─────┘  │  │
│  └───────┼─────────────┼────────────┼─────────────┼─────────┘  │
│          │             │            │             │             │
│  ┌───────┴─────────────┴────────────┴─────────────┴─────────┐  │
│  │                   Core Simulation Layer                   │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │  ImpactSimulator (Main Orchestrator)              │    │  │
│  │  │  - Assembles system matrices                      │    │  │
│  │  │  - Coordinates time stepping                      │    │  │
│  │  │  - Tracks energy balance                          │    │  │
│  │  └──┬───────────────────────────────────────────┬───┘    │  │
│  │     │                                           │        │  │
│  │  ┌──▼────────────┐  ┌─────────────┐  ┌────────▼──────┐ │  │
│  │  │ TrainBuilder  │  │  Structural │  │    Bouc-Wen   │ │  │
│  │  │ - Geometry    │  │  Dynamics   │  │   Hysteresis  │ │  │
│  │  │ - Mass matrix │  │ - Stiffness │  │              │  │  │
│  │  └───────────────┘  │ - Damping   │  └───────────────┘ │  │
│  │                     └─────────────┘                    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                  Physics Models Layer                   │  │
│  │                                                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │ ContactModels│  │FrictionModels│  │HHTAlphaInteg.│  │  │
│  │  │              │  │              │  │              │  │  │
│  │  │ 9 models:    │  │ 4 models:    │  │ - Predictor  │  │  │
│  │  │ - Hooke      │  │ - Coulomb    │  │ - Corrector  │  │  │
│  │  │ - Hertz      │  │ - LuGre      │  │ - Accel. eq. │  │  │
│  │  │ - Hunt-Cross.│  │ - Dahl       │  │              │  │  │
│  │  │ - Lankarani  │  │ - Brown-McPh.│  │              │  │  │
│  │  │ - ... etc    │  │              │  │              │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                   Support Modules                       │  │
│  │                                                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │  │
│  │  │Parametric│  │ Studies  │  │ Export   │  │ Report │  │  │
│  │  │ Envelope │  │ - Converg│  │ - CSV    │  │ - PDF  │  │  │
│  │  │          │  │ - Sensit.│  │ - Excel  │  │ - Plots│  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### Core Engine (`engine.py`)

**Responsibilities:**
- System assembly (mass, stiffness, damping matrices)
- Time-stepping coordination
- Newton-Raphson nonlinear solver
- Energy balance tracking
- Result DataFrame construction

**Key Classes:**

```
SimulationParams
├── Physical parameters (masses, lengths, velocities)
├── Material parameters (stiffness, friction, etc.)
├── Numerical parameters (dt, alpha_hht, tolerances)
└── Model selection (contact_model, friction_model)

ImpactSimulator
├── __init__(params: SimulationParams)
├── run() -> pd.DataFrame
│   ├── initialize_state()
│   ├── time_loop()
│   │   ├── assemble_forces()
│   │   ├── newton_solver() or picard_solver()
│   │   └── update_state()
│   └── build_results_dataframe()
├── assemble_mass_matrix()
├── assemble_stiffness_matrix()
└── track_energy()
```

**Dependencies:**
- `ContactModels` (from `contact.py`)
- `FrictionModels` (from `friction.py`)
- `HHTAlphaIntegrator` (from `integrator.py`)
- `BoucWenModel` (hysteresis, defined locally)

### Contact Models (`contact.py`)

**Design Pattern:** Strategy pattern with static methods

```
ContactModels
├── MODELS: Dict[str, Callable]
│   ├── "hooke"                → lambda(k, d, cr, dv, v0)
│   ├── "hertz"                → lambda(k, d, cr, dv, v0)
│   ├── "hunt-crossley"        → lambda(k, d, cr, dv, v0)
│   └── ... (9 total models)
├── compute_force(u, du, v0, k, cr, model) -> np.ndarray
├── list_models() -> list[str]
└── get_model_info(model) -> str
```

**Key Algorithm:**
```python
def compute_force(u_contact, du_contact, v0, k_wall, cr_wall, model):
    1. Convert penetration u to depth δ = -u
    2. Identify active contacts (mask where δ > 0)
    3. For active contacts:
       - Prevent v0 division by zero
       - Look up model function
       - Compute raw force
       - Enforce unilateral constraint (clip to compression)
    4. Return force array
```

### Friction Models (`friction.py`)

**Design Pattern:** Static methods with optional internal state

```
FrictionModels
├── lugre(z_prev, v, F_c, F_s, v_s, σ0, σ1, σ2, h) -> (F, z_new)
│   └── Uses closed-form ODE solution for stability
├── dahl(z_prev, v, F_c, σ0, h) -> (F, z_new)
│   └── Simplified dynamic friction
├── coulomb_stribeck(v, Fc, Fs, vs, Fv) -> F
│   └── Stateless friction
└── brown_mcphee(v, Fc, Fs, vs) -> F
    └── Smooth approximation
```

**State Management:**
- LuGre and Dahl: Maintain internal state `z` per DOF
- Coulomb and Brown-McPhee: Stateless (function of velocity only)

### Time Integration (`integrator.py`)

**Design Pattern:** Object-oriented with state

```
HHTAlphaIntegrator
├── __init__(alpha: float)
│   ├── Compute β = (1+α)²/4
│   ├── Compute γ = 0.5 + α
│   └── Initialize counters (n_lu)
├── predict(q, qp, qpp, qpp_new, h) -> (q_new, qp_new)
│   └── Newmark displacement/velocity update
├── compute_acceleration(M, R_new, R_old, C, qp_new, qp_old) -> a_new
│   └── Solve M·a = (1-α)·R_new + α·R_old - C·v
└── get_stability_info() -> dict
```

**HHT-α Formulation:**

```
Displacement:  q_{n+1} = q_n + h·v_n + h²·[(1/2-β)·a_n + β·a_{n+1}]
Velocity:      v_{n+1} = v_n + h·[(1-γ)·a_n + γ·a_{n+1}]
Acceleration:  M·a_{n+1} = (1-α)·F_{n+1} + α·F_n - (1-α)·C·v_{n+1} - α·C·v_n
```

### Parametric Studies (`parametric.py`)

**Design Pattern:** Functional with dataclasses

```
ScenarioDefinition (dataclass)
├── name: str
├── weight: float
├── params: dict
└── base_config: dict | None

run_parametric_envelope(scenarios, quantity, n_workers)
├── For each scenario in parallel:
│   └── run_simulation(scenario.params)
├── Interpolate all results to common time grid
├── Compute envelope: max, mean, min
└── Return (envelope_df, summary_df, metadata)
```

---

## Data Flow

### Single Simulation Flow

```
User Input (YAML/Dict/SimulationParams)
         │
         ▼
   run_simulation()
         │
         ├─► Validate parameters
         │
         ├─► Create ImpactSimulator instance
         │       │
         │       ├─► TrainBuilder.build_geometry()
         │       ├─► StructuralDynamics.assemble_M()
         │       ├─► StructuralDynamics.assemble_K()
         │       ├─► StructuralDynamics.assemble_C()
         │       └─► HHTAlphaIntegrator(alpha)
         │
         ▼
   ImpactSimulator.run()
         │
         ├─► Initialize state: q₀, v₀, a₀
         │
         ├─► Time loop (for n steps):
         │       │
         │       ├─► Compute internal forces (springs, damping)
         │       │       └─► BoucWenModel.update()
         │       │
         │       ├─► Compute contact forces
         │       │       └─► ContactModels.compute_force()
         │       │
         │       ├─► Compute friction forces
         │       │       └─► FrictionModels.{lugre|dahl|...}()
         │       │
         │       ├─► Solve for acceleration (Newton-Raphson):
         │       │       │
         │       │       └─► While not converged:
         │       │               ├─► Predict displacement
         │       │               ├─► Assemble residual
         │       │               ├─► Compute Jacobian (finite diff.)
         │       │               ├─► Line search
         │       │               └─► Update solution
         │       │
         │       ├─► HHTAlphaIntegrator.predict() → q, v
         │       │
         │       ├─► Track energy:
         │       │       ├─► Kinetic: ½·v^T·M·v
         │       │       ├─► Strain: ½·q^T·K·q + hysteresis
         │       │       └─► Dissipated: friction + contact damping
         │       │
         │       └─► Store state to history
         │
         ├─► Verify energy balance
         │
         └─► Build results DataFrame
                 │
                 ▼
          pd.DataFrame (time history)
```

### Parametric Study Flow

```
User defines scenarios:
  [Scenario1(v=200), Scenario2(v=120), Scenario3(v=80)]
         │
         ▼
run_parametric_envelope()
         │
         ├─► Distribute scenarios to workers (parallel)
         │       │
         │       ├─► Worker 1: run_simulation(Scenario1)
         │       ├─► Worker 2: run_simulation(Scenario2)
         │       └─► Worker 3: run_simulation(Scenario3)
         │
         ├─► Collect all DataFrames
         │
         ├─► Interpolate to common time grid
         │
         ├─► Compute envelope:
         │       ├─► max = pointwise maximum
         │       ├─► mean = weighted average
         │       └─► min = pointwise minimum
         │
         └─► Return (envelope_df, summary_df, metadata)
```

---

## Physics Engine

### Lumped Mass System

The train is modeled as a chain of lumped masses:

```
Wall  ←→  [M₁] ←→ [M₂] ←→ [M₃] ←→ ... ←→ [Mₙ]
       k₁,c₁   k₂,c₂   k₃,c₃
```

**Degrees of Freedom:**
- `n_cars` masses (train cars)
- 1 wall (fixed, not a DOF)
- Optional: 1 building/pier SDOF

**System Matrices:**

```
Mass Matrix (M):
  M = diag([m₁, m₂, ..., mₙ])

Stiffness Matrix (K):
  Tri-diagonal for spring connections
  K = [  k₁+k₂    -k₂       0      ...
          -k₂    k₂+k₃    -k₃     ...
           0      -k₃    k₃+k₄    ...
          ...     ...     ...     ... ]

Damping Matrix (C):
  Rayleigh damping: C = α·M + β·K
```

### Force Assembly

At each time step, total force on each mass:

```
F_total[i] = F_internal[i] + F_contact[i] + F_friction[i] + F_mass_contact[i]

where:
  F_internal   = -K·q - C·v + BoucWen hysteresis
  F_contact    = Wall contact (first mass only)
  F_friction   = Friction between masses
  F_mass_contact = Mass-to-mass collision (deactivated springs)
```

### Newton-Raphson Solver

For each time step, solve nonlinear equilibrium:

```
Find q_{n+1} such that:
  R(q_{n+1}) = M·a_{n+1}(q_{n+1}) - F(q_{n+1}) = 0

Iterative solver:
  1. Guess q_{n+1} (from predictor)
  2. While ||R|| > tol:
     a. Compute Jacobian: J = ∂R/∂q (finite differences)
     b. Solve: J·Δq = -R
     c. Line search: q ← q + λ·Δq (λ chosen by Armijo condition)
     d. Check convergence: ||R|| < tol
```

### Energy Balance

Continuous verification throughout simulation:

```
E_total(t) = KE(t) + SE(t) + DE(t)

where:
  KE(t) = ½·v^T·M·v           (kinetic)
  SE(t) = ½·q^T·K·q + ∫Fᵦw·dq (elastic + hysteresis)
  DE(t) = ∫(Fᵣᵢcₜ + Fₐₐₘₚ)·v·dt (dissipated)

Conservation check:
  E_total(t) ≈ E_initial  (within numerical tolerance)
```

---

## Extensibility

### Adding a Physics Model

1. **Create new module** (e.g., `plasticity.py`)
2. **Define model class:**
   ```python
   class PlasticityModels:
       @staticmethod
       def ramberg_osgood(strain, sigma_y, n):
           # Implementation
           pass
   ```
3. **Import in `engine.py`:**
   ```python
   from .plasticity import PlasticityModels
   ```
4. **Integrate into force assembly:**
   ```python
   F_plastic = PlasticityModels.ramberg_osgood(...)
   F_total += F_plastic
   ```

### Adding a Study Type

1. **Create study module:** `studies/my_study.py`
2. **Implement study function:**
   ```python
   def run_my_study(base_params, study_params) -> pd.DataFrame:
       results = []
       for param_value in study_params:
           params = base_params.copy()
           params['my_param'] = param_value
           df = run_simulation(params)
           results.append(df)
       return combine_results(results)
   ```
3. **Register CLI command:** in `studies/cli.py`

### Adding a UI Component

1. **Create UI module:** `ui/my_component.py`
2. **Implement Streamlit component:**
   ```python
   def render_my_component():
       st.subheader("My Component")
       param = st.slider("Parameter", 0, 100)
       return param
   ```
3. **Integrate in `app.py`:**
   ```python
   from .ui.my_component import render_my_component
   param = render_my_component()
   ```

---

## Technology Stack

### Core Dependencies

```
numpy           # Array operations, linear algebra
scipy           # Scientific computing utilities
pandas          # DataFrames for time history
matplotlib      # Plotting (CLI, reports)
plotly          # Interactive plots (UI)
```

### Interface Dependencies

```
typer           # CLI framework
rich            # CLI formatting
streamlit       # Web UI
openpyxl        # Excel export
PyYAML          # Configuration files
```

### Development Dependencies

```
pytest          # Testing
black           # Code formatting
mypy            # Type checking
ruff            # Linting
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Dominant Factor |
|-----------|-----------|----------------|
| Mass matrix assembly | O(n) | n = n_cars |
| Stiffness matrix assembly | O(n) | Sparse, tri-diagonal |
| Force evaluation | O(n) | Vectorized |
| Linear solve (dense LU) | O(n³) | **Bottleneck** |
| Newton iteration | O(n³) | LU solve per iteration |
| Time loop | O(N·k·n³) | N = n_steps, k = avg Newton iters |

### Typical Performance

```
Small system (5 cars, 0.5s, dt=1e-4):
  - Steps: 5,000
  - Newton iters/step: ~3
  - Wall time: ~50 ms

Large system (14 cars, 2s, dt=1e-4):
  - Steps: 20,000
  - Newton iters/step: ~4
  - Wall time: ~500 ms
```

### Optimization Strategies

1. **Sparse matrices:** For very large systems (n > 50), use sparse LU
2. **Adaptive time stepping:** Increase dt when solution is smooth
3. **Preconditioned iterative solvers:** For massive systems
4. **Parallel parametric studies:** Utilize multi-core for scenarios
5. **Numba JIT:** Compile hot loops for 10-100× speedup

---

## Security Considerations

### Input Validation

- All user inputs validated before simulation
- Parameter ranges checked (e.g., mass > 0, dt > 0)
- Model names validated against known models

### Numerical Stability

- Newton solver max iterations enforced
- Simulation duration capped
- Energy balance violations trigger warnings
- Overflow/underflow protection in force calculations

---

## Deployment

### Packaging

```bash
# Build distribution
python -m build

# Install from wheel
pip install dist/railway_simulator-1.0.0-py3-none-any.whl
```

### Docker (Future)

```dockerfile
FROM python:3.10-slim
COPY . /app
RUN pip install /app
CMD ["railway-sim", "ui"]
```

---

## Future Architecture Considerations

### Scalability

- **GPU acceleration:** CUDA for large matrix operations
- **Distributed computing:** MPI for massive parametric studies
- **Cloud deployment:** AWS Lambda for on-demand simulation

### Additional Features

- **3D visualization:** Three.js for interactive train geometry
- **Real-time updates:** WebSocket streaming of results
- **Database integration:** PostgreSQL for result storage
- **REST API:** FastAPI for web services

---

**Architecture Version:** 1.0
**Last Updated:** 2025-12-20
**Maintained By:** Sebastián Labbé
