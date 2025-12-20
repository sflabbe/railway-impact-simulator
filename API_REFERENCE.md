# API Reference

This document provides a comprehensive reference for the Railway Impact Simulator API.

## Table of Contents

- [Core Simulation](#core-simulation)
- [Contact Models](#contact-models)
- [Friction Models](#friction-models)
- [Time Integration](#time-integration)
- [Parametric Studies](#parametric-studies)
- [Configuration](#configuration)
- [Utilities](#utilities)

---

## Core Simulation

### `run_simulation()`

Main entry point for running a single railway impact simulation.

**Module:** `railway_simulator.core.engine`

**Signature:**
```python
def run_simulation(
    params: SimulationParams | Dict[str, Any]
) -> pd.DataFrame
```

**Parameters:**
- `params`: Either a `SimulationParams` dataclass instance or a dictionary of parameters

**Returns:**
- `pd.DataFrame`: Time history results with columns:
  - `Time_s`: Time in seconds
  - `Impact_Force_MN`: Total impact force in MN
  - `Penetration_mm`: Maximum penetration in mm
  - `KE_MJ`, `SE_MJ`, `DE_MJ`: Kinetic, strain, dissipated energy
  - Plus additional columns for each DOF

**Example:**
```python
from railway_simulator.core.engine import run_simulation

params = {
    'v_impact_kmh': 80,
    'duration_s': 1.0,
    'dt': 1e-4,
    'contact_model': 'hunt-crossley',
    # ... additional parameters
}

df = run_simulation(params)
print(f"Peak force: {df['Impact_Force_MN'].max():.2f} MN")
```

---

### `SimulationParams`

Dataclass containing all simulation parameters.

**Module:** `railway_simulator.core.engine`

**Key Attributes:**

#### Train Configuration
- `v_impact_kmh` (float): Impact velocity in km/h
- `n_cars` (int): Number of train cars (default: 5)
- `m_car_kg` (float): Mass per car in kg (default: 40,000)
- `L_car_m` (float): Length per car in meters (default: 25.0)

#### Contact Parameters
- `contact_model` (str): Contact model name (see [Contact Models](#contact-models))
- `k_wall_N_per_m` (float): Wall stiffness (N/m or N/m^1.5 for Hertzian)
- `cr_wall` (float): Coefficient of restitution (0.0-1.0)

#### Hysteresis Spring (Bouc-Wen)
- `k_bw_N_per_m` (float): Initial elastic stiffness
- `alpha_bw` (float): Post-yield stiffness ratio (0.0-1.0)
- `beta_bw`, `gamma_bw` (float): Hysteresis shape parameters
- `n_bw` (int): Smoothness parameter

#### Friction Parameters
- `friction_model` (str): One of `'coulomb'`, `'lugre'`, `'dahl'`, `'brown_mcphee'`
- `mu_coulomb` (float): Coulomb friction coefficient
- `mu_static` (float): Static friction coefficient

#### Numerical Parameters
- `dt` (float): Time step size in seconds
- `duration_s` (float): Simulation duration
- `alpha_hht` (float): HHT-α parameter (-1/3 to 0.0)
- `solver` (str): `'newton'` or `'picard'`
- `newton_max_iter` (int): Maximum Newton iterations (default: 20)
- `newton_tol` (float): Newton convergence tolerance (default: 1e-6)

**Example:**
```python
from railway_simulator.core.engine import SimulationParams

params = SimulationParams(
    v_impact_kmh=100.0,
    n_cars=5,
    m_car_kg=45000,
    duration_s=2.0,
    dt=1e-4,
    contact_model='lankarani-nikravesh',
    k_wall_N_per_m=1e8,
    cr_wall=0.5,
    solver='newton'
)
```

---

### `TrainConfig`

Configuration for train geometry.

**Module:** `railway_simulator.core.engine`

**Methods:**

#### `ice1_aluminum()`
```python
@staticmethod
def ice1_aluminum() -> dict
```
Returns configuration dictionary for ICE-1 aluminum crush structure.

**Returns:**
- Dictionary with keys: `k_bw_N_per_m`, `alpha_bw`, `beta_bw`, `gamma_bw`, `n_bw`, `A_N`

---

## Contact Models

### `ContactModels`

Normal contact force model implementations.

**Module:** `railway_simulator.core.contact`

**Available Models:**

| Model | Equation | Use Case |
|-------|----------|----------|
| `hooke` | F = -k·δ | Linear elastic contact |
| `hertz` | F = -k·δ^1.5 | Hertzian contact (spheres) |
| `hunt-crossley` | F = -k·δ^1.5·[1 + (3/2)(1-cr)·(δ̇/v₀)] | General impact with damping |
| `lankarani-nikravesh` | F = -k·δ^1.5·[1 + (3/4)(1-cr²)·(δ̇/v₀)] | Improved energy dissipation |
| `flores` | F = -k·δ^1.5·[1 + (8/5cr)(1-cr)·(δ̇/v₀)] | Alternative dissipative |
| `gonthier` | F = -k·δ^1.5·[1 + (1-cr²)/cr·(δ̇/v₀)] | Modified damping |
| `ye` | F = -k·δ·[1 + (3/2cr)(1-cr)·(δ̇/v₀)] | Linear with damping |
| `pant-wijeyewickrema` | F = -k·δ·[1 + (3/2cr²)(1-cr²)·(δ̇/v₀)] | Linear alternative |
| `anagnostopoulos` | F = -k·δ·[1 + (3/2cr)(1-cr)·(δ̇/v₀)] | Linear (default) |

**Methods:**

#### `compute_force()`
```python
@staticmethod
def compute_force(
    u_contact: np.ndarray,
    du_contact: np.ndarray,
    v0: np.ndarray,
    k_wall: float,
    cr_wall: float,
    model: str
) -> np.ndarray
```

Compute contact forces with unilateral constraint (compression only).

**Parameters:**
- `u_contact`: Penetration displacement (negative = contact)
- `du_contact`: Penetration velocity
- `v0`: Initial impact velocity
- `k_wall`: Stiffness (N/m for linear, N/m^1.5 for Hertzian)
- `cr_wall`: Coefficient of restitution
- `model`: Model name (lowercase)

**Returns:**
- Contact force array (negative = compression)

**Example:**
```python
from railway_simulator.core.contact import ContactModels
import numpy as np

# Single mass impact
F = ContactModels.compute_force(
    u_contact=np.array([-0.01]),  # 10mm penetration
    du_contact=np.array([-0.5]),   # approaching
    v0=np.array([2.0]),            # 2 m/s impact
    k_wall=1e8,
    cr_wall=0.6,
    model='hunt-crossley'
)
```

#### `list_models()`
```python
@staticmethod
def list_models() -> list[str]
```
Returns sorted list of available contact model names.

#### `get_model_info()`
```python
@staticmethod
def get_model_info(model: str) -> str
```
Get description and equation for a specific model.

---

## Friction Models

### `FrictionModels`

Friction model implementations.

**Module:** `railway_simulator.core.friction`

**Available Models:**

| Model | State Variables | Recommended Use |
|-------|----------------|-----------------|
| Coulomb-Stribeck | None | Simple friction with stick-slip |
| LuGre | 1 (bristle state) | Accurate dynamic friction |
| Dahl | 1 (internal state) | Simplified dynamic friction |
| Brown-McPhee | None | Smooth velocity-based friction |

**Methods:**

#### `lugre()`
```python
@staticmethod
def lugre(
    z_prev: float,
    v: float,
    F_coulomb: float,
    F_static: float,
    v_stribeck: float,
    sigma_0: float,
    sigma_1: float,
    sigma_2: float,
    h: float
) -> Tuple[float, float]
```

LuGre dynamic friction model with internal bristle state.

**Parameters:**
- `z_prev`: Previous bristle state
- `v`: Current velocity (m/s)
- `F_coulomb`: Coulomb friction force (N)
- `F_static`: Static friction force (N)
- `v_stribeck`: Stribeck velocity (m/s)
- `sigma_0`: Bristle stiffness (N/m)
- `sigma_1`: Bristle damping (N·s/m)
- `sigma_2`: Viscous damping (N·s/m)
- `h`: Time step (s)

**Returns:**
- `(F, z)`: Friction force (N) and updated bristle state

**Example:**
```python
from railway_simulator.core.friction import FrictionModels

F, z_new = FrictionModels.lugre(
    z_prev=0.0,
    v=0.5,
    F_coulomb=100.0,
    F_static=120.0,
    v_stribeck=0.01,
    sigma_0=1e5,
    sigma_1=1e3,
    sigma_2=10.0,
    h=1e-4
)
```

#### `dahl()`
```python
@staticmethod
def dahl(
    z_prev: float,
    v: float,
    F_coulomb: float,
    sigma_0: float,
    h: float
) -> Tuple[float, float]
```

Dahl friction model (simplified dynamic friction).

#### `coulomb_stribeck()`
```python
@staticmethod
def coulomb_stribeck(
    v: float,
    Fc: float,
    Fs: float,
    vs: float,
    Fv: float
) -> float
```

Coulomb + Stribeck + viscous friction (no internal state).

**Parameters:**
- `v`: Velocity (m/s)
- `Fc`: Coulomb force (N)
- `Fs`: Static force (N)
- `vs`: Stribeck velocity (m/s)
- `Fv`: Viscous coefficient (N·s/m)

#### `brown_mcphee()`
```python
@staticmethod
def brown_mcphee(
    v: float,
    Fc: float,
    Fs: float,
    vs: float
) -> float
```

Smooth friction approximation using tanh and rational functions.

---

## Time Integration

### `HHTAlphaIntegrator`

HHT-α implicit time integration for structural dynamics.

**Module:** `railway_simulator.core.integrator`

**Constructor:**
```python
def __init__(self, alpha: float)
```

**Parameters:**
- `alpha`: HHT parameter (-1/3 to 0.0)
  - α = 0: No damping (Newmark trapezoidal)
  - α = -0.3: Moderate damping (recommended)
  - α = -1/3: Maximum stable damping

**Attributes:**
- `alpha` (float): HHT parameter
- `beta` (float): Newmark β = (1+α)²/4
- `gamma` (float): Newmark γ = 0.5 + α
- `n_lu` (int): Counter for LU factorizations

**Methods:**

#### `predict()`
```python
def predict(
    self,
    q: np.ndarray,
    qp: np.ndarray,
    qpp: np.ndarray,
    qpp_new: np.ndarray,
    h: float
) -> Tuple[np.ndarray, np.ndarray]
```

Newmark predictor-corrector step.

**Parameters:**
- `q`: Current displacement
- `qp`: Current velocity
- `qpp`: Current acceleration
- `qpp_new`: New acceleration
- `h`: Time step

**Returns:**
- `(q_new, qp_new)`: Updated displacement and velocity

#### `compute_acceleration()`
```python
def compute_acceleration(
    self,
    M: np.ndarray,
    R_internal: np.ndarray,
    R_internal_old: np.ndarray,
    R_contact: np.ndarray,
    R_contact_old: np.ndarray,
    R_friction: np.ndarray,
    R_friction_old: np.ndarray,
    R_mass_contact: np.ndarray,
    R_mass_contact_old: np.ndarray,
    C: np.ndarray,
    qp: np.ndarray,
    qp_old: np.ndarray
) -> np.ndarray
```

Solve HHT-α equilibrium equation for acceleration.

#### `get_stability_info()`
```python
def get_stability_info(self) -> dict
```

Returns dictionary with stability parameters and properties.

**Example:**
```python
from railway_simulator.core.integrator import HHTAlphaIntegrator

integrator = HHTAlphaIntegrator(alpha=-0.3)
info = integrator.get_stability_info()
print(f"Spectral radius at ∞: {info['spectral_radius_inf']:.3f}")
```

---

## Parametric Studies

### `run_parametric_envelope()`

Run multiple scenarios and compute envelope (max/mean/min).

**Module:** `railway_simulator.core.parametric`

**Signature:**
```python
def run_parametric_envelope(
    scenarios: list[ScenarioDefinition],
    quantity: str = "Impact_Force_MN",
    n_workers: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]
```

**Parameters:**
- `scenarios`: List of `ScenarioDefinition` instances
- `quantity`: Column name to compute envelope for
- `n_workers`: Number of parallel workers

**Returns:**
- `envelope_df`: DataFrame with `max`, `mean`, `min` columns
- `summary_df`: Summary statistics
- `metadata`: Run metadata (git hash, timestamp, etc.)

**Example:**
```python
from railway_simulator.core.parametric import ScenarioDefinition, run_parametric_envelope

scenarios = [
    ScenarioDefinition(name="High Speed", weight=0.3, params={'v_impact_kmh': 200}),
    ScenarioDefinition(name="Medium Speed", weight=0.5, params={'v_impact_kmh': 120}),
    ScenarioDefinition(name="Low Speed", weight=0.2, params={'v_impact_kmh': 80}),
]

envelope, summary, meta = run_parametric_envelope(scenarios, quantity="Impact_Force_MN")
print(f"Peak envelope force: {envelope['max'].max():.2f} MN")
```

### `ScenarioDefinition`

Dataclass for defining parametric scenarios.

**Attributes:**
- `name` (str): Scenario name
- `weight` (float): Weight for mean computation (0-1)
- `params` (dict): Parameter overrides
- `base_config` (dict, optional): Base configuration

---

## Configuration

### Loading Configurations

**YAML Format:**
```yaml
# configs/example.yml
v_impact_kmh: 80.0
n_cars: 5
m_car_kg: 40000
L_car_m: 25.0
contact_model: hunt-crossley
k_wall_N_per_m: 1.0e8
cr_wall: 0.5
# ... additional parameters
```

**Loading:**
```python
import yaml

with open('configs/ice1_aluminum.yml') as f:
    params = yaml.safe_load(f)

df = run_simulation(params)
```

### CLI Configuration

```bash
# Run with config file
railway-sim run --config configs/ice1_aluminum.yml

# Override parameters
railway-sim run --config configs/ice1_aluminum.yml --speed-kmh 120

# Parametric study
railway-sim parametric --base-config configs/ice1_aluminum.yml \
  --speeds "200:0.3,120:0.5,80:0.2"
```

---

## Utilities

### Strain Rate Metrics

**Module:** `railway_simulator.core.engine`

```python
def strain_rate_metrics(
    df: pd.DataFrame,
    t_col: str = "Time_s",
    penetration_col: str = "Penetration_mm",
    penetration_units: str = "mm",
    L_ref_m: float = 1.0,
    contact_force_col: str | None = "Impact_Force_MN",
    force_threshold: float = 0.001,
    pen_threshold: float = 1e-9,
    smooth_window: int = 5
) -> Dict[str, float]
```

Compute strain-rate proxy metrics from time history.

**Returns:**
- Dictionary with keys:
  - `peak_strain_rate_per_sec`: Peak ε̇ (1/s)
  - `mean_strain_rate_active_per_sec`: Mean during active contact
  - `duration_active_s`: Duration of active contact

---

## Performance Metrics

### Estimating FLOPS

The simulation performance can be estimated from:

```python
n_steps = int(duration_s / dt)
n_dof = n_cars + 1  # DOFs in system
flops_per_step = 2 * n_dof**3  # Dense LU solve
total_flops = flops_per_step * n_steps * avg_newton_iters
```

Typical performance:
- **Small system** (5 cars, 6 DOF): ~10-50 ms per simulation
- **Large system** (14 cars, 15 DOF): ~100-500 ms per simulation
- Newton iterations: 2-5 per step (depends on nonlinearity)

---

## Error Handling

All simulation functions may raise:

- `ValueError`: Invalid parameters (e.g., negative mass, invalid model name)
- `RuntimeError`: Convergence failure in Newton solver
- `KeyError`: Missing required parameters in configuration

**Example:**
```python
try:
    df = run_simulation(params)
except RuntimeError as e:
    print(f"Simulation failed: {e}")
    # Try with smaller time step or higher tolerance
```

---

## References

1. **HHT-α Method**: Hilber, H. M., Hughes, T. J., and Taylor, R. L. (1977). "Improved numerical dissipation for time integration algorithms in structural dynamics."

2. **Contact Models**: Hunt & Crossley (1975), Lankarani & Nikravesh (1990), Flores et al. (2011)

3. **Friction Models**: Canudas de Wit et al. (1995) - LuGre, Dahl (1976), Brown & McPhee (2016)

4. **DZSF Bericht 53** (2024): German Railway Crash Energy Management Report

---

## Version History

- **v1.0.0** (2024): Initial release with modular architecture
- Refactored contact, friction, and integrator modules into separate files
- Comprehensive documentation and validation

---

**See Also:**
- [Developer Guide](DEVELOPER_GUIDE.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [Validation: Pioneer Test](VALIDATION_Pioneer.md)
- [Contact Model Verification](CONTACT_MODEL_VERIFICATION.md)
