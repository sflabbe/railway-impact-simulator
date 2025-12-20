# Developer Guide

This guide is for developers who want to contribute to, extend, or understand the internals of the Railway Impact Simulator.

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Adding New Features](#adding-new-features)
- [Performance Optimization](#performance-optimization)
- [Debugging](#debugging)
- [Common Tasks](#common-tasks)

---

## Getting Started

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/railway-impact-simulator.git
cd railway-impact-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with all dependencies
pip install -e ".[ui,dev]"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/railway_simulator --cov-report=html

# Run specific test file
pytest tests/test_engine.py -v

# Run with live logging
pytest tests/ -v --log-cli-level=INFO
```

### Code Quality Tools

```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/

# All quality checks
black src/ tests/ && isort src/ tests/ && mypy src/ && ruff check src/
```

---

## Project Structure

### Directory Layout

```
railway-impact-simulator/
├── src/railway_simulator/          # Main package
│   ├── core/                        # Core simulation engine
│   │   ├── engine.py                # Main simulator, params, constants
│   │   ├── contact.py               # Contact force models
│   │   ├── friction.py              # Friction models
│   │   ├── integrator.py            # HHT-α time integration
│   │   ├── parametric.py            # Parametric studies
│   │   ├── report.py                # PDF report generation
│   │   └── app.py                   # Streamlit web UI
│   ├── ui/                          # UI components (Streamlit)
│   │   ├── parameters.py            # Parameter input widgets
│   │   ├── plotting.py              # Plotly visualizations
│   │   ├── simulation.py            # Simulation execution
│   │   ├── sdof.py                  # SDOF building analysis
│   │   ├── export.py                # Export functionality
│   │   └── about.py                 # About page
│   ├── studies/                     # Convergence & sensitivity studies
│   │   ├── __init__.py              # Study utilities
│   │   ├── convergence.py           # Time-step convergence
│   │   ├── numerics_sensitivity.py  # dt/alpha/tolerance sweeps
│   │   ├── strain_rate_sensitivity.py # DIF sensitivity
│   │   └── cli.py                   # Study CLI commands
│   ├── materials/                   # Material models
│   │   └── strain_rate.py           # DIF models (reference)
│   └── cli.py                       # Main CLI application
├── configs/                         # YAML configuration files
│   ├── ice1_aluminum.yml
│   ├── ice1_steel.yml
│   └── ...
├── tests/                           # Unit and integration tests
│   ├── test_engine.py
│   ├── test_contact.py
│   ├── test_friction.py
│   ├── test_parametric.py
│   └── ...
├── examples/                        # Example scripts
├── docs/                            # Additional documentation
├── tools/                           # Build and deployment tools
└── pyproject.toml                   # Project metadata

```

### Module Responsibilities

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `core/engine.py` | Main simulation logic | `ImpactSimulator`, `run_simulation()`, `SimulationParams` |
| `core/contact.py` | Contact force models | `ContactModels` |
| `core/friction.py` | Friction models | `FrictionModels` |
| `core/integrator.py` | Time integration | `HHTAlphaIntegrator` |
| `core/parametric.py` | Multi-scenario studies | `run_parametric_envelope()` |
| `ui/parameters.py` | UI parameter widgets | `build_parameter_ui()` |
| `ui/plotting.py` | Visualization | `plot_time_history()`, `plot_train_geometry()` |
| `studies/convergence.py` | Convergence analysis | `run_convergence_study()` |
| `cli.py` | Command-line interface | `app` (Typer) |

---

## Development Workflow

### Feature Development

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/add-new-contact-model
   ```

2. **Implement the feature:**
   - Add code to appropriate module
   - Follow existing patterns and conventions
   - Add docstrings and type hints

3. **Add tests:**
   ```python
   # tests/test_contact.py
   def test_new_contact_model():
       F = ContactModels.compute_force(
           u_contact=np.array([-0.01]),
           du_contact=np.array([-1.0]),
           v0=np.array([2.0]),
           k_wall=1e8,
           cr_wall=0.5,
           model='new-model'
       )
       assert F[0] < 0  # Compression force
   ```

4. **Run tests and quality checks:**
   ```bash
   pytest tests/ -v
   black src/ tests/
   mypy src/
   ```

5. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat: add new contact model implementation"
   git push origin feature/add-new-contact-model
   ```

6. **Create pull request**

### Bug Fix Workflow

1. **Reproduce the bug:**
   - Create a minimal test case
   - Add as a failing test

2. **Fix the bug:**
   - Identify root cause
   - Implement fix
   - Ensure test passes

3. **Verify no regressions:**
   ```bash
   pytest tests/ -v  # All tests should pass
   ```

4. **Document the fix:**
   ```bash
   git commit -m "fix: correct energy balance in LuGre friction model"
   ```

---

## Code Style

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length:** 100 characters (not 79)
- **Docstring format:** NumPy style
- **Type hints:** Required for all public APIs
- **Imports:** Organized with `isort`

### Naming Conventions

```python
# Constants: UPPER_SNAKE_CASE
MAX_ITERATIONS = 100
GRAVITY = 9.81

# Classes: PascalCase
class ContactModels:
    pass

# Functions/methods: snake_case
def run_simulation(params):
    pass

# Private functions/methods: _leading_underscore
def _coerce_scalar_types(data):
    pass

# Variables: snake_case
impact_force = 100.0
n_cars = 5

# Physics variables: descriptive or standard notation
v_impact_kmh = 80.0  # Velocity in km/h
k_wall_N_per_m = 1e8  # Stiffness in N/m
alpha_hht = -0.3      # HHT parameter
```

### Docstring Format

Use NumPy-style docstrings:

```python
def compute_force(
    u_contact: np.ndarray,
    du_contact: np.ndarray,
    k_wall: float,
    model: str
) -> np.ndarray:
    """Compute contact forces using specified model.

    This function computes normal contact forces for all degrees
    of freedom using various contact force models from the literature.

    Parameters
    ----------
    u_contact : np.ndarray
        Penetration displacement, shape (n_dof,). Negative values
        indicate contact (penetration into the wall).
    du_contact : np.ndarray
        Penetration velocity, shape (n_dof,)
    k_wall : float
        Wall stiffness parameter in N/m or N/m^1.5
    model : str
        Contact model name (e.g., 'hunt-crossley')

    Returns
    -------
    np.ndarray
        Contact force array, shape (n_dof,). Negative values indicate
        compression forces.

    Raises
    ------
    ValueError
        If model name is not recognized

    Examples
    --------
    >>> F = compute_force(
    ...     u_contact=np.array([-0.01]),
    ...     du_contact=np.array([-1.0]),
    ...     k_wall=1e8,
    ...     model='hunt-crossley'
    ... )

    Notes
    -----
    All models enforce unilateral contact (compression only).
    See ContactModels.MODELS for available model equations.

    References
    ----------
    .. [1] Hunt, K. H., and Crossley, F. R. E. (1975)
    """
    # Implementation
```

### Type Hints

Use modern Python 3.10+ type hints:

```python
from __future__ import annotations  # At top of file

# Use built-in types (not typing module)
def process_data(
    values: list[float],           # Not List[float]
    mapping: dict[str, int],       # Not Dict[str, int]
    optional_param: float | None   # Not Optional[float]
) -> tuple[float, float]:          # Not Tuple[float, float]
    ...
```

---

## Testing

### Test Structure

```python
# tests/test_contact.py
import pytest
import numpy as np
from railway_simulator.core.contact import ContactModels


class TestContactModels:
    """Tests for contact force models."""

    def test_hunt_crossley_compression(self):
        """Test Hunt-Crossley model produces compression force."""
        F = ContactModels.compute_force(
            u_contact=np.array([-0.01]),
            du_contact=np.array([-1.0]),
            v0=np.array([2.0]),
            k_wall=1e8,
            cr_wall=0.5,
            model='hunt-crossley'
        )
        assert F[0] < 0, "Force should be compressive (negative)"

    def test_no_penetration_no_force(self):
        """Test that no force is computed when not in contact."""
        F = ContactModels.compute_force(
            u_contact=np.array([0.01]),  # Positive = no contact
            du_contact=np.array([1.0]),
            v0=np.array([2.0]),
            k_wall=1e8,
            cr_wall=0.5,
            model='hunt-crossley'
        )
        assert F[0] == 0.0, "No force when not in contact"

    @pytest.mark.parametrize('model', ContactModels.MODELS.keys())
    def test_all_models(self, model):
        """Test all contact models run without errors."""
        F = ContactModels.compute_force(
            u_contact=np.array([-0.01]),
            du_contact=np.array([-1.0]),
            v0=np.array([2.0]),
            k_wall=1e8,
            cr_wall=0.5,
            model=model
        )
        assert F[0] < 0
```

### Fixtures

```python
# tests/conftest.py
import pytest
from railway_simulator.core.engine import SimulationParams


@pytest.fixture
def default_params():
    """Default simulation parameters for testing."""
    return SimulationParams(
        v_impact_kmh=80.0,
        n_cars=3,
        m_car_kg=40000,
        duration_s=0.5,
        dt=1e-4,
        contact_model='hunt-crossley'
    )


@pytest.fixture
def mock_simulation_result():
    """Mock simulation result DataFrame."""
    import pandas as pd
    import numpy as np

    t = np.linspace(0, 1, 1000)
    return pd.DataFrame({
        'Time_s': t,
        'Impact_Force_MN': np.sin(t),
        'Penetration_mm': np.maximum(0, np.sin(t))
    })
```

### Energy Balance Tests

Critical for validating physics:

```python
def test_energy_conservation(default_params):
    """Test total energy conservation."""
    df = run_simulation(default_params)

    # Total energy should be conserved
    E_total = df['KE_MJ'] + df['SE_MJ'] + df['DE_MJ']
    E_initial = E_total.iloc[0]

    # Allow 1% tolerance for numerical error
    np.testing.assert_allclose(
        E_total,
        E_initial,
        rtol=0.01,
        err_msg="Energy not conserved"
    )
```

---

## Adding New Features

### Adding a New Contact Model

1. **Update contact.py:**

```python
# src/railway_simulator/core/contact.py

class ContactModels:
    MODELS = {
        # ... existing models ...
        "my-new-model": lambda k, d, cr, dv, v0: (
            -k * d ** 2.0 * (1.0 + some_damping_term)
        ),
    }
```

2. **Add description:**

```python
def get_model_info(model: str) -> str:
    descriptions = {
        # ... existing descriptions ...
        "my-new-model": "My Novel Model: F = -k*δ²*[...]",
    }
```

3. **Add tests:**

```python
def test_my_new_model():
    F = ContactModels.compute_force(
        u_contact=np.array([-0.01]),
        du_contact=np.array([-1.0]),
        v0=np.array([2.0]),
        k_wall=1e8,
        cr_wall=0.5,
        model='my-new-model'
    )
    assert F[0] < 0
```

4. **Add validation:**
   - Create validation test case
   - Compare with literature or experimental data
   - Document in `CONTACT_MODEL_VERIFICATION.md`

### Adding a New Friction Model

Similar process in `friction.py`:

```python
class FrictionModels:
    @staticmethod
    def my_new_friction(v: float, param1: float, param2: float) -> float:
        """My new friction model.

        Parameters
        ----------
        v : float
            Velocity (m/s)
        param1, param2 : float
            Model parameters

        Returns
        -------
        float
            Friction force (N)

        References
        ----------
        .. [1] Citation here
        """
        return some_function(v, param1, param2)
```

### Adding a Study Module

```python
# src/railway_simulator/studies/my_study.py

def run_my_study(base_params: dict, output_dir: str) -> pd.DataFrame:
    """Run my custom study.

    Parameters
    ----------
    base_params : dict
        Base simulation parameters
    output_dir : str
        Directory to save results

    Returns
    -------
    pd.DataFrame
        Study results
    """
    # Study implementation
    pass
```

Register in `studies/cli.py`:

```python
@app.command()
def my_study(
    config: Path,
    output_dir: Path = Path("results/my_study")
):
    """Run my custom study."""
    from railway_simulator.studies.my_study import run_my_study

    params = load_config(config)
    results = run_my_study(params, output_dir)
    logger.info(f"Study complete: {output_dir}")
```

---

## Performance Optimization

### Profiling

```python
# Profile a simulation
import cProfile
import pstats

with cProfile.Profile() as pr:
    df = run_simulation(params)

stats = pstats.Stats(pr)
stats.sort_stats('cumtime')
stats.print_stats(20)  # Top 20 functions
```

### Optimization Tips

1. **Vectorize operations:**
   ```python
   # Bad: Loop over DOFs
   for i in range(n_dof):
       F[i] = compute_force_scalar(u[i], du[i])

   # Good: Vectorized
   F = compute_force_vectorized(u, du)
   ```

2. **Avoid repeated array allocations:**
   ```python
   # Bad: Allocates every iteration
   for i in range(n_steps):
       result = np.zeros(n_dof)

   # Good: Allocate once, reuse
   result = np.zeros(n_dof)
   for i in range(n_steps):
       result[:] = 0  # Reset in-place
   ```

3. **Use numba for hot loops:**
   ```python
   from numba import jit

   @jit(nopython=True)
   def compute_contact_forces_fast(u, du, k, cr):
       # Implementation
       pass
   ```

4. **Parallel parametric studies:**
   ```python
   envelope, summary, meta = run_parametric_envelope(
       scenarios,
       n_workers=4  # Use 4 cores
   )
   ```

---

## Debugging

### Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# In code
logger = logging.getLogger(__name__)
logger.debug(f"Contact force: {F}")
logger.info(f"Simulation complete")
logger.warning(f"Newton solver converged slowly")
logger.error(f"Energy balance violated")
```

### Common Issues

**Newton solver not converging:**
```python
# Increase iterations or tolerance
params.newton_max_iter = 50
params.newton_tol = 1e-5

# Or use Picard solver
params.solver = 'picard'
```

**Energy balance violated:**
```python
# Check energy tracking
df['Energy_Error'] = (
    df['KE_MJ'] + df['SE_MJ'] + df['DE_MJ'] - df['KE_MJ'].iloc[0]
)
print(f"Max energy error: {df['Energy_Error'].abs().max():.6f} MJ")
```

**Numerical instability:**
```python
# Reduce time step
params.dt = 5e-5  # Half the previous dt

# Increase numerical damping
params.alpha_hht = -0.3  # More damping
```

---

## Common Tasks

### Running a Custom Configuration

```python
from railway_simulator.core.engine import run_simulation

params = {
    'v_impact_kmh': 100,
    'n_cars': 7,
    'm_car_kg': 50000,
    'L_car_m': 28.0,
    'k_wall_N_per_m': 1.5e8,
    'cr_wall': 0.6,
    'duration_s': 2.0,
    'dt': 1e-4,
    'contact_model': 'lankarani-nikravesh',
    'friction_model': 'lugre',
    # ... more parameters
}

df = run_simulation(params)
df.to_csv('my_results.csv', index=False)
```

### Batch Processing

```python
import yaml
from pathlib import Path

configs = Path('configs').glob('*.yml')

for config_file in configs:
    with open(config_file) as f:
        params = yaml.safe_load(f)

    df = run_simulation(params)

    output_file = f"results/{config_file.stem}.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ {config_file.stem}")
```

### Comparing Contact Models

```python
import matplotlib.pyplot as plt

models = ['hooke', 'hertz', 'hunt-crossley', 'lankarani-nikravesh']
results = {}

for model in models:
    params['contact_model'] = model
    df = run_simulation(params)
    results[model] = df

# Plot comparison
fig, ax = plt.subplots()
for model, df in results.items():
    ax.plot(df['Time_s'], df['Impact_Force_MN'], label=model)
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Impact Force (MN)')
plt.savefig('contact_model_comparison.png')
```

---

## Contributing

### Pull Request Guidelines

1. **Branch naming:**
   - Feature: `feature/description`
   - Bug fix: `fix/description`
   - Documentation: `docs/description`

2. **Commit messages:**
   - Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
   - Be descriptive: `feat: add Gonthier contact model with validation`

3. **PR description:**
   - What: Describe the changes
   - Why: Explain the motivation
   - How: Outline the approach
   - Testing: Describe test coverage

4. **Code review:**
   - All tests must pass
   - Code coverage should not decrease
   - Documentation must be updated

---

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Tag release: `git tag v1.0.0`
5. Push tag: `git push origin v1.0.0`
6. Build and publish: `python -m build && twine upload dist/*`

---

## Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [Type Hints (PEP 484)](https://peps.python.org/pep-0484/)

---

**Last Updated:** 2024-12-20
