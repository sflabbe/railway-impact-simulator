# Studies patch v3 (aligned with railway-impact-simulator v0.2.0)

## Fix for your current error (NameError: app is not defined)
In `src/railway_simulator/cli.py`, ensure these lines appear **after**
the `app = typer.Typer(...)` definition (not above it):

```python
from .studies.cli import register_study_commands
register_study_commands(app)
```

Recommended placement: immediately after the `app = typer.Typer(...)` block.

Then reinstall editable:

```bash
pip install -e .
```

## Provided study commands
- `railway-sim convergence --config ... --dts "..." --out ...`
- `railway-sim sensitivity --config ... <param_path> --values "..." --out ...`
- `railway-sim strain-rate-sensitivity --config ... --difs "..." --k-path k_wall --out ...`

## Notes
- Config files in this repo are often **override-only**. Studies therefore merge
  overrides with `get_default_simulation_params()` so baseline values like `k_wall`
  are always available.
- Convergence study recomputes `step` whenever `h_init` changes, matching the engine.
