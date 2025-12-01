# Railway Impact Simulator - Optimization and Bug Fix Report

**Date:** 2025-12-01
**Analysis of:** railway-impact-simulator codebase

## Executive Summary

This report identifies **7 bugs** and **5 optimization opportunities** in the Railway Impact Simulator codebase. The issues range from critical import errors that break functionality to performance optimizations and numerical accuracy improvements.

---

## 🐛 Critical Bugs (Must Fix)

### 1. Missing Functions in `parametric.py` Causing Import Errors
**File:** `src/railway_simulator/core/app.py` (lines 22-25)
**Severity:** CRITICAL - Breaks Streamlit app parametric features

**Issue:**
```python
from .parametric import (
    build_speed_scenarios,  # ❌ Function does not exist
    run_parametric_envelope,
    make_envelope_figure,   # ❌ Function does not exist
)
```

**Impact:** The Streamlit app will crash on startup when trying to import these missing functions.

**Fix:** Add the missing functions to `parametric.py`:
```python
def build_speed_scenarios(
    base_params: dict,
    speeds_kmh: list[float],
    weights: list[float],
    prefix: str = "v"
) -> List[ScenarioDefinition]:
    """Build scenario definitions from speeds and weights."""
    scenarios = []
    for v_kmh, w in zip(speeds_kmh, weights):
        name = f"{prefix}{int(round(v_kmh))}"
        params_i = dict(base_params)
        params_i["v0_init"] = -v_kmh / 3.6
        scenarios.append(ScenarioDefinition(
            name=name,
            params=params_i,
            weight=w,
            meta={"speed_kmh": v_kmh}
        ))
    return scenarios

def make_envelope_figure(
    envelope_df: pd.DataFrame,
    quantity: str,
    title: str = "Envelope"
):
    """Create a Plotly figure for the envelope."""
    import plotly.graph_objects as go

    if quantity in envelope_df.columns:
        y_col = quantity
    else:
        y_col = f"{quantity}_envelope"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=envelope_df["Time_ms"],
        y=envelope_df[y_col],
        mode="lines",
        line=dict(width=2),
        name=f"Envelope {y_col}"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time [ms]",
        yaxis_title=quantity,
        height=500
    )
    return fig
```

---

### 2. Double Counting of Linear Solves in Performance Metrics
**File:** `src/railway_simulator/core/engine.py` (lines 568, 801)
**Severity:** MEDIUM - Incorrect performance statistics

**Issue:**
```python
# Line 568 in compute_acceleration()
self.n_lu += 1

# Line 801 in run() - REDUNDANT
self.linear_solves += 1
```

Both counters increment for the same operation, causing `linear_solves` to be double the actual value.

**Fix:** Remove line 801 and use only the integrator's counter:
```python
# In run() method, after the time-stepping loop:
# Update the total count from integrator
self.linear_solves = self.integrator.n_lu
```

---

### 3. Unsafe Floating-Point Equality Check
**File:** `src/railway_simulator/core/engine.py` (line 442)
**Severity:** MEDIUM - Potential numerical instability

**Issue:**
```python
if L0 == 0:  # ❌ Exact float comparison
    continue
```

**Fix:** Use tolerance-based comparison:
```python
if L0 < 1e-12:  # ✅ Safe for floating point
    continue
```

---

### 4. Incorrect v0_contact Initialization on First Contact
**File:** `src/railway_simulator/core/engine.py` (lines 1000-1004)
**Severity:** MEDIUM - Incorrect initial contact velocity

**Issue:**
```python
if (not contact_active) and np.any(du_contact[:n] < 0.0):
    contact_active = True
    v0_contact = np.where(du_contact < 0.0, du_contact, 1.0)
    v0_contact[v0_contact == 0.0] = 1.0
```

The `v0_contact` is set for ALL degrees of freedom (including y-coordinates), not just x-coordinates where contact occurs.

**Fix:**
```python
if (not contact_active) and np.any(du_contact[:n] < 0.0):
    contact_active = True
    # Only set v0 for x-DOFs that are actually in contact
    mask_contact_x = du_contact[:n] < 0.0
    v0_contact[:n] = np.where(mask_contact_x, du_contact[:n], 1.0)
    v0_contact[:n][v0_contact[:n] == 0.0] = 1.0
```

---

## ⚡ Performance Optimizations

### 5. Pre-compute Friction Disablement Check
**File:** `src/railway_simulator/core/engine.py` (lines 903-911)
**Severity:** LOW - Minor performance gain

**Issue:** The friction disablement check is performed on every time step:
```python
friction_off = (
    p.friction_model in ("none", "off", "", None)
    or (abs(p.mu_s) < 1e-12 and abs(p.mu_k) < 1e-12)
    or (...)
)
```

**Fix:** Pre-compute during initialization:
```python
# In __init__ or setup():
self.friction_enabled = not (
    self.params.friction_model in ("none", "off", "", None)
    or (abs(self.params.mu_s) < 1e-12 and abs(self.params.mu_k) < 1e-12)
    or (abs(self.params.sigma_0) < 1e-12
        and abs(self.params.sigma_1) < 1e-12
        and abs(self.params.sigma_2) < 1e-12)
)

# In _compute_friction():
if not self.friction_enabled:
    R_friction[:, step_idx + 1] = 0.0
    z_friction[:, step_idx + 1] = z_friction[:, step_idx]
    return
```

---

### 6. Use Actual Performance Metrics Instead of Estimates
**File:** `src/railway_simulator/cli.py` (lines 234-243, 296-298)
**Severity:** LOW - Inaccurate reporting

**Issue:** Performance metrics use hardcoded estimates:
```python
# Heuristic estimate: ~3 Newton iterations per time step
avg_newton = 3.0
linear_solves = int(steps * avg_newton)
```

But actual data is available from the DataFrame attributes (set in engine.py line 1106).

**Fix:**
```python
# Try to get actual metrics from DataFrame attrs
n_lu_actual = results_df.attrs.get("n_lu", None)
if n_lu_actual is not None:
    linear_solves = n_lu_actual
else:
    # Fallback to estimate
    avg_newton = 3.0
    linear_solves = int(steps * avg_newton)
```

---

### 7. Vectorize Bouc-Wen Force Computation
**File:** `src/railway_simulator/core/engine.py` (lines 171-189)
**Severity:** LOW - Moderate performance gain for large models

**Issue:** Loop over springs could be vectorized:
```python
for i in range(n_springs):
    xfunc[i] = BoucWenModel.integrate_rk4(...)
    f_spring = (a * (fy[i] / uy[i]) * u[i] + ...)
```

**Fix:** This would require significant refactoring but could provide 2-5x speedup for models with many springs. Keep current implementation for maintainability unless profiling shows it's a bottleneck.

---

### 8. Memory Optimization for Long Simulations
**File:** `src/railway_simulator/core/engine.py` (lines 673-691)
**Severity:** LOW - Only matters for very long simulations

**Issue:** All time history is stored in memory:
```python
q = np.zeros((dof, p.step + 1))
qp = np.zeros((dof, p.step + 1))
qpp = np.zeros((dof, p.step + 1))
```

For a simulation with 10,000 steps and 14 DOFs, this is ~1.7 MB per array, which is acceptable.

**Recommendation:** No change needed unless users report memory issues with very long simulations (>100k steps).

---

### 9. Add Caching for Repeated Simulations
**File:** `src/railway_simulator/core/parametric.py`
**Severity:** LOW - Potential speedup for parametric studies

**Issue:** Each scenario in parametric studies is independent and could be parallelized.

**Fix:** Add optional multiprocessing:
```python
from concurrent.futures import ProcessPoolExecutor
import os

def run_parametric_envelope(
    scenarios: List[ScenarioDefinition],
    quantity: str = "Impact_Force_MN",
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run parametric envelope with optional parallel execution.
    """
    if parallel and len(scenarios) > 1:
        max_workers = max_workers or min(len(scenarios), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_simulation, scen.params)
                      for scen in scenarios]
            dfs = [f.result() for f in futures]
            results = list(zip(scenarios, dfs))
    else:
        results = [(scen, run_simulation(scen.params)) for scen in scenarios]

    # Rest of the function remains the same
    ...
```

---

## 📝 Code Quality Improvements

### 10. Add Type Hints Consistently
**Files:** Multiple
**Severity:** LOW - Improves maintainability

Many functions are missing return type hints. Example:
```python
# Current:
def _ascii_plot(x, y, y_label, x_label, width=70, height=20):

# Better:
def _ascii_plot(
    x: np.ndarray,
    y: np.ndarray,
    y_label: str,
    x_label: str,
    width: int = 70,
    height: int = 20,
) -> str:
```

---

## 🧪 Testing Recommendations

1. **Unit tests for Bouc-Wen model**: Verify hysteresis loops against known analytical solutions
2. **Integration tests for energy conservation**: Check that energy balance error remains below threshold
3. **Regression tests for parametric studies**: Ensure envelope computation is correct
4. **Performance benchmarks**: Track simulation speed for standard test cases

---

## Priority Ranking

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| 🔴 HIGH | Missing functions in parametric.py (#1) | Breaks app | Low |
| 🟡 MEDIUM | Double counting linear solves (#2) | Wrong metrics | Low |
| 🟡 MEDIUM | Float equality check (#3) | Potential crash | Low |
| 🟡 MEDIUM | v0_contact initialization (#4) | Wrong physics | Low |
| 🟢 LOW | Pre-compute friction check (#5) | Minor speedup | Low |
| 🟢 LOW | Use actual metrics (#6) | Better reporting | Low |
| 🟢 LOW | Vectorize Bouc-Wen (#7) | Moderate speedup | High |
| 🟢 LOW | Memory optimization (#8) | Rare issue | Medium |
| 🟢 LOW | Parallel parametric (#9) | Good speedup | Medium |

---

## Conclusion

The codebase is generally well-structured and follows good scientific computing practices. The critical issues are import errors that should be fixed immediately. The performance optimizations are nice-to-have improvements that can be implemented incrementally.

**Estimated time to fix all critical and medium issues:** 2-3 hours
