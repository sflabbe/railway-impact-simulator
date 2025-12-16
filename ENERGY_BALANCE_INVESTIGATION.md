# Energy Balance Investigation Report

**Case:** ICE3-like train, 157 km/h, 0.8 s simulation, 0° impact angle, dt = 0.1 ms

**Problem:** Energy balance error of ~398 MJ with initial mechanical energy of ~386 MJ (>100% error)

**Date:** 2025-12-16

---

## 1. Observed Behavior

From the simulation results:
- Initial mechanical energy: 386.09 MJ
- Maximum energy balance deviation: 398.16 MJ
- Energy balance error: >100% of initial energy

The energy plot shows:
- **Total tracked energy** (purple): rises from ~400 MJ to ~600 MJ
- **Kinetic energy** (orange): drops from ~400 MJ to near 0 MJ
- **Mechanical energy T+V** (beige dashed): ~300-400 MJ
- **Dissipated energy** (gray): remains relatively small

**This indicates massive energy creation, which is non-physical.**

---

## 2. Root Causes Identified

### 2.1 CRITICAL BUG: Incorrect Spring Energy Calculation

**Location:** `src/railway_simulator/core/engine.py:955-957`

```python
# 2) Elastic energy in train springs
E_spring[step_idx + 1] = 0.5 * float(
    np.sum(self.k_lin * u_spring[:, step_idx + 1] ** 2)
)
```

**Problem:** This computes spring energy assuming **purely elastic springs** using `E = 0.5 * k * u²`, but the actual spring forces are computed using the **Bouc-Wen hysteretic model**.

**Bouc-Wen force formula** (line 281-282):
```python
f_spring = (a * (fy[i] / uy[i]) * u[i] +
            (1.0 - a) * fy[i] * xfunc[i])
```

Where:
- `a = bw_a` is the elastic fraction
- `(1-a)` is the hysteretic fraction
- `xfunc` is the hysteretic state variable

**In the aluminum configuration** (commonly used):
- `bw_a = 0.0` → **NO elastic component**
- Force is purely hysteretic: `f_spring = fy * xfunc`

**Why this is wrong:**
1. For hysteretic systems, potential energy is **path-dependent** - it depends on the loading history, not just the current displacement
2. During loading and unloading, the force-displacement curve follows different paths
3. Energy is dissipated through hysteresis loops
4. Using `E = 0.5 * k * u²` assumes a conservative (path-independent) potential, which is fundamentally incompatible with hysteresis

**Impact:** This is the PRIMARY cause of the energy balance error. The code is computing "stored spring energy" that doesn't actually exist in a hysteretic system.

---

### 2.2 Missing Contact Damping Dissipation

**Location:** `src/railway_simulator/core/engine.py:959-975`

```python
# 3) Elastic energy in wall contact
delta = np.maximum(-u_contact[:n, step_idx + 1], 0.0)
model = self.params.contact_model.lower()
if model in ["hooke", "ye", "pant-wijeyewickrema", "anagnostopoulos"]:
    exp = 1.0
else:
    exp = 1.5

if np.any(delta > 0.0):
    if exp == 1.0:
        E_contact[step_idx + 1] = 0.5 * self.params.k_wall * float(np.sum(delta ** 2))
    else:
        E_contact[step_idx + 1] = (
            self.params.k_wall / (exp + 1.0) * float(np.sum(delta ** (exp + 1.0)))
        )
```

**Problem:** This only computes the **elastic** portion of contact energy. The contact models include **velocity-dependent damping terms** that dissipate energy, but this dissipation is NOT tracked.

**Example - Lankarani-Nikravesh model** (line 363-365):
```python
"lankarani-nikravesh": lambda k, d, cr, dv, v0: (
    -k * d ** 1.5 * (1.0 + 3.0 * (1.0 - cr ** 2) / 4.0 * (dv / v0))
)
```

The force has two components:
- **Elastic:** `-k * d^1.5`
- **Damping:** `-k * d^1.5 * [3.0 * (1.0 - cr²) / 4.0 * (dv / v0)]`

The damping term is velocity-dependent and dissipates energy. However:
- The elastic energy is computed (lines 969-973)
- The damping dissipation is **NOT tracked** in any energy term
- This energy simply "disappears" from the total balance

**Impact:** Energy dissipated through contact damping is not accounted for, leading to apparent energy loss or creation depending on the contact dynamics.

---

### 2.3 No Self-Contact Detection Between Masses

**Location:** `src/railway_simulator/core/engine.py:1090-1143` (`_compute_contact` method)

```python
def _compute_contact(
    self,
    step_idx: int,
    q: np.ndarray,
    ...
) -> Tuple[bool, np.ndarray]:
    """Compute contact forces at wall and update contact state."""
    ...
    # Check for contact (x < 0)
    if np.any(q[:n] < 0.0):
        for i in range(n):
            if q[i] < 0.0:
                u_contact[i, step_idx + 1] = q[i]
    ...
```

**Problem:** Contact is only detected with the **wall at x=0**. There is **NO detection** of contact/penetration between adjacent train masses.

**Scenario:**
1. During a high-speed impact (157 km/h), spring deformations can be very large
2. If spring compression `u_spring[i]` becomes larger than the initial distance between masses `self.u10[i]`, masses can overlap
3. Without self-contact detection, masses can **penetrate through each other**
4. This creates unphysical configurations where:
   - Adjacent masses occupy the same space
   - Spring forces become unrealistically large
   - Spurious energy can be created

**Verification needed:** Check if `u_spring` values exceed `self.u10` during the simulation, which would indicate mass overlap.

**Impact:** If masses are penetrating, this could contribute to spurious forces and energy creation.

---

### 2.4 Timestep May Be Too Large

**Current timestep:** dt = 0.1 ms (1.0e-4 s)

**Wall stiffness:** k_wall = 45 MN/m (typical from configs)

**Contact period estimate:**
For a mass `m` impacting a spring with stiffness `k`, the natural period is:
```
T = 2π√(m/k)
```

For the front mass (m ≈ 4000 kg) and wall stiffness:
```
T = 2π√(4000/45e6) ≈ 0.59 ms
```

**Recommended timesteps per period:** At least 20-50 timesteps per period for accurate integration of stiff contact dynamics.

**Current timesteps per period:** 0.59 ms / 0.1 ms ≈ **5.9 timesteps**

**Assessment:** The timestep is marginally acceptable but may be insufficient for:
- Accurate resolution of contact dynamics
- Preventing numerical instabilities in stiff contact
- Accurate energy tracking during rapid force changes

**Impact:** Insufficient time resolution during contact can lead to numerical errors in energy calculations and force integration.

---

### 2.5 HHT-α Integration and Numerical Damping

**Location:** `src/railway_simulator/core/engine.py:601-673` (HHTAlphaIntegrator)

**Current settings:**
- `alpha_hht = -0.10` (from aluminum config, default is -0.15)

**HHT-α method properties:**
- For `α < 0`: Method has numerical damping for high frequencies
- For `α = 0`: Reduces to Newmark-β with no numerical damping
- For `α > 0`: Numerically unstable (not used)

**The method is second-order accurate and unconditionally stable** for `α ∈ [-1/3, 0]`.

**Potential issue:** While HHT-α is energy-conserving for linear systems, for **highly nonlinear** systems with:
- Hysteresis
- Contact (unilateral constraints)
- Friction

The numerical damping introduced by `α < 0` may interact with the nonlinear force calculations in complex ways.

**Impact:** Likely a minor contributor compared to the spring energy calculation bug, but could affect energy balance accuracy.

---

## 3. Recommended Fixes

### 3.1 Fix Spring Energy Calculation (HIGH PRIORITY)

**Current approach is fundamentally flawed.** For hysteretic systems, we need to track **work done** rather than potential energy.

**Recommended implementation:**

```python
# Initialize cumulative spring work
W_spring = np.zeros(p.step + 1)

# In time-stepping loop (after computing spring forces):
# Track work done by spring forces this timestep
if step_idx > 0:
    # Mid-point velocities for spring deformations
    u_spring_mid = 0.5 * (u_spring[:, step_idx + 1] + u_spring[:, step_idx])
    du_spring_dt = (u_spring[:, step_idx + 1] - u_spring[:, step_idx]) / self.h

    # Get spring forces (already computed in R_spring_nodal)
    # Work = Force · velocity · dt
    # For Bouc-Wen: account for both elastic and hysteretic parts
    for i in range(n - 1):
        # Spring force for this element (from R_spring_nodal)
        F_spring_i = # extract from R_spring_nodal

        # Work done this timestep
        dW = F_spring_i * du_spring_dt[i] * self.h
        W_spring[step_idx + 1] += dW

# Energy dissipated by hysteresis = Initial elastic - Current elastic - Work done
# For purely hysteretic (bw_a = 0), all work goes into dissipation
```

**Alternative:** For the **mixed elastic-hysteretic** case (`0 < bw_a < 1`):
- Track elastic energy: `E_elastic = 0.5 * bw_a * (fy/uy) * u²`
- Track work done by hysteretic component separately
- Hysteretic dissipation = Work input - Change in elastic energy

### 3.2 Track Contact Damping Dissipation (HIGH PRIORITY)

Add a new energy term `E_contact_damp` to track dissipation from contact damping:

```python
# Initialize
E_contact_damp = np.zeros(p.step + 1)

# In contact computation, separate elastic and damping forces
# For Lankarani-Nikravesh:
F_elastic = -k_wall * delta ** 1.5
F_damping = -k_wall * delta ** 1.5 * (3.0 * (1.0 - cr_wall**2) / 4.0) * (dv / v0)
F_total = F_elastic + F_damping

# Track elastic energy (current approach)
E_contact = k_wall / 2.5 * sum(delta ** 2.5)

# Track damping dissipation
v_contact = # contact velocity
p_contact_damp = -F_damping · v_contact  # power dissipated
dE_contact_damp = max(p_contact_damp, 0.0) * self.h
E_contact_damp[step_idx + 1] = E_contact_damp[step_idx] + dE_contact_damp
```

Update total dissipation:
```python
E_diss_tracked = E_damp_rayleigh + E_fric + E_contact_damp
```

### 3.3 Add Self-Contact Detection (MEDIUM PRIORITY)

Implement inter-mass contact detection:

```python
def _compute_self_contact(
    self,
    step_idx: int,
    q: np.ndarray,
    qp: np.ndarray,
    R_self_contact: np.ndarray,
) -> None:
    """Detect and enforce contact between adjacent masses."""
    n = self.params.n_masses

    for i in range(n - 1):
        # Current positions
        x_i = q[i]
        y_i = q[n + i]
        x_j = q[i + 1]
        y_j = q[n + i + 1]

        # Current distance
        dx = x_j - x_i
        dy = y_j - y_i
        dist = np.hypot(dx, dy)

        # Minimum allowed distance (e.g., 10% of initial distance)
        dist_min = 0.1 * self.u10[i]

        # Check for penetration
        if dist < dist_min:
            penetration = dist_min - dist

            # Normal direction
            if dist > 1e-12:
                nx = dx / dist
                ny = dy / dist
            else:
                nx, ny = 1.0, 0.0

            # Contact force (simple penalty)
            k_contact = 1e7  # stiffness parameter
            F_mag = k_contact * penetration

            # Apply to both masses
            R_self_contact[i] += F_mag * nx
            R_self_contact[n + i] += F_mag * ny
            R_self_contact[i + 1] -= F_mag * nx
            R_self_contact[n + i + 1] -= F_mag * ny
```

### 3.4 Reduce Timestep (MEDIUM PRIORITY)

For high-speed impacts (157 km/h) with stiff contact, recommend:

```python
# For speeds > 100 km/h or very stiff contact:
h_init = 5.0e-5  # 0.05 ms instead of 0.1 ms
```

This provides ~12 timesteps per contact period instead of ~6, improving accuracy.

**Trade-off:** 2x computational cost, but better energy accuracy and stability.

### 3.5 Add Energy Balance Diagnostics (LOW PRIORITY)

Add detailed energy tracking output to help identify energy balance issues:

```python
# Compute energy components
E_in = E_kin[0]  # Initial kinetic
E_stored = E_spring + E_contact  # Stored potential
E_dissipated = E_damp_rayleigh + E_fric + E_contact_damp
E_total = E_stored + E_dissipated + E_kin
E_error = E_total - E_in

# Log energy balance
logger.info(f"Energy balance at t={self.t[step_idx]:.4f} s:")
logger.info(f"  Initial:     {E_in/1e6:.2f} MJ")
logger.info(f"  Kinetic:     {E_kin[step_idx]/1e6:.2f} MJ")
logger.info(f"  Spring:      {E_spring[step_idx]/1e6:.2f} MJ")
logger.info(f"  Contact:     {E_contact[step_idx]/1e6:.2f} MJ")
logger.info(f"  Dissipated:  {E_dissipated[step_idx]/1e6:.2f} MJ")
logger.info(f"  Error:       {E_error[step_idx]/1e6:.2f} MJ ({100*E_error[step_idx]/E_in:.1f}%)")
```

---

## 4. Summary and Conclusions

### Primary Issue
The **spring energy calculation** is fundamentally incorrect for hysteretic Bouc-Wen springs. The code assumes conservative elastic springs (`E = 0.5*k*u²`) but the forces follow a hysteretic model. This is the main source of the energy balance error.

### Secondary Issues
1. **Contact damping dissipation** is not tracked
2. **Self-contact** between masses is not detected
3. **Timestep** may be marginally too large for stiff high-speed contact

### Recommended Action Plan

**Phase 1 (Critical):**
1. Fix spring energy calculation to properly account for Bouc-Wen hysteresis
2. Add contact damping dissipation tracking

**Phase 2 (Important):**
3. Add self-contact detection between masses
4. Reduce default timestep for high-speed impacts

**Phase 3 (Nice to have):**
5. Add detailed energy balance diagnostics and logging

### Expected Outcome
After implementing the fixes in Phase 1, the energy balance error should reduce from >100% to <5% for typical cases.

---

## 5. Code Locations Reference

| Issue | File | Line Numbers | Function/Class |
|-------|------|--------------|----------------|
| Spring energy calc | `src/railway_simulator/core/engine.py` | 955-957 | `ImpactSimulator.run()` |
| Bouc-Wen forces | `src/railway_simulator/core/engine.py` | 281-282 | `BoucWenModel.compute_forces()` |
| Contact energy | `src/railway_simulator/core/engine.py` | 959-975 | `ImpactSimulator.run()` |
| Contact models | `src/railway_simulator/core/engine.py` | 354-435 | `ContactModels` |
| Contact detection | `src/railway_simulator/core/engine.py` | 1090-1143 | `ImpactSimulator._compute_contact()` |
| Energy bookkeeping | `src/railway_simulator/core/engine.py` | 817-987 | `ImpactSimulator.run()` |
| HHT integrator | `src/railway_simulator/core/engine.py` | 601-673 | `HHTAlphaIntegrator` |

---

**Report prepared by:** Claude Code Analysis
**Status:** Investigation complete, awaiting implementation of fixes
