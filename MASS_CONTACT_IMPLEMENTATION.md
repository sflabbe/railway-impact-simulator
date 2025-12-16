# Mass-to-Mass Contact Implementation Guide

## Problem Statement

When springs are highly compressed during impact, adjacent train masses can overlap. Instead of self-penetration, we want to enforce contact constraints when masses "touch."

## Proposed Solution: Contact at Zero Spring Length

### Core Idea

- Each spring has an initial length `L0[i] = u10[i]` (computed from initial geometry)
- Current spring length: `L_current[i] = L0[i] + u_spring[i]`
- When `L_current[i] ≤ L_min` (some minimum threshold), masses i and i+1 are in contact
- Add a **unilateral contact force** to prevent further compression

### Implementation in `src/railway_simulator/core/engine.py`

#### Step 1: Add Contact Tracking State

In `ImpactSimulator.__init__`:

```python
# Add after line 761 (after self.u10 initialization)
# Minimum allowed spring lengths (e.g., 5% of initial length)
self.L_min = 0.05 * self.u10  # [m]

# Mass-to-mass contact state
self.mass_contact_active = np.zeros(n - 1, dtype=bool)
```

#### Step 2: Add Mass Contact Detection Method

Add new method to `ImpactSimulator` class:

```python
def _compute_mass_contact(
    self,
    step_idx: int,
    u_spring: np.ndarray,
    q: np.ndarray,
    qp: np.ndarray,
    R_mass_contact: np.ndarray,
) -> None:
    """
    Detect and enforce contact between adjacent masses.

    When spring compression causes adjacent masses to touch
    (spring length ≤ L_min), add contact force to prevent
    further compression.

    Args:
        step_idx: Current timestep index
        u_spring: Spring deformations (negative = compression)
        q: Current positions
        qp: Current velocities
        R_mass_contact: Mass contact force vector (output)
    """
    n = self.params.n_masses
    k_contact = 1e8  # Contact stiffness [N/m] - very stiff
    c_contact = 1e5  # Contact damping [N·s/m]

    R_mass_contact[:, step_idx + 1] = 0.0

    for i in range(n - 1):
        # Current spring length
        L_current = self.u10[i] + u_spring[i, step_idx + 1]

        # Check if masses are in contact (spring fully compressed)
        if L_current <= self.L_min[i]:
            penetration = self.L_min[i] - L_current
            self.mass_contact_active[i] = True

            # Get positions and velocities of masses i and i+1
            r1 = q[[i, n + i], step_idx + 1]
            r2 = q[[i + 1, n + i + 1], step_idx + 1]
            v1 = qp[[i, n + i], step_idx + 1]
            v2 = qp[[i + 1, n + i + 1], step_idx + 1]

            # Contact normal (from mass i to mass i+1)
            dr = r2 - r1
            dist = np.linalg.norm(dr)
            if dist > 1e-12:
                n_vec = dr / dist
            else:
                n_vec = np.array([1.0, 0.0])

            # Relative velocity along normal
            dv = v2 - v1
            v_rel_normal = np.dot(dv, n_vec)

            # Contact force (penalty + damping)
            # Elastic component
            F_elastic = k_contact * penetration

            # Damping component (only if approaching)
            F_damping = 0.0
            if v_rel_normal < 0:  # masses approaching
                F_damping = c_contact * abs(v_rel_normal)

            F_contact = F_elastic + F_damping

            # Apply force along normal direction
            # Force on mass i: push away from mass i+1
            R_mass_contact[i, step_idx + 1] -= F_contact * n_vec[0]
            R_mass_contact[n + i, step_idx + 1] -= F_contact * n_vec[1]

            # Force on mass i+1: equal and opposite
            R_mass_contact[i + 1, step_idx + 1] += F_contact * n_vec[0]
            R_mass_contact[n + i + 1, step_idx + 1] += F_contact * n_vec[1]

        elif L_current > self.L_min[i] * 1.1:  # Add hysteresis
            # Release contact if spring extends beyond 110% of minimum
            self.mass_contact_active[i] = False
```

#### Step 3: Integrate into Main Simulation Loop

Modify `ImpactSimulator.run()` method:

```python
# Add after line 802 (in state arrays initialization section)
R_mass_contact = np.zeros((dof, p.step + 1))

# Add after line 891 (after friction computation)
# --- Mass-to-mass contact ---
self._compute_mass_contact(
    step_idx,
    u_spring,
    q[:, step_idx + 1],
    qp[:, step_idx + 1],
    R_mass_contact,
)

# Modify line 907 (acceleration computation) to include mass contact:
qpp[:, step_idx + 1] = self.integrator.compute_acceleration(
    self.M,
    R_internal[:, step_idx + 1], R_internal[:, step_idx],
    R_contact[:, step_idx + 1], R_contact[:, step_idx],
    R_friction[:, step_idx + 1], R_friction[:, step_idx],
    R_mass_contact[:, step_idx + 1], R_mass_contact[:, step_idx],  # ADD THIS
    self.C,
    qp[:, step_idx + 1],
    qp[:, step_idx],
)
```

#### Step 4: Update HHT Integrator

Modify `HHTAlphaIntegrator.compute_acceleration()` to accept mass contact forces:

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
    R_mass_contact: np.ndarray,        # ADD THIS
    R_mass_contact_old: np.ndarray,    # ADD THIS
    C: np.ndarray,
    qp: np.ndarray,
    qp_old: np.ndarray
) -> np.ndarray:
    """Compute acceleration using HHT-α method."""
    R_total_new = R_internal + R_contact + R_friction + R_mass_contact  # MODIFY
    R_total_old = R_internal_old + R_contact_old + R_friction_old + R_mass_contact_old  # MODIFY

    force = (
        (1.0 - self.alpha) * R_total_new +
        self.alpha * R_total_old -
        (1.0 - self.alpha) * (C @ qp) -
        self.alpha * (C @ qp_old)
    )

    self.n_lu += 1
    return np.linalg.solve(M, force)
```

#### Step 5: Add Energy Tracking for Mass Contact

Add energy dissipation tracking for mass contact:

```python
# In energy bookkeeping section (around line 980)

# 6) Mass-to-mass contact dissipation
F_mass_contact = R_mass_contact[:, step_idx + 1]
p_mass_contact = -float(np.dot(F_mass_contact, v_mid))
dE_mass_contact = max(p_mass_contact, 0.0) * self.h
E_mass_contact[step_idx + 1] = E_mass_contact[step_idx] + dE_mass_contact
```

---

## Option 2: Effective Mass Approach

Alternative implementation for **wall contact only**:

### Concept

When masses are "locked together" by full compression:
- Compute effective mass for wall contact
- Front mass sees combined inertia of locked masses

### Implementation Sketch

```python
def _get_effective_contact_mass(self, u_spring, masses):
    """
    Compute effective mass for wall contact.

    Masses that are locked together (spring fully compressed)
    contribute to effective mass.
    """
    n = len(masses)
    effective_mass = np.zeros(n)

    # Start from front
    m_eff = masses[0]
    for i in range(n - 1):
        # Check if mass i and i+1 are locked
        L_current = self.u10[i] + u_spring[i]

        if L_current <= self.L_min[i]:
            # Locked: add mass i+1 to effective mass
            m_eff += masses[i + 1]
        else:
            # Not locked: mass i has accumulated effective mass
            effective_mass[i] = m_eff
            m_eff = masses[i + 1]

    effective_mass[n - 1] = m_eff
    return effective_mass

# Use in wall contact force calculation
m_eff = self._get_effective_contact_mass(u_spring[:, step_idx + 1], self.params.masses)
# Use m_eff[0] for front mass wall contact
```

---

## Comparison of Approaches

| Aspect | Contact Constraint (Option 1) | Effective Mass (Option 2) |
|--------|-------------------------------|---------------------------|
| **Complexity** | Medium | Low |
| **Accuracy** | High - enforces kinematic constraint | Medium - only affects wall contact |
| **Energy conservation** | Can track contact dissipation | Limited - doesn't prevent mass overlap |
| **Reversibility** | Yes - with hysteresis | Harder to implement |
| **Physics** | Correct - prevents penetration | Approximate - heuristic |

---

## Recommended Approach

**Use Option 1 (Contact Constraint)** for the following reasons:

1. **Physically correct**: Enforces non-penetration constraint between masses
2. **Energy tracking**: Can properly track contact dissipation
3. **Robust**: Works for all impact scenarios, not just wall contact
4. **Reversible**: Contact can activate/deactivate as springs compress/extend

---

## Testing Strategy

### Test Case 1: Moderate Impact (No Mass Contact Expected)
- 80 km/h ICE-1 impact
- Springs should not fully compress
- Verify energy balance improves without activating mass contact

### Test Case 2: Severe Impact (Mass Contact Expected)
- 157 km/h ICE-3 impact (current problematic case)
- Front springs likely to fully compress
- Check that mass contact activates (`self.mass_contact_active[0] == True`)
- Verify no penetration: `L_current[i] >= L_min[i]` for all i

### Test Case 3: Energy Conservation
- Run with mass contact enabled
- Track `E_mass_contact_dissipated`
- Verify total energy balance error < 5%

---

## Implementation Checklist

- [ ] Add `L_min` and `mass_contact_active` to `ImpactSimulator.__init__`
- [ ] Implement `_compute_mass_contact()` method
- [ ] Add `R_mass_contact` to state arrays
- [ ] Integrate mass contact force into main loop
- [ ] Update `HHTAlphaIntegrator.compute_acceleration()` signature
- [ ] Add energy tracking for mass contact dissipation
- [ ] Update energy balance calculation to include `E_mass_contact`
- [ ] Add logging/diagnostics for contact activation
- [ ] Write unit tests for contact detection
- [ ] Run validation tests on ICE-1 and ICE-3 cases

---

## Parameters to Tune

| Parameter | Suggested Value | Description |
|-----------|----------------|-------------|
| `L_min` | `0.05 * u10[i]` | Minimum spring length (5% of initial) |
| `k_contact` | `1e8` N/m | Contact stiffness (very stiff) |
| `c_contact` | `1e5` N·s/m | Contact damping coefficient |
| Hysteresis | 1.1 × L_min | Release contact at 110% of L_min |

Adjust these based on simulation stability and physical realism.
