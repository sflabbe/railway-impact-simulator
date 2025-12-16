# Energy Balance Bug Analysis

## Problem Statement

User reports energy balance error where **total tracked energy > initial kinetic energy**:
- Initial: E_kin ≈ 5 MJ, E_total ≈ 5 MJ ✓
- Final (t=300ms): E_kin ≈ 0 MJ, E_mech ≈ 8 MJ, E_total ≈ 8-9 MJ ❌
- **ERROR: System creates ~3-4 MJ of spurious energy**

This violates energy conservation: `E_total = E_initial = constant`

## Root Cause Analysis

### Bug 1: Contact Damping Coefficient Formula (**CRITICAL**)

**Location**: `src/railway_simulator/core/engine.py:1048-1061`

**Issue**: For dissipative contact models, damping coefficient uses **wrong exponent** for penetration depth δ.

**Lankarani-Nikravesh Model**:
- Contact force: `F = -k·δ^1.5·(1 + 3(1-cr²)/(4v₀)·v)`
- Elastic component: `F_elastic = -k·δ^1.5`
- Damping component: `F_damp = -k·δ^1.5·[3(1-cr²)/(4v₀)]·v`
- Damping coefficient: `c(δ) = k·δ^1.5·[3(1-cr²)/(4v₀)]` ← **Should be δ^1.5**

**Current Code** (line 1051):
```python
c_eff = p.k_wall * delta[i] ** 0.5 * 3.0 * (1.0 - p.cr_wall ** 2) / 4.0 / v0_i
```
Uses `δ^0.5` ← **WRONG!**

**Impact**:
- For δ = 0.3 m: Under-estimates damping by factor of **δ = 0.3** (~70% error)
- For δ = 0.5 m: Under-estimates damping by factor of **δ = 0.5** (~50% error)
- **Result**: Dissipation is severely under-tracked

**Same bug affects**:
- Hunt-Crossley (line 1049): Should be `δ^1.5`, not `δ^0.5`
- Flores (line 1053): Should be `δ^1.5`, not `δ^0.5`
- Gonthier (line 1055): Should be `δ^1.5`, not `δ^0.5`
- Ye/Pant/Anagnostopoulos (line 1057): Should be `δ^2`, not `δ^1` (different model family)

### Bug 2: Potential Energy Over-Estimation (Under Investigation)

**Hypothesis**: Under-tracking dissipation causes apparent energy creation.

**Energy Balance**:
```
E_total_tracked = E_kin + E_spring + E_contact + E_dissipated_tracked
```

If `E_dissipated_tracked << E_dissipated_actual`, then:
```
E_total_tracked = E_initial - (E_dissipated_actual - E_dissipated_tracked)
E_total_tracked > E_initial  ← APPEARS to create energy!
```

**But this doesn't fully explain the issue!** Even with under-tracked dissipation:
```
At peak compression: E_pot_max + E_diss = E_initial
If E_pot = 8 MJ and E_initial = 5 MJ:
Then E_diss = -3 MJ ← IMPOSSIBLE!
```

**This suggests ADDITIONAL bug**: Either:
1. E_spring or E_contact is being over-calculated
2. Forces in simulation are generating spurious work
3. HHT-α integration error is larger than expected

### Bug 3: Missing Dissipation from Contact Force Work

**Hypothesis**: The way we compute contact energy dissipation might not account for ALL the work done by the velocity-dependent damping force during the full impact cycle.

## Required Fixes

### Fix 1: Correct Contact Damping Coefficients

For models with `F = -k·δ^n·(1 + c_term·v)`:
- Damping coefficient: `c(δ) = k·δ^n·c_term`
- Power dissipated: `P = c(δ)·v²`

**Lankarani-Nikravesh** (n=1.5):
```python
c_eff = k_wall * delta[i] ** 1.5 * 3.0 * (1.0 - cr_wall ** 2) / (4.0 * v0_i)
```

**Hunt-Crossley** (n=1.5):
```python
c_eff = k_wall * delta[i] ** 1.5 * 3.0 * (1.0 - cr_wall) / (2.0 * v0_i)
```

**Flores** (n=1.5):
```python
c_eff = k_wall * delta[i] ** 1.5 * 8.0 * (1.0 - cr_wall) / (5.0 * cr_wall * v0_i)
```

**Gonthier** (n=1.5):
```python
c_eff = k_wall * delta[i] ** 1.5 * (1.0 - cr_wall ** 2) / (cr_wall * v0_i)
```

**Pant-Wijeyewickrema** (n=1.0):
```python
c_eff = k_wall * delta[i] ** 1.0 * 3.0 * (1.0 - cr_wall ** 2) / (2.0 * cr_wall ** 2 * v0_i)
```

**Ye/Anagnostopoulos** (n=1.0):
```python
c_eff = k_wall * delta[i] ** 1.0 * 3.0 * (1.0 - cr_wall) / (2.0 * cr_wall * v0_i)
```

### Fix 2: Investigate Alternative Energy Tracking

Consider computing dissipation from WORK DONE instead of power:
```python
# Current: P = c(δ)·v²
# Alternative: dW = F_damp · dx = F_damp · v · dt
```

This might capture energy dissipation more accurately during transient impacts.

## Expected Results After Fix

After correcting damping coefficients:
- Dissipation should increase by factor of **δ ≈ 2-5x** for typical impacts
- Energy balance error should reduce from ~60-80% to <5%
- E_total_tracked should remain ≈ E_initial throughout simulation

## Testing Plan

1. Apply Fix 1 (correct damping coefficient exponents)
2. Test with research locomotive at 50, 100, 200 km/h
3. Verify: `|E_total - E_initial| / E_initial < 0.05` for all speeds
4. If still >5% error, investigate Fix 2 (alternative dissipation tracking)
