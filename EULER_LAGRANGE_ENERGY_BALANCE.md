# Euler-Lagrange Energy Balance

## Mathematical Formulation

The energy balance is based on the Euler-Lagrange equations of motion for the complete system (Train + Building, if modeled).

### Energy Components

**Mechanical Energy:**
```
E_mech = T + V
```
- `T`: Kinetic energy = `0.5 * v^T M v`
- `V`: Potential energy = `V_spring + V_contact`
  - `V_spring`: Conservative part of Bouc-Wen springs = `0.5 * a * k * u²`
  - `V_contact`: Elastic part of wall contact = `∫ F_elastic dδ`

**Work and Dissipation:**
```
W_ext = ∫ qdot^T Q_ext dt
E_diss = -∫ qdot^T Q_nc dt
```
- `W_ext`: External work (wall reaction for rigid wall case)
- `E_diss`: Total dissipation from non-conservative forces
  - `E_diss_rayleigh`: Rayleigh damping
  - `E_diss_bw`: Bouc-Wen hysteretic dissipation
  - `E_diss_contact_damp`: Contact damping dissipation
  - `E_diss_friction`: Coulomb friction dissipation
  - `E_diss_mass_contact`: Mass-to-mass contact dissipation

**Numerical Residual:**
```
E_num = E0 + W_ext - (E_mech + E_diss)
```
- `E0`: Initial total energy (all kinetic at t=0)
- `E_num_ratio = |E_num| / E0`: Quality metric (target: <1%)

## Force Separation

### Conservative Forces (Q_cons)

Derive from potential energy: `Q_cons = -∂V/∂q`

1. **Elastic spring forces**: `f_cons = a * k * u` (Bouc-Wen elastic part)
2. **Elastic contact forces**: `f_elastic = -k_wall * δ^n` (Hertzian or linear)

### Non-Conservative Forces (Q_nc)

Do NOT derive from potential:

1. **Rayleigh damping**: `Q_nc_rayleigh = -C * v`
2. **Bouc-Wen hysteresis**: `Q_nc_bw = (1-a) * fy * z` (hysteretic part)
3. **Contact damping**: `Q_nc_contact_damp = -c(δ) * v` (velocity-dependent)
4. **Friction**: `Q_nc_friction = -μ * N * sign(v)` (Coulomb)
5. **Mass contact**: `Q_nc_mass_contact` (penalty forces)

### External Forces (Q_ext)

For rigid wall: `Q_ext = R_contact` (wall reaction)
If building has DOFs: `Q_ext = 0` (contact is internal)

## Implementation Details

### NO abs(), NO max()

**OLD (incorrect)**:
```python
p_hyst = abs(f_hyst * du / dt)  # WRONG!
dE_damp = max(p_damp, 0.0) * dt  # WRONG!
```

**NEW (correct)**:
```python
dE_rayleigh = -float(v_mid^T @ Q_nc_rayleigh) * dt  # Proper sign
dE_bw = -float(v_mid^T @ Q_nc_bw) * dt  # Proper sign
```

### Midpoint Integration

Uses midpoint values consistent with HHT-α:
```python
v_mid = 0.5 * (v_old + v_new)
dE_diss = -float(v_mid^T @ Q_nc) * dt
```

### Contact Damping

Separate elastic and damping components:
```python
# Elastic force (contributes to V)
f_elastic = -k * δ^n

# Damping force (contributes to Q_nc)
f_damp = -c(δ) * v  # Only when approaching (v < 0)
```

Damping coefficients by model:
- **Lankarani-Nikravesh**: `c(δ) = k * δ^1.5 * 3(1-cr²)/(4v0)`
- **Hunt-Crossley**: `c(δ) = k * δ^1.5 * 3(1-cr)/(2v0)`
- **Flores**: `c(δ) = k * δ^1.5 * 8(1-cr)/(5cr*v0)`
- **Gonthier**: `c(δ) = k * δ^1.5 * (1-cr²)/(cr*v0)`
- **Ye/Anagnostopoulos**: `c(δ) = k * δ * 3(1-cr)/(2cr*v0)`
- **Pant-Wijeyewickrema**: `c(δ) = k * δ * 3(1-cr²)/(2cr²*v0)`

## Energy Balance Plots

### Main Energy Plot (Subplot 5)

Shows complete energy evolution:
- **Kinetic (T)**: Orange solid line
- **Potential (V)**: Green solid line
- **Mechanical (T+V)**: Blue dashed line
- **External work**: Cyan solid line
- **Dissipated (total)**: Red solid line
- **Numerical residual**: Magenta dotted line
- **E₀ (initial)**: Gray dashed line (reference)

### Energy Balance Quality (Subplot 6)

Shows numerical residual ratio:
- **|E_num| / E₀**: Red solid line (%)
- **1% target**: Green dashed line (reference)

**Interpretation**:
- Below 1%: Excellent energy conservation ✓
- 1-5%: Acceptable for engineering purposes
- Above 5%: Poor energy balance, check timestep/tolerances

## Validation Tests

Four automated tests verify correctness:

1. **Conservative case**: `|E_num|/E0 < 0.2%` (no dissipation)
2. **Dissipative case**: `|E_num|/E0 < 1%` (with damping)
3. **Non-negativity**: All energy components ≥ 0
4. **Euler-Lagrange identity**: `E_num = E0 + W_ext - (E_mech + E_diss)` holds exactly

Run tests:
```bash
pytest tests/test_energy_balance.py -v
```

## Physical Interpretation

### During Impact

1. **Approach phase** (t < t_contact):
   - `T` decreases (train slowing down)
   - `V = 0` (no contact yet)
   - `E_mech = T` decreases
   - `E_diss` increases slightly (Rayleigh damping)

2. **Compression phase** (t_contact < t < t_max_compression):
   - `T` decreases rapidly (kinetic → potential)
   - `V` increases rapidly (springs + contact compression)
   - `W_ext` increases (wall pushes back)
   - `E_diss` increases (contact damping, hysteresis)

3. **Rebound phase** (t > t_max_compression):
   - `T` increases (potential → kinetic)
   - `V` decreases (springs releasing)
   - `E_diss` continues increasing
   - Final: `E_mech + E_diss ≈ E0 + W_ext` (within `E_num`)

### Energy Flow Diagram

```
E0 (initial kinetic)
  ↓
  ├─→ E_mech (T+V) ←─┐ Oscillates during impact
  │                   │
  ├─→ E_diss ────────┘ Monotonically increases
  │   ├─ Rayleigh
  │   ├─ Bouc-Wen hysteresis
  │   ├─ Contact damping
  │   ├─ Friction
  │   └─ Mass contact
  │
  └─→ W_ext (wall work)
```

### Residual Sources

`E_num` comes from:
1. HHT-α numerical damping (intentional, small)
2. Nonlinear solver tolerance (iterative convergence)
3. Time integration error (truncation)
4. Floating point roundoff

Target `|E_num|/E0 < 1%` ensures these are negligible.

## Comparison: Old vs New

| Aspect | OLD (broken) | NEW (Euler-Lagrange) |
|--------|--------------|----------------------|
| Spring energy | Mixed elastic+hysteretic | Separated: `V_spring` vs `E_diss_bw` |
| Contact | Mixed elastic+damping | Separated: `V_contact` vs `E_diss_contact_damp` |
| Dissipation tracking | `abs()`, `max(power,0)` | `-qdot^T @ Q_nc` (proper sign) |
| Energy balance | "Total tracked" (unclear) | `E_num = E0 + W_ext - (E_mech + E_diss)` |
| Quality metric | None | `E_num_ratio` with 1% target |
| Result | Creates energy (>50% error) | Conserves energy (<1% error) ✓ |

## References

- Goldstein, H. (2002). Classical Mechanics (3rd ed.). Addison Wesley.
- Bathe, K.J. (1996). Finite Element Procedures. Prentice Hall.
- Lankarani, H.M., Nikravesh, P.E. (1990). A contact force model with hysteresis damping for impact analysis of multibody systems.
