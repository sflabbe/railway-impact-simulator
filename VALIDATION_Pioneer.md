# Validation: Pioneer wagon crash test

This document summarises how the `ice1_80kmh.yml` configuration relates to the full-scale **Pioneer passenger wagon crash test** and how you can reproduce the comparison.

The idea is simple:  
- use a **minimal discrete vehicle model**,  
- choose **realistic crush characteristics**,  
- and verify that the simulated force–time and energy histories are consistent with the full-scale test data discussed in the DZSF report.

---

## 1. Scenario overview

- Single passenger vehicle impacting a **rigid barrier**.
- Impact speed around **80 km/h**.
- Vehicle idealised as a **7-mass system** with 6 nonlinear springs.
- Contact with the barrier through a stiff nonlinear contact spring.
- No building / pier DOF activated in the default validation config.

The reference configuration is stored in:

```text
configs/ice1_80kmh.yml
```

---

## 2. Model ingredients

In the validation setup, the following ingredients are active:

- **Train model**
  - `n_masses = 7`
  - `masses` and `x_init` chosen to approximate the Pioneer vehicle layout
  - all `y_init` = 0 (pure longitudinal motion)

- **Crushing behaviour**
  - each inter-mass spring is a **Bouc–Wen element**
  - `fy` and `uy` represent approximate yield force and “plastic” displacement
  - Bouc–Wen shape parameters `bw_a`, `bw_A`, `bw_beta`, `bw_gamma`, `bw_n`
    tuned to get a realistic plateau and reloading behaviour

- **Contact**
  - `contact_model = "lankarani-nikravesh"` (Hunt–Crossley style nonlinearity)
  - `k_wall` high enough to approximate a rigid barrier
  - `cr_wall` controls the restitution

- **Integration**
  - HHT-α with `alpha_hht = -0.15`
  - initial time step `h_init = 1e-4 s`
  - total duration `T_max = 0.4 s`
  - Newton tolerance and max iterations chosen to keep the solution stable

All of this is encoded in `configs/ice1_80kmh.yml` so you can track every assumption.

---

## 3. Quantities used for validation

For the Pioneer case we mainly look at:

- **Impact force history**
  - `Impact_Force_MN` over `Time_ms`
  - peak force and force plateau
  - general shape of the force pulse

- **Penetration**
  - `Penetration_mm` vs time
  - indicative of crush stroke and stiffness

- **Energy balance**
  - `E_kin_J`, `E_spring_J`, `E_contact_J`, `E_damp_rayleigh_J`, `E_friction_J`
  - `E_total_initial_J`, `E_total_tracked_J`, `E_balance_error_J`
  - used to check that energy bookkeeping is consistent (no artificial gain/loss)

The DZSF report contains reference curves and numbers. The idea here is not to reproduce every detail perfectly, but to make sure that:

- peak force and plateau level are in a realistic range,
- impact duration is plausible,
- dissipated energy matches the available kinetic energy within a reasonable tolerance,
- energy balance errors remain small.

---

## 4. How to reproduce the validation run

From the repository root:

```bash
# Single Pioneer-type impact
railway-sim run \
  --config configs/ice1_80kmh.yml \
  --output-dir results/ice1_80 \
  --ascii-plot \
  --plot \
  --pdf-report
```

This will give you:

- `results/ice1_80/results.csv` – full time history  
- `results/ice1_80/summary.json` – key scalar metrics  
- `results/ice1_80/railway_sim_run.log` – log file with performance stats  
- `results/ice1_80/ice1_80kmh_report.pdf` – compact PDF report

You can then:

- overlay `Impact_Force_MN` on the digitised Pioneer test curve,
- compare impact duration and plateau level,
- inspect energies in the PDF or CSV.

---

## 5. Notes and limitations

- The model is intentionally **low-order**:
  - no local buckling,
  - no detailed vehicle interior,
  - no explicit representation of couplers or bogie frames beyond the lumped-mass idealisation.

- The goal is:
  - to capture **global impact force and energy dissipation**,  
  - not to predict local failure modes.

- If you need detailed structural response of a pier, abutment or deck:
  - use the **SDOF building DOF** as an intermediate step, and/or
  - transfer the impact force history into a proper FE model.

For any safety-critical assessment, always combine the results with:
- the DZSF report,
- applicable standards and guidelines,
- and normal engineering judgement.
