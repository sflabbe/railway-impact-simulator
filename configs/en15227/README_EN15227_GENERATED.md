# EN 15227 case configs (generated)

Generated on 2025-12-22T15:04:59.

This folder contains per-train YAML variants for EN 15227-inspired cases C1–C3.

Each variant keeps the original **distributed** mass/force pattern and rescales:

- `masses`: scaled so sum(masses) matches the case reference mass

- `fy`: scaled by the same factor (keeps stiffness-to-mass ratio roughly comparable)

- `uy`: unchanged

- `v0_init`: set to case speed (C1/C2: 10.0 m/s, C3: 30.56 m/s) with the sign taken from the base file (defaults to negative).

- `collision.scenario`: set to `EN15227_2011_C{1|2|3}_baseline`


Reference masses used:

- C1: 80,000 kg (wagon)
- C2: 129,000 kg (regional train)
- C3: 129,000 kg (train mass; C3 obstacle mass is not modeled in the current wall-impact formulation)


## Summary table

| base                    | variant                                     | case   |   M0_kg |   target_kg |     scale |   v0_init_mps | scenario                 |
|:------------------------|:--------------------------------------------|:-------|--------:|------------:|----------:|--------------:|:-------------------------|
| generic_passenger.yml   | en15227/generic_passenger__EN15227_C1.yml   | C1     |   54000 |       80000 | 1.48148   |        -10    | EN15227_2011_C1_baseline |
| generic_passenger.yml   | en15227/generic_passenger__EN15227_C2.yml   | C2     |   54000 |      129000 | 2.38889   |        -10    | EN15227_2011_C2_baseline |
| generic_passenger.yml   | en15227/generic_passenger__EN15227_C3.yml   | C3     |   54000 |      129000 | 2.38889   |        -30.56 | EN15227_2011_C3_baseline |
| ice1_aluminum.yml       | en15227/ice1_aluminum__EN15227_C1.yml       | C1     |   40000 |       80000 | 2         |        -10    | EN15227_2011_C1_baseline |
| ice1_aluminum.yml       | en15227/ice1_aluminum__EN15227_C2.yml       | C2     |   40000 |      129000 | 3.225     |        -10    | EN15227_2011_C2_baseline |
| ice1_aluminum.yml       | en15227/ice1_aluminum__EN15227_C3.yml       | C3     |   40000 |      129000 | 3.225     |        -30.56 | EN15227_2011_C3_baseline |
| ice1_coach.yml          | en15227/ice1_coach__EN15227_C1.yml          | C1     |   57300 |       80000 | 1.39616   |        -10    | EN15227_2011_C1_baseline |
| ice1_coach.yml          | en15227/ice1_coach__EN15227_C2.yml          | C2     |   57300 |      129000 | 2.25131   |        -10    | EN15227_2011_C2_baseline |
| ice1_coach.yml          | en15227/ice1_coach__EN15227_C3.yml          | C3     |   57300 |      129000 | 2.25131   |        -30.56 | EN15227_2011_C3_baseline |
| ice1_powercar.yml       | en15227/ice1_powercar__EN15227_C1.yml       | C1     |   78000 |       80000 | 1.02564   |        -10    | EN15227_2011_C1_baseline |
| ice1_powercar.yml       | en15227/ice1_powercar__EN15227_C2.yml       | C2     |   78000 |      129000 | 1.65385   |        -10    | EN15227_2011_C2_baseline |
| ice1_powercar.yml       | en15227/ice1_powercar__EN15227_C3.yml       | C3     |   78000 |      129000 | 1.65385   |        -30.56 | EN15227_2011_C3_baseline |
| ice1_steel.yml          | en15227/ice1_steel__EN15227_C1.yml          | C1     |   40000 |       80000 | 2         |        -10    | EN15227_2011_C1_baseline |
| ice1_steel.yml          | en15227/ice1_steel__EN15227_C2.yml          | C2     |   40000 |      129000 | 3.225     |        -10    | EN15227_2011_C2_baseline |
| ice1_steel.yml          | en15227/ice1_steel__EN15227_C3.yml          | C3     |   40000 |      129000 | 3.225     |        -30.56 | EN15227_2011_C3_baseline |
| ice1_trainset_14car.yml | en15227/ice1_trainset_14car__EN15227_C1.yml | C1     |  843600 |       80000 | 0.0948317 |        -10    | EN15227_2011_C1_baseline |
| ice1_trainset_14car.yml | en15227/ice1_trainset_14car__EN15227_C2.yml | C2     |  843600 |      129000 | 0.152916  |        -10    | EN15227_2011_C2_baseline |
| ice1_trainset_14car.yml | en15227/ice1_trainset_14car__EN15227_C3.yml | C3     |  843600 |      129000 | 0.152916  |        -30.56 | EN15227_2011_C3_baseline |
| traxx_freight.yml       | en15227/traxx_freight__EN15227_C1.yml       | C1     |   88000 |       80000 | 0.909091  |        -10    | EN15227_2011_C1_baseline |
| traxx_freight.yml       | en15227/traxx_freight__EN15227_C2.yml       | C2     |   88000 |      129000 | 1.46591   |        -10    | EN15227_2011_C2_baseline |
| traxx_freight.yml       | en15227/traxx_freight__EN15227_C3.yml       | C3     |   88000 |      129000 | 1.46591   |        -30.56 | EN15227_2011_C3_baseline |