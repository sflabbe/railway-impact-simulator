# EN15227 case variants (auto-generated)

Generated variants in `configs/en15227/`.

Rules applied:

- masses := masses * s
- fy := fy * s (if present)
- k_train := k_train * s (if present)
- uy unchanged
- v0_init magnitude: C1/C2 = 10.00 m/s, C3 = 30.56 m/s; sign preserved from base
- collision.scenario set to EN15227_2011_C{1|2|3}_baseline


## Scale factors

| base | variant | case | M0_kg | M_target_kg | scale | v0_init | had_k_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generic_passenger.yml | generic_passenger__EN15227_C1.yml | C1 | 54000 | 80000 | 1.481481 | -10.00 | False |
| generic_passenger.yml | generic_passenger__EN15227_C2.yml | C2 | 54000 | 129000 | 2.388889 | -10.00 | False |
| generic_passenger.yml | generic_passenger__EN15227_C3.yml | C3 | 54000 | 129000 | 2.388889 | -30.56 | False |
| ice1_aluminum.yml | ice1_aluminum__EN15227_C1.yml | C1 | 40000 | 80000 | 2.000000 | -10.00 | False |
| ice1_aluminum.yml | ice1_aluminum__EN15227_C2.yml | C2 | 40000 | 129000 | 3.225000 | -10.00 | False |
| ice1_aluminum.yml | ice1_aluminum__EN15227_C3.yml | C3 | 40000 | 129000 | 3.225000 | -30.56 | False |
| ice1_coach.yml | ice1_coach__EN15227_C1.yml | C1 | 57300 | 80000 | 1.396161 | -10.00 | True |
| ice1_coach.yml | ice1_coach__EN15227_C2.yml | C2 | 57300 | 129000 | 2.251309 | -10.00 | True |
| ice1_coach.yml | ice1_coach__EN15227_C3.yml | C3 | 57300 | 129000 | 2.251309 | -30.56 | True |
| ice1_powercar.yml | ice1_powercar__EN15227_C1.yml | C1 | 78000 | 80000 | 1.025641 | -10.00 | True |
| ice1_powercar.yml | ice1_powercar__EN15227_C2.yml | C2 | 78000 | 129000 | 1.653846 | -10.00 | True |
| ice1_powercar.yml | ice1_powercar__EN15227_C3.yml | C3 | 78000 | 129000 | 1.653846 | -30.56 | True |
| ice1_steel.yml | ice1_steel__EN15227_C1.yml | C1 | 40000 | 80000 | 2.000000 | -10.00 | False |
| ice1_steel.yml | ice1_steel__EN15227_C2.yml | C2 | 40000 | 129000 | 3.225000 | -10.00 | False |
| ice1_steel.yml | ice1_steel__EN15227_C3.yml | C3 | 40000 | 129000 | 3.225000 | -30.56 | False |
| ice1_trainset_14car.yml | ice1_trainset_14car__EN15227_C1.yml | C1 | 843600 | 80000 | 0.094832 | -10.00 | True |
| ice1_trainset_14car.yml | ice1_trainset_14car__EN15227_C2.yml | C2 | 843600 | 129000 | 0.152916 | -10.00 | True |
| ice1_trainset_14car.yml | ice1_trainset_14car__EN15227_C3.yml | C3 | 843600 | 129000 | 0.152916 | -30.56 | True |
| traxx_freight.yml | traxx_freight__EN15227_C1.yml | C1 | 88000 | 80000 | 0.909091 | -10.00 | False |
| traxx_freight.yml | traxx_freight__EN15227_C2.yml | C2 | 88000 | 129000 | 1.465909 | -10.00 | False |
| traxx_freight.yml | traxx_freight__EN15227_C3.yml | C3 | 88000 | 129000 | 1.465909 | -30.56 | False |
