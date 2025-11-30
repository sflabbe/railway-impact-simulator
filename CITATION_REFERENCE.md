# Citation reference

If you use this code or the underlying methodology, please cite both:

1. the **software**, and  
2. the **DZSF research report** that contains the full background.

---

## 1. Short copy-paste citations

**Software**

> Labbé, S. (2025). *Railway Impact Simulator: HHT-α implicit integration with Bouc–Wen hysteresis* [Computer software]. GitHub. https://github.com/sflabbe/railway-impact-simulator

**Research report**

> Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024). *Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr* (DZSF Bericht 53). Deutsches Zentrum für Schienenverkehrsforschung beim Eisenbahn-Bundesamt. https://doi.org/10.48755/dzsf.240006.01

---

## 2. BibTeX examples

### 2.1 Software

```bibtex
@software{labbe_railway_impact_simulator_2025,
  author    = {Sebasti{\'a}n Labb{\'e}},
  title     = {{Railway Impact Simulator}: HHT-\alpha implicit integration
               with Bouc--Wen hysteresis},
  year      = {2025},
  url       = {https://github.com/sflabbe/railway-impact-simulator},
  note      = {Version used in this work, see repository for details}
}
```

### 2.2 DZSF research report

```bibtex
@techreport{stempniewski_ueberpruefung_2024,
  author       = {Stempniewski, Lothar and Labb{\'e}, Sebasti{\'a}n
                  and Siegel, Steffen and Bosch, Robin},
  title        = {{\"U}berpr{\"u}fung und Anpassung der Anpralllasten
                  aus dem Eisenbahnverkehr},
  institution  = {Deutsches Zentrum f{\"u}r Schienenverkehrsforschung
                  beim Eisenbahn-Bundesamt},
  year         = {2024},
  number       = {DZSF-Bericht 53},
  doi          = {10.48755/dzsf.240006.01}
}
```

---

## 3. Optional background references

If you need to explain the numerical details in a paper or report, you may want to add:

- original **HHT-α** paper (Hilber–Hughes–Taylor),
- original **Bouc–Wen** hysteresis formulations,
- contact model references (Hunt–Crossley, Lankarani–Nikravesh, Anagnostopoulos),
- friction models (LuGre, Dahl, Brown–McPhee).

These are not hard requirements for using the code, but they are often expected in academic publications. The exact references depend on which model variants you actually enable in your runs.

---

## 4. Suggested wording for reports

You can describe the tool in a methods section roughly like this:

> Impact loads were computed with a dedicated Python tool (“Railway Impact Simulator”) that implements a discrete multi-mass vehicle model with Bouc–Wen crushing springs and HHT-α time integration. Several nonlinear contact and friction laws are available, and the parameter set was calibrated against the full-scale Pioneer passenger wagon crash test as documented in DZSF Bericht 53 (Stempniewski et al., 2024).
