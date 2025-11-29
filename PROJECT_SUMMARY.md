# Railway Impact Simulator - Project Summary

## Overview

This repository contains a Python implementation of advanced railway impact simulation 
methodologies using HHT-α implicit time integration and Bouc-Wen hysteresis models.

---

## Copyright & Licensing

### Dual License Structure

This project operates under a dual licensing model:

#### 1. Software Implementation (Code)
**Copyright:** © 2025 Sebastián Labbé  
**License:** MIT License  
**Applies to:** All Python code, scripts, and software documentation

```
MIT License

Copyright (c) 2025 Sebastián Labbé

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

#### 2. Research Methodology (Original Research)
**Copyright:** © 2024 Karlsruher Institut für Technologie (KIT), Siegel und Wünschel PartG mbB  
**License:** CC BY 4.0  
**Applies to:** Research methodologies, theoretical frameworks, and scientific content

**Research Report:**  
Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024).  
Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr.  
DZSF Bericht 53. DOI: 10.48755/dzsf.240006.01

---

## Project Information

### Software Developer
**Sebastián Labbé, Dipl.-Ing.**  
Karlsruher Institut für Technologie (KIT)  
Institut für Massivbau und Baustofftechnologie  
76131 Karlsruhe, Germany

### Research Authors (Original Methodology)
1. Univ.-Prof. Dr.-Ing. Lothar Stempniewski (KIT)
2. Dipl.-Ing. Sebastián Labbé (KIT)
3. Dr.-Ing. Steffen Siegel (Siegel und Wünschel PartG mbB)
4. Robin Bosch, M.Sc. (Siegel und Wünschel PartG mbB)

### Institutional Affiliations

**Karlsruher Institut für Technologie (KIT)**  
Institut für Massivbau und Baustofftechnologie  
Gotthard-Franz-Straße 3  
76131 Karlsruhe, Germany  
https://www.ibb.kit.edu

**Siegel und Wünschel beratende Ingenieure PartG mbB**  
Zehntwiesenstraße 35a  
76275 Ettlingen, Germany  
https://www.siegel-wuenschel.de

### Commissioning Authority

**Eisenbahn-Bundesamt (EBA)**  
Federal Railway Authority  
53175 Bonn, Germany  
https://www.eba.bund.de

### Publisher

**Deutsches Zentrum für Schienenverkehrsforschung (DZSF)**  
German Centre for Rail Transport Research  
01219 Dresden, Germany  
https://www.dzsf.bund.de

---

## Technical Implementation

### Features
- ✅ HHT-α implicit time integration
- ✅ Bouc-Wen hysteresis model for nonlinear material behavior
- ✅ Multiple contact models (Hooke, Hertz, Hunt-Crossley, Lankarani-Nikravesh, Flores, etc.)
- ✅ Multiple friction models (LuGre, Dahl, Coulomb-Stribeck, Brown-McPhee)
- ✅ 2D structural dynamics with bar elements
- ✅ Rayleigh damping
- ✅ Newton-Raphson iteration for implicit integration
- ✅ Interactive Streamlit web interface
- ✅ Real-time visualization with Plotly
- ✅ Export functionality (CSV, TXT, XLSX)

### Programming Languages & Tools
- Python 3.8+
- Streamlit (web interface)
- NumPy (numerical computations)
- SciPy (scientific computing)
- Pandas (data handling)
- Plotly (visualization)

---

## Citation Guidelines

### For Academic/Research Use

When using this software in academic research or publications, provide **dual citation**:

**1. Software Citation (Code Implementation):**
```
Labbé, S. (2025). Railway Impact Simulator: HHT-α Implicit Integration 
with Bouc-Wen Hysteresis [Computer software]. GitHub. 
https://github.com/[repository-url]
```

**2. Research Methodology Citation:**
```
Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024).
Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr.
Berichte des Deutschen Zentrums für Schienenverkehrsforschung, Bericht 53.
Deutsches Zentrum für Schienenverkehrsforschung beim Eisenbahn-Bundesamt.
https://doi.org/10.48755/dzsf.240006.01
```

**BibTeX Format:**
```bibtex
@software{Labbe2025RailwaySimulator,
  author = {Labbé, Sebastián},
  title = {Railway Impact Simulator: HHT-α Implicit Integration 
           with Bouc-Wen Hysteresis},
  year = {2025},
  url = {https://github.com/[repository-url]}
}

@techreport{Stempniewski2024Anpralllasten,
  author      = {Stempniewski, Lothar and Labbé, Sebastián and 
                 Siegel, Steffen and Bosch, Robin},
  title       = {Überprüfung und Anpassung der Anpralllasten 
                 aus dem Eisenbahnverkehr},
  institution = {Deutsches Zentrum für Schienenverkehrsforschung 
                 beim Eisenbahn-Bundesamt},
  year        = {2024},
  number      = {53},
  doi         = {10.48755/dzsf.240006.01}
}
```

---

## File Structure

```
railway-impact-simulator/
├── LICENSE                                    # MIT License
├── README.md                                  # Main documentation
├── CITATION_REFERENCE.md                      # Quick citation guide
├── FIXES_APPLIED.md                          # Technical fixes documentation
├── railway_impact_simulator_refactored.py    # Main simulator code
└── requirements.txt                           # Python dependencies
```

---

## Installation & Usage

### Requirements
```bash
pip install streamlit numpy pandas scipy plotly xlsxwriter openpyxl
```

### Running the Simulator
```bash
streamlit run railway_impact_simulator_refactored.py
```

The application will open in your default web browser at `http://localhost:8501`

### Basic Usage Flow
1. Configure simulation parameters in the sidebar:
   - Time integration settings (velocity, duration, steps)
   - Train geometry (research model or example trains)
   - Material properties (Bouc-Wen parameters)
   - Contact and friction models
2. Click "Run Simulation"
3. View results (force-time, penetration-time, hysteresis plots)
4. Export data in preferred format (CSV, TXT, XLSX)

---

## Disclaimer

### Software Warranty
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### Engineering Applications
This software is provided for **research and educational purposes**. 

For **safety-critical** or **regulatory applications**:
- Consult with qualified professional engineers
- Validate results independently
- Contact relevant authorities (e.g., Eisenbahn-Bundesamt)

The author makes no warranty regarding suitability for any particular 
application, especially safety-critical engineering design.

### Research Attribution
The responsibility for the content of the original research publication lies 
with the research authors as stated in the DZSF report.

*Die Verantwortung für den Inhalt der Forschungsveröffentlichung liegt bei 
den Autorinnen und Autoren.*

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

Please maintain:
- Code quality and documentation
- Proper attribution and citations
- Compatibility with existing features

---

## Contact Information

### Software Support
**Sebastián Labbé**  
Karlsruher Institut für Technologie (KIT)  
Email: [your email if you wish to include]

### Research Questions
**Institut für Massivbau und Baustofftechnologie (KIT)**  
https://www.ibb.kit.edu

### Regulatory/Standards Questions
**Eisenbahn-Bundesamt (EBA)**  
https://www.eba.bund.de/DE/Service/Kontakt/kontakt_node.html

---

## Acknowledgments

This software implementation was developed at the **Karlsruher Institut für 
Technologie (KIT)** based on research commissioned by the **Eisenbahn-Bundesamt 
(EBA)** and published by the **Deutsches Zentrum für Schienenverkehrsforschung 
(DZSF)**.

Special thanks to:
- The research team for developing the underlying methodologies
- EBA for funding the original research
- DZSF for publishing and disseminating the research
- The open-source Python community for essential tools and libraries

---

## Version History

### Version 1.0 (2025)
- Initial public release
- HHT-α time integration implementation
- Bouc-Wen hysteresis model
- Multiple contact and friction models
- Streamlit web interface
- Export functionality
- Comprehensive documentation

---

## Related Resources

### Research Report
**Official PDF:**  
https://www.dzsf.bund.de/SharedDocs/Downloads/DZSF/Veroeffentlichungen/Forschungsberichte/2024/ForBe_53_2024_Anpralllasten.pdf

**DOI:**  
https://doi.org/10.48755/dzsf.240006.01

### Institutional Links
- KIT: https://www.kit.edu
- DZSF: https://www.dzsf.bund.de
- EBA: https://www.eba.bund.de

### Technical References
- Streamlit Documentation: https://docs.streamlit.io
- NumPy Documentation: https://numpy.org/doc
- SciPy Documentation: https://docs.scipy.org

---

**Software © 2025 Sebastián Labbé (MIT License)**  
**Research © 2024 KIT & Siegel und Wünschel PartG mbB (CC BY 4.0)**

Last Updated: November 2024
