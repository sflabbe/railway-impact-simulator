"""
About page components for the Railway Impact Simulator UI.

Contains header information and citation details for the research report.
"""

import streamlit as st


def display_header():
    """Display header with institutional logos and research information."""

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### KIT")
        st.markdown("**Karlsruher Institut für Technologie**")

    with col2:
        st.markdown("### EBA")
        st.markdown("**Eisenbahn-Bundesamt**")

    with col3:
        st.markdown("### DZSF")
        st.markdown("**Deutsches Zentrum für Schienenverkehrsforschung**")

    st.markdown("---")

    # Research information
    st.markdown(
        """
    ### Research Background

    **Report Title (German):**
    *Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr*

    **Report Title (English):**
    *Review and Adjustment of Impact Loads from Railway Traffic*

    **Authors:**
    - Lothar Stempniewski (KIT)
    - Sebastián Labbé (KIT)
    - Steffen Siegel (Siegel und Wünschel PartG mbB)
    - Robin Bosch (Siegel und Wünschel PartG mbB)

    **Research Institutions:**
    - Karlsruher Institut für Technologie (KIT)
      Institut für Massivbau und Baustofftechnologie

    **Publication:**
    DZSF Bericht 53 (2024)
    Project Number: 2018-08-U-1217
    Study Completion: June 2021
    Publication Date: June 2024

    **DOI:** [10.48755/dzsf.240006.01](https://doi.org/10.48755/dzsf.240006.01)
    **ISSN:** 2629-7973
    **License:** CC BY 4.0

    **Download Report:**
    [DZSF Forschungsbericht 53/2024 (PDF)](https://www.dzsf.bund.de/SharedDocs/Downloads/DZSF/Veroeffentlichungen/Forschungsberichte/2024/ForBe_53_2024_Anpralllasten.pdf?__blob=publicationFile&v=2)

    **Commissioned by:**
    Eisenbahn-Bundesamt (EBA)

    **Published by:**
    Deutsches Zentrum für Schienenverkehrsforschung (DZSF)
    """
    )

    st.markdown("---")


def display_citation():
    """Show how to cite the underlying research report."""
    st.markdown("---")
    st.markdown(
        """
    ### Citation

    If you use this simulator in your research, please cite the original research report:

    **Plain Text:**
    ```
    Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024).
    Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr.
    Berichte des Deutschen Zentrums für Schienenverkehrsforschung, Bericht 53.
    Deutsches Zentrum für Schienenverkehrsforschung beim Eisenbahn-Bundesamt.
    https://doi.org/10.48755/dzsf.240006.01
    ```

    **BibTeX:**
    ```bibtex
    @techreport{Stempniewski2024Anpralllasten,
      author       = {Stempniewski, Lothar and
                      Labbé, Sebastián and
                      Siegel, Steffen and
                      Bosch, Robin},
      title        = {Überprüfung und Anpassung der Anpralllasten
                      aus dem Eisenbahnverkehr},
      institution  = {Deutsches Zentrum für Schienenverkehrsforschung
                      beim Eisenbahn-Bundesamt},
      year         = {2024},
      type         = {Bericht},
      number       = {53},
      address      = {Dresden, Germany},
      note         = {Projektnummer 2018-08-U-1217,
                      Commissioned by Eisenbahn-Bundesamt},
      doi          = {10.48755/dzsf.240006.01},
      issn         = {2629-7973},
      url          = {https://www.dzsf.bund.de/SharedDocs/Downloads/DZSF/Veroeffentlichungen/Forschungsberichte/2024/ForBe_53_2024_Anpralllasten.pdf}
    }
    ```

    **APA 7th Edition:**
    ```
    Stempniewski, L., Labbé, S., Siegel, S., & Bosch, R. (2024).
    Überprüfung und Anpassung der Anpralllasten aus dem Eisenbahnverkehr
    (DZSF Bericht No. 53). Deutsches Zentrum für Schienenverkehrsforschung
    beim Eisenbahn-Bundesamt. https://doi.org/10.48755/dzsf.240006.01
    ```

    ---
    **License:** This work is licensed under CC BY 4.0
    """
    )
