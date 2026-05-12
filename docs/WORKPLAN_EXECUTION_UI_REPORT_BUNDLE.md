# Workplan execution: Streamlit report bundle generation

This martillazo connects the persisted LaTeX chapter generator to the Streamlit
Project Workbench.

## User-facing workflow

1. Start the UI:

```bash
railway-sim ui
```

2. Open **Project Workbench**.
3. Create or open a `project.sqlite` database.
4. Run or select a persisted full-train study.
5. Open the **Report bundle** tab.
6. Select the study, title, author, damping ratio and output folder.
7. Click **Generate LaTeX chapter bundle**.
8. Download the generated ZIP or compile `main.tex` from the output folder.

## Generated output

The report tab calls the same `build_latex_chapter` implementation as the CLI.
The output folder contains:

```text
chapter_impact_parametric_study.tex
main.tex
references.bib
chapter_metadata.json
README.md
figures/
tables/
```

The ZIP download is generated from the existing output folder. No extra hidden
state is stored in Streamlit.

## Implementation notes

New helpers in `railway_simulator.ui.project_workbench`:

- `sanitize_path_component`
- `default_chapter_output_dir`
- `build_workbench_chapter_bundle`
- `zip_report_bundle_bytes`

The UI does not duplicate report logic. It only resolves the selected study and
forwards the request to `railway_simulator.reporting.build_latex_chapter`.

## Acceptance checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q tests/test_project_workbench.py tests/test_latex_chapter_report.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q -m 'not slow'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q -m slow
PYTHONPATH=src python -m compileall -q src tests scripts examples
```
