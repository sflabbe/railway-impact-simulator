# Workplan execution: LaTeX thesis chapter bundle

This martillazo adds a reproducible academic reporting layer on top of the
persistent project workbench.

## New command

```bash
railway-sim report build-chapter \
  --db projects/stempi/project.sqlite \
  --study-id <study_id> \
  --output-dir results/stempi_chapter \
  --author "S. Labbe" \
  --zeta 0.05
```

The command generates:

- `chapter_impact_parametric_study.tex`: thesis-ready chapter source
- `main.tex`: standalone wrapper for review compilation
- `references.bib`: bibliography entries used by the generated chapter
- `chapter_metadata.json`: machine-readable provenance metadata
- `tables/run_summary.csv` and `.tex`
- `tables/srs_curves_long.csv`
- `tables/srs_maxima.csv` and `.tex`
- `figures/srs_overlay.png`
- `figures/srs_envelope.png`
- `figures/srs_full_train_vs_lok_ratio.png` when matched Lok/Zug curves exist

## Technical intent

The generator does not hand-edit numerical values into the text. It reads the
SQLite project database, joins run/scenario/SRS metadata, loads persisted CSV
curves, regenerates plots and writes a LaTeX bundle. This keeps the chapter
traceable to the same project artifacts used by the UI and CLI.

## Source concepts captured in the chapter

The chapter template integrates the current technical narrative used in the
impact chapter draft and the contact-model post-mortem:

- EN reference force `F_EN = 6 MN`
- first-contact scope
- lumped-mass Bouc-Wen model
- Lankarani-Nikravesh contact law
- HHT-alpha integration
- geometric angle filter
- post-derailment runout equation
- SRS equivalent-force postprocessing
- per-mass contact state patch
- signed penetration-rate convention
- persisted DB/CSV/figure reproducibility trail

## Validation

Executed after implementation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q -m 'not slow'
# 164 passed, 17 deselected

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q -m slow
# 17 passed, 164 deselected

PYTHONPATH=src python -m compileall -q src tests scripts examples
# ok
```
