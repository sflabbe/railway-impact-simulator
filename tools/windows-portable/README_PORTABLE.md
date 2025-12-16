# Portable Windows bundle (self-contained ZIP)

This tooling builds a **self-contained ZIP** for Windows that runs on a machine
**without Python installed**.

It works by:
1) downloading the official **Python embeddable** ZIP,
2) installing this project (and optionally UI extras) into it,
3) packaging everything as `RailwayImpactSimulator_Portable_Windows.zip`.

## Build (on your Windows machine)

From the repo root, double-click one of:

- `BUILD_PORTABLE_CLI.cmd` (CLI only)
- `BUILD_PORTABLE_UI.cmd` (includes Streamlit UI)

Output:
- `dist_portable\RailwayImpactSimulator_Portable_Windows.zip`

## Run (on your professor's Windows machine)

1) Unzip `RailwayImpactSimulator_Portable_Windows.zip`
2) Double-click:
   - `Example_Run_ICE1_80kmh.bat` (one example run)
   - `Run_CLI.bat` (opens a terminal with the right PATH)
   - `Run_UI.bat` (Streamlit UI at http://127.0.0.1:8501) — only if you built with UI

## Notes

- Default embeddable Python version: **3.12.12** (change in the PS1 if desired).
- If the build machine has restricted internet, you can pre-download:
  - `python-<ver>-embed-<arch>.zip` into `dist_portable\`
  - `get-pip.py` into `dist_portable\`
