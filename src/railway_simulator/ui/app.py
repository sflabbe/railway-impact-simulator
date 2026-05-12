"""Streamlit application entry point.

The implementation still lives in ``railway_simulator.core.app`` for backwards
compatibility with older scripts.  New launchers should point here so UI code has
an explicit home under ``railway_simulator.ui``.
"""

from __future__ import annotations

from railway_simulator.core.app import main


if __name__ == "__main__":
    main()
