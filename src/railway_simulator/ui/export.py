"""
Export utilities for the Railway Impact Simulator UI.

Provides functions to export simulation results to various formats.
"""

import io

import pandas as pd


def to_excel(df: pd.DataFrame) -> bytes:
    """
    Generate Excel file for download.

    Args:
        df: DataFrame to export

    Returns:
        Bytes representation of Excel file
    """
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Dynamic Load History")
    except ImportError:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Dynamic Load History")
    return output.getvalue()
