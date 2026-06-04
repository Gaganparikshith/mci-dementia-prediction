#!/usr/bin/env python3
"""
README - check_ucsffsx7_columns.py
==================================

This script is a quick diagnostic utility for checking the column names in
UCSFFSX7_21May2026.csv before running build_clinical_plus_volumes.py.

The main purpose is to identify whether important MRI volumetric columns are
present in the UCSFFSX7 file, especially columns related to hippocampus, ICV,
ventricles, temporal lobe regions, entorhinal cortex, fusiform cortex, and
subject identifiers.

This helps confirm the correct column names before building the Phase 1.5
clinical + MRI volumetric dataset.

Input
-----
C:/Users/ASUS/Desktop/Research Resources/DATA/UCSFFSX7_21May2026.csv

What this script checks
-----------------------
1. Loads only the first few rows of the UCSFFSX7 CSV file.
2. Prints the total number of columns.
3. Searches for important keywords in the column names.
4. Prints matching column names for each keyword.

Run
---
python check_ucsffsx7_columns.py

Next step
---------
Use the printed column names inside build_clinical_plus_volumes.py.
"""

from pathlib import Path

import pandas as pd


# Input file
UCSFFSX7_CSV = Path(
    r"C:\Users\ASUS\Desktop\Research Resources\DATA\UCSFFSX7_21May2026.csv"
)


# Keywords to search in the UCSFFSX7 column names
KEYWORDS = [
    "HIPPO",
    "ICV",
    "VENT",
    "BRAIN",
    "ENTORH",
    "FUSIF",
    "MIDTMP",
    "TEMP",
    "INTRA",
    "SUBCORT",
    "RID",
    "PTID",
    "VISCODE",
    "EXAMDATE",
    "ST88",
    "ST124",
    "ST14",
    "ST15",
    "ST162",
    "ST26",
    "ST32",
    "ST48",
    "ST97",
    "ST75",
    "ST81",
]


def load_columns(csv_path):
    """Load only a few rows and return the column names."""
    df = pd.read_csv(
        csv_path,
        nrows=5,
        low_memory=False,
    )

    return list(df.columns)


def print_matching_columns(columns, keywords):
    """Print column names that match each keyword."""
    print(f"Total columns: {len(columns)}\n")

    for keyword in keywords:
        matches = [
            column
            for column in columns
            if keyword.upper() in column.upper()
        ]

        if matches:
            print(f"  {keyword:<12}: {matches[:8]}")


def main():
    if not UCSFFSX7_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {UCSFFSX7_CSV}")

    columns = load_columns(UCSFFSX7_CSV)

    print_matching_columns(
        columns,
        KEYWORDS,
    )


if __name__ == "__main__":
    main()