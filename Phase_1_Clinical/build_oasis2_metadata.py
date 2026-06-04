"""
README - build_oasis2_metadata.py
=================================

This script builds the OASIS-2 external validation file for the Phase 1
clinical-only model.

It reads dementia_dataset.csv, identifies MCI-like subjects using baseline
CDR, assigns pMCI-like and sMCI-like labels based on follow-up CDR trajectory,
maps available OASIS-2 variables to ADNI-style E4 feature names, and saves the
final validation-ready CSV file.

Input
-----
C:/Users/ASUS/Desktop/Research Resources/DATA/dementia_dataset.csv

Output
------
C:/Users/ASUS/Desktop/Research Resources/DementiaResearch/data/metadata/oasis2_validation_ready.csv

Labelling logic
---------------
Only subjects with baseline CDR = 0.5 are included.

pMCI-like, LABEL = 1
    Baseline CDR = 0.5 and follow-up CDR becomes >= 1.0,
    or the Group column contains "Converted".

sMCI-like, LABEL = 0
    Baseline CDR = 0.5 and follow-up CDR remains <= 0.5.

Feature mapping
---------------
M/F   -> PTGENDER
        M = 0, F = 1

EDUC  -> PTEDUCAT

MMSE  -> MMSE_BL

All other ADNI E4 features are not available in OASIS-2 and are kept as NaN.
They will be median-imputed later by validate_oasis2.py using the ADNI
training-set imputation values.

Run
---
python build_oasis2_metadata.py

Next step
---------
python "Clinical Part/validate_oasis2.py"
"""

import os
import sys

import numpy as np
import pandas as pd


# Paths
INPUT_CSV = r"C:\Users\ASUS\Desktop\Research Resources\DATA\dementia_dataset.csv"

OUTPUT_DIR = (
    r"C:\Users\ASUS\Desktop\Research Resources"
    r"\DementiaResearch\data\metadata"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "oasis2_validation_ready.csv",
)


# ADNI E4 feature names expected by validate_oasis2.py
E4_FEATURES = [
    "PTGENDER",
    "PTEDUCAT",
    "MMSE_BL",
    "FAQ_BL",
    "GDS_BL",
    "MOCA_BL",
    "ADAS11_BL",
    "ADAS13_BL",
    "RAVLT_immediate",
    "RAVLT_forgetting",
    "RAVLT_delayed",
    "RAVLT_forget_rate",
    "DigitSpan",
    "TrailsB",
    "MMSE_FAQ_composite",
    "ADAS_MMSE_gap",
    "CDRSB_BL",
    "CDR_GLOBAL_BL",
]


def section(title):
    print("\n" + "-" * 62)
    print(f"  {title}")
    print("-" * 62)


def load_oasis2_data():
    print("=" * 62)
    print("  BUILD OASIS-2 VALIDATION FILE")
    print("=" * 62)

    if not os.path.isfile(INPUT_CSV):
        print(f"[ERROR] Input file not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    print(f"\n  Loaded : {INPUT_CSV}")
    print(f"  Shape  : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    return df


def inspect_oasis2_data(df):
    section("Group distribution")
    print(df["Group"].value_counts().to_string())

    section("CDR distribution")
    print(df["CDR"].value_counts().sort_index().to_string())

    section("Unique subjects")
    n_subjects = df["Subject ID"].nunique()
    n_rows = len(df)

    print(f"  {n_subjects} unique subjects, {n_rows} total rows")

    return n_subjects, n_rows


def map_gender(value):
    value = str(value).strip().upper()

    if value == "M":
        return 0

    if value == "F":
        return 1

    return np.nan


def assign_label(subject_rows):
    baseline_row = subject_rows.iloc[0]
    baseline_cdr = baseline_row["CDR"]

    if pd.isna(baseline_cdr) or float(baseline_cdr) != 0.5:
        return None, "not_mci"

    followup_rows = subject_rows.iloc[1:]
    followup_cdr = pd.to_numeric(followup_rows["CDR"], errors="coerce")

    group_values = (
        subject_rows["Group"]
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )

    converted_flag = any("convert" in group for group in group_values)

    followup_high = (
        (followup_cdr >= 1.0).any()
        if len(followup_cdr) > 0
        else False
    )

    followup_stable = (
        (followup_cdr <= 0.5).all()
        if len(followup_cdr) > 0
        else True
    )

    if converted_flag or followup_high:
        return 1, "labelled"

    if followup_stable:
        return 0, "labelled"

    return None, "ambiguous"


def build_subject_record(subject_id, subject_rows, label):
    baseline_row = subject_rows.iloc[0]

    return {
        "SUBJECT_ID": subject_id,
        "LABEL": label,

        # Available OASIS-2 features mapped to ADNI E4 names
        "PTGENDER": map_gender(baseline_row["M/F"]),
        "PTEDUCAT": pd.to_numeric(baseline_row["EDUC"], errors="coerce"),
        "MMSE_BL": pd.to_numeric(baseline_row["MMSE"], errors="coerce"),

        # Extra tracking columns for external validation reporting
        "Age": pd.to_numeric(baseline_row["Age"], errors="coerce"),
        "CDR_bl": baseline_row["CDR"],
        "n_visits": len(subject_rows),
        "max_CDR": pd.to_numeric(subject_rows["CDR"], errors="coerce").max(),
        "Group_bl": baseline_row["Group"],

        # Remaining ADNI E4 features unavailable in OASIS-2
        "FAQ_BL": np.nan,
        "GDS_BL": np.nan,
        "MOCA_BL": np.nan,
        "ADAS11_BL": np.nan,
        "ADAS13_BL": np.nan,
        "RAVLT_immediate": np.nan,
        "RAVLT_forgetting": np.nan,
        "RAVLT_delayed": np.nan,
        "RAVLT_forget_rate": np.nan,
        "DigitSpan": np.nan,
        "TrailsB": np.nan,
        "MMSE_FAQ_composite": np.nan,
        "ADAS_MMSE_gap": np.nan,
        "CDRSB_BL": np.nan,
        "CDR_GLOBAL_BL": np.nan,
    }


def build_validation_metadata(df):
    section("Labelling pMCI-like and sMCI-like subjects")

    print("  Criteria:")
    print("    Baseline CDR = 0.5")
    print("    pMCI-like: any follow-up CDR >= 1.0 or Group contains 'Converted'")
    print("    sMCI-like: all follow-up CDR <= 0.5")

    df = df.sort_values(["Subject ID", "Visit"]).reset_index(drop=True)

    records = []
    excluded_not_mci = 0
    excluded_ambiguous = 0

    for subject_id, subject_rows in df.groupby("Subject ID"):
        subject_rows = subject_rows.sort_values("Visit").reset_index(drop=True)

        label, status = assign_label(subject_rows)

        if status == "not_mci":
            excluded_not_mci += 1
            continue

        if status == "ambiguous":
            excluded_ambiguous += 1
            continue

        record = build_subject_record(
            subject_id,
            subject_rows,
            label,
        )

        records.append(record)

    output_df = pd.DataFrame(records)

    return output_df, excluded_not_mci, excluded_ambiguous


def print_label_report(output_df, total_subjects, excluded_not_mci, excluded_ambiguous):
    section("Label report")

    print(f"  Total unique subjects        : {total_subjects}")
    print(f"  Excluded baseline CDR != 0.5 : {excluded_not_mci}")
    print(f"  Excluded ambiguous CDR       : {excluded_ambiguous}")
    print("  " + "-" * 45)
    print(f"  Final labelled subjects      : {len(output_df)}")

    if len(output_df) == 0:
        print("\n[ERROR] No subjects labelled. Check CDR values in the input file.")
        sys.exit(1)

    p_count = int((output_df["LABEL"] == 1).sum())
    s_count = int((output_df["LABEL"] == 0).sum())

    print(f"    pMCI-like, LABEL=1        : {p_count}")
    print(f"    sMCI-like, LABEL=0        : {s_count}")

    prevalence = (output_df["LABEL"] == 1).mean() * 100
    print(f"\n  pMCI-like prevalence         : {prevalence:.1f}%")


def print_feature_coverage(output_df):
    section("Feature coverage in output file")

    available = 0

    for feature in E4_FEATURES:
        if feature in output_df.columns:
            coverage = output_df[feature].notna().mean() * 100

            if coverage > 10:
                flag = "available"
                available += 1
            else:
                flag = "absent, NaN -> imputed"

            print(f"  {feature:<28} {coverage:>6.1f}%  {flag}")

        else:
            print(f"  {feature:<28}   0.0%  absent, NaN -> imputed")

    absent = len(E4_FEATURES) - available

    print(f"\n  Available: {available}/{len(E4_FEATURES)} ADNI features")
    print(f"  Absent   : {absent}/{len(E4_FEATURES)} features")
    print(f"\n  NOTE: Only {available}/{len(E4_FEATURES)} ADNI features are available.")
    print("  The remaining features will be median-imputed from the ADNI train set.")
    print("  This feature mismatch explains why OASIS-2 validation is expected")
    print("  to perform near chance level.")

    return available


def save_output(output_df):
    section("Saving output")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"  Saved  : {OUTPUT_CSV}")
    print(f"  Rows   : {len(output_df)}")
    print(f"  Columns: {len(output_df.columns)}")


def print_next_step(available):
    print(
        f"""
Done
----
oasis2_validation_ready.csv is ready.

Next step:
python "Clinical Part/validate_oasis2.py"

Expected result:
AUC-ROC around 0.48, chance level.

Reason:
Only {available}/{len(E4_FEATURES)} ADNI features are available in OASIS-2.

Framing:
This should be reported as a feature-mismatch finding, not as a complete
generalisation failure.
{"=" * 62}
"""
    )


def main():
    df = load_oasis2_data()

    total_subjects, _ = inspect_oasis2_data(df)

    output_df, excluded_not_mci, excluded_ambiguous = build_validation_metadata(df)

    print_label_report(
        output_df,
        total_subjects,
        excluded_not_mci,
        excluded_ambiguous,
    )

    available = print_feature_coverage(output_df)

    save_output(output_df)

    print_next_step(available)


if __name__ == "__main__":
    main()