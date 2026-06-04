"""
MRI/check_mri_manifest.py
==========================
Phase 2 — Step 2 of 2

Loads mri_manifest.csv and runs a full verification check
before any preprocessing or CNN training begins.

Prints:
  - Total rows
  - Unique subjects
  - Unique labelled subjects
  - Scans per subject
  - Label distribution
  - Dataset distribution
  - File extension distribution
  - Missing labels
  - Examples of matched and unmatched rows
  - Data leakage warning (image-level vs subject-level split)
  - GO / NO-GO verdict

Saves:
  Outputs/tables/mri_manifest_summary.csv

Run:
  python MRI/check_mri_manifest.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
METADATA_DIR   = r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch\data\metadata"
SPLITS_DIR     = r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch\data\splits"
RESULTS_DIR    = r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch\results"

MANIFEST_CSV       = os.path.join(METADATA_DIR, "mri_manifest.csv")
TRAIN_SUBJECTS_CSV = os.path.join(SPLITS_DIR,   "train_subjects.csv")
TEST_SUBJECTS_CSV  = os.path.join(SPLITS_DIR,    "test_subjects.csv")
OUTPUT_SUMMARY_CSV = os.path.join(RESULTS_DIR,   "tables", "mri_manifest_summary.csv")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section(title):
    print(f"\n── {title} {'─' * max(0, 57 - len(title))}")


def ok(msg):
    print(f"  ✓  {msg}")


def warn(msg):
    print(f"  ⚠  [WARNING] {msg}")


def err(msg):
    print(f"  ✗  [ERROR]   {msg}")


def load_split_ids(csv_path):
    """Load a set of subject IDs from a split CSV."""
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.upper()
    # Try common column names
    for col in ["RID", "SUBJECT_ID", "PTID", "ID"]:
        if col in df.columns:
            return set(df[col].astype(str).str.strip())
    # Fall back to first column
    return set(df.iloc[:, 0].astype(str).str.strip())


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  CHECK MRI MANIFEST — Phase 2")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Load manifest ─────────────────────────────────────────────
    if not os.path.isfile(MANIFEST_CSV):
        print(f"\n[ERROR] Manifest not found: {MANIFEST_CSV}")
        print("        Run MRI/build_mri_manifest.py first.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_CSV)
    print(f"\n  Loaded: {MANIFEST_CSV}")
    print(f"  Rows  : {len(df)}   Columns: {len(df.columns)}")

    # Check required columns exist
    required = {"dataset", "subject_id", "scan_path", "file_ext",
                "label", "matched_label", "notes"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"\n[ERROR] Manifest is missing columns: {missing_cols}")
        print("        Re-run build_mri_manifest.py to regenerate.")
        sys.exit(1)

    # Normalise matched_label to bool
    df["matched_label"] = (
        df["matched_label"].astype(str).str.lower().isin(["true", "1", "yes"])
    )

    issues = []   # collect issue tags for final verdict
    summary_rows = []

    # ─────────────────────────────────────────
    # CHECK 1 — Basic counts
    # ─────────────────────────────────────────
    section("1. BASIC COUNTS")

    total     = len(df)
    matched   = df["matched_label"].sum()
    unmatched = total - matched
    match_pct = matched / total * 100 if total > 0 else 0

    print(f"  Total rows             : {total}")
    print(f"  Rows with label        : {matched}  ({match_pct:.1f}%)")
    print(f"  Rows without label     : {unmatched}")

    summary_rows.append({"metric": "total_rows",    "value": total})
    summary_rows.append({"metric": "matched_rows",  "value": matched})
    summary_rows.append({"metric": "unmatched_rows","value": unmatched})

    if match_pct == 0:
        err("No files matched to any label.")
        issues.append("no_labels")
    elif match_pct < 50:
        warn(f"Match rate is only {match_pct:.1f}%.")
        warn("Check that RIDs in folder names match final_metadata.csv.")
        issues.append("low_match_rate")
    elif match_pct < 80:
        warn(f"Match rate is {match_pct:.1f}%. Review mri_manifest_unmatched.csv.")
    else:
        ok(f"Match rate: {match_pct:.1f}%")

    # ─────────────────────────────────────────
    # CHECK 2 — Unique subjects
    # ─────────────────────────────────────────
    section("2. UNIQUE SUBJECTS")

    df_with_id      = df[df["subject_id"].astype(str).str.strip() != ""]
    n_unique        = df_with_id["subject_id"].nunique()
    n_no_id         = (df["subject_id"].astype(str).str.strip() == "").sum()

    df_labelled     = df[df["matched_label"]]
    n_labelled_subj = df_labelled["subject_id"].nunique()

    print(f"  Rows with subject_id   : {len(df_with_id)}")
    print(f"  Rows without subject_id: {n_no_id}")
    print(f"  Unique subjects        : {n_unique}")
    print(f"  Unique labelled subj.  : {n_labelled_subj}")

    summary_rows.append({"metric": "unique_subjects",         "value": n_unique})
    summary_rows.append({"metric": "unique_labelled_subjects", "value": n_labelled_subj})

    if n_labelled_subj < 50:
        err(f"Only {n_labelled_subj} labelled subjects — too few for CNN training.")
        issues.append("too_few_subjects")
    elif n_labelled_subj < 150:
        warn(f"Only {n_labelled_subj} labelled subjects. CNN may underfit.")
    else:
        ok(f"{n_labelled_subj} labelled subjects available.")

    # ─────────────────────────────────────────
    # CHECK 3 — Label distribution
    # ─────────────────────────────────────────
    section("3. LABEL DISTRIBUTION")

    if len(df_labelled) > 0:
        label_counts = df_labelled["label"].value_counts()
        for lbl, cnt in label_counts.items():
            pct = cnt / matched * 100
            print(f"  {str(lbl):<20}: {cnt:>6} files  ({pct:.1f}%)")
            summary_rows.append({"metric": f"label_{lbl}", "value": cnt})

        if len(label_counts) >= 2:
            minority = label_counts.min()
            majority = label_counts.max()
            ratio    = majority / minority if minority > 0 else float("inf")
            if ratio > 4:
                warn(f"Class imbalance {ratio:.1f}:1. Use SMOTE or class weights in CNN.")
                issues.append("class_imbalance")
            else:
                ok(f"Class balance ratio: {ratio:.1f}:1")
    else:
        warn("No labelled rows — cannot check label distribution.")

    # ─────────────────────────────────────────
    # CHECK 4 — Dataset distribution
    # ─────────────────────────────────────────
    section("4. DATASET DISTRIBUTION")

    dataset_counts = df["dataset"].value_counts()
    for ds, cnt in dataset_counts.items():
        print(f"  {str(ds):<12}: {cnt:>6} files")
        summary_rows.append({"metric": f"dataset_{ds}", "value": cnt})

    if "UNKNOWN" in dataset_counts and dataset_counts["UNKNOWN"] > 0:
        warn(f"{dataset_counts['UNKNOWN']} files with unknown dataset.")

    # ─────────────────────────────────────────
    # CHECK 5 — File extension distribution
    # ─────────────────────────────────────────
    section("5. FILE EXTENSION DISTRIBUTION")

    ext_counts = df["file_ext"].value_counts()
    for ext, cnt in ext_counts.items():
        print(f"  {str(ext):<15}: {cnt:>6} files")
        summary_rows.append({"metric": f"ext_{ext}", "value": cnt})

    # ─────────────────────────────────────────
    # CHECK 6 — Scans per subject
    # ─────────────────────────────────────────
    section("6. SCANS PER SUBJECT")

    if n_unique > 0:
        counts_per_subj = df_with_id.groupby("subject_id").size()
        single          = (counts_per_subj == 1).sum()
        multi           = (counts_per_subj > 1).sum()
        max_scans       = counts_per_subj.max()
        mean_scans      = counts_per_subj.mean()

        print(f"  Subjects with 1 scan   : {single}")
        print(f"  Subjects with >1 scan  : {multi}")
        print(f"  Max scans per subject  : {max_scans}")
        print(f"  Mean scans per subject : {mean_scans:.2f}")

        summary_rows.append({"metric": "subjects_single_scan", "value": single})
        summary_rows.append({"metric": "subjects_multi_scan",  "value": multi})

        if multi > 0:
            warn(f"{multi} subjects have multiple scans.")
            print(f"  During preprocessing, ONE scan per subject will be selected")
            print(f"  (baseline/first visit). All scans for one subject will stay")
            print(f"  in the same train or test set — no image-level leakage.")

            # Show top 5 examples
            top5 = counts_per_subj[counts_per_subj > 1].sort_values(ascending=False).head(5)
            print(f"\n  Top multi-scan subjects:")
            for sid, cnt in top5.items():
                print(f"    {sid}: {cnt} scans")

    # ─────────────────────────────────────────
    # CHECK 7 — Data leakage check
    # ─────────────────────────────────────────
    section("7. DATA LEAKAGE CHECK")

    train_ids = load_split_ids(TRAIN_SUBJECTS_CSV)
    test_ids  = load_split_ids(TEST_SUBJECTS_CSV)

    if train_ids is None or test_ids is None:
        warn("train_subjects.csv or test_subjects.csv not found.")
        print(f"  Expected at: {SPLITS_DIR}")
        print(f"  These will be created when preprocessing runs.")
        print(f"  Ensure subject-level split is used (not image-level).")
        issues.append("no_splits_yet")
    else:
        mri_subject_ids = set(df_with_id["subject_id"].astype(str).str.strip())
        train_in_mri    = train_ids & mri_subject_ids
        test_in_mri     = test_ids  & mri_subject_ids
        overlap         = train_ids & test_ids  # should always be empty

        print(f"  Train subjects in manifest : {len(train_in_mri)}")
        print(f"  Test subjects in manifest  : {len(test_in_mri)}")
        print(f"  Overlap (train ∩ test)      : {len(overlap)}")

        if len(overlap) > 0:
            err(f"LEAKAGE: {len(overlap)} subjects appear in BOTH train and test!")
            err("Re-run build_dataset.py to fix the splits immediately.")
            issues.append("data_leakage")
        else:
            ok("No subject-level leakage between train and test splits.")

    # ─────────────────────────────────────────
    # CHECK 8 — Example rows
    # ─────────────────────────────────────────
    section("8. EXAMPLE ROWS")

    matched_examples = df[df["matched_label"]].head(3)
    if not matched_examples.empty:
        print("  Matched examples:")
        for _, row in matched_examples.iterrows():
            print(f"    subject={row['subject_id']}  label={row['label']}"
                  f"  ext={row['file_ext']}  file={row['file_name']}")

    unmatched_examples = df[~df["matched_label"]].head(3)
    if not unmatched_examples.empty:
        print(f"\n  Unmatched examples:")
        for _, row in unmatched_examples.iterrows():
            print(f"    subject={row['subject_id']}  ext={row['file_ext']}"
                  f"  note={row['notes']}")

    # ─────────────────────────────────────────
    # SAVE SUMMARY
    # ─────────────────────────────────────────
    section("SAVING SUMMARY")

    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_CSV), exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"  [SAVED] {OUTPUT_SUMMARY_CSV}")

    # ─────────────────────────────────────────
    # GO / NO-GO VERDICT
    # ─────────────────────────────────────────
    section("GO / NO-GO VERDICT")

    blocking = {"no_labels", "too_few_subjects", "data_leakage"}
    has_blocking = bool(set(issues) & blocking)

    if not issues:
        print("\n  ✅  ALL CHECKS PASSED")
        print("      Manifest is ready for preprocessing.")
        print("      Next: python MRI/preprocess_mri_2d_slices.py")
    elif has_blocking:
        print("\n  ❌  CRITICAL ISSUES — DO NOT PROCEED TO CNN")
        for issue in issues:
            if issue in blocking:
                print(f"      → {issue}")
        print("      Fix the issues above and re-run build_mri_manifest.py.")
    else:
        print("\n  ⚠   WARNINGS PRESENT — review before proceeding")
        for issue in issues:
            print(f"      → {issue}")
        print("      If acceptable, proceed to preprocess_mri_2d_slices.py.")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()