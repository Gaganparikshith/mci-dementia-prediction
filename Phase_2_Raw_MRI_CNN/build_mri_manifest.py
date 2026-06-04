#!/usr/bin/env python3
"""
README - build_mri_manifest.py
==============================

This script builds the MRI manifest for Phase 2 of the project.

It scans the raw MRI folders recursively, detects MRI files, extracts subject
IDs from file paths, matches each scan to available ADNI or OASIS-2 labels,
and saves a clean manifest CSV.

This script does not train any model and does not split the data. It only
creates a structured file list for later MRI preprocessing.

Inputs
------
Raw MRI folders:
    ADNI MRI folders
    OASIS-2 raw MRI folders

Label files:
    final_metadata.csv
        Used for ADNI labels.

    oasis2_validation_ready.csv
        Used for OASIS-2 labels.

"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# Raw MRI folders to scan
MRI_SEARCH_FOLDERS = [
    r"C:\Users\ASUS\Desktop\Research Resources\MRI DATA\Imaging_Cohort_3564___767_MRI_1",
    r"C:\Users\ASUS\Desktop\Research Resources\MRI DATA\Imaging_Cohort_3564___767_MRI_2",
    r"C:\Users\ASUS\Desktop\Research Resources\MRI DATA\OAS2_RAW_PART1",
    r"C:\Users\ASUS\Desktop\Research Resources\MRI DATA\OAS2_RAW_PART2",
]


# Metadata paths
METADATA_DIR = Path(
    r"C:\Users\ASUS\Desktop\Research Resources"
    r"\DementiaResearch\data\metadata"
)

ADNI_META_CSV = METADATA_DIR / "final_metadata.csv"
OASIS2_META_CSV = METADATA_DIR / "oasis2_validation_ready.csv"


# Output paths
OUTPUT_MANIFEST = METADATA_DIR / "mri_manifest.csv"
OUTPUT_UNMATCHED = METADATA_DIR / "mri_manifest_unmatched.csv"


# MRI file extensions to collect
MRI_EXTENSIONS = {
    ".nii",
    ".dcm",
    ".img",
    ".hdr",
    ".mgz",
    ".mgh",
}


# Folders to skip while scanning
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".venv",
    "venv",
    "node_modules",
}


def section(title):
    print()
    print("-" * 65)
    print(f"  {title}")
    print("-" * 65)


def detect_dataset(path_value):
    """
    Detect whether a path belongs to ADNI, OASIS-2, or an unknown dataset.
    """
    path_text = str(path_value).upper()

    if "IMAGING_COHORT" in path_text:
        return "ADNI"

    if "OAS2_RAW" in path_text:
        return "OASIS2"

    return "UNKNOWN"


def is_mri_file(file_name):
    """Return True if the file extension looks like an MRI file."""
    lower_name = file_name.lower()

    if lower_name.endswith(".nii.gz"):
        return True

    extension = os.path.splitext(lower_name)[1]

    return extension in MRI_EXTENSIONS


def get_file_extension(file_name):
    """Return .nii.gz as one extension, otherwise return the normal extension."""
    if file_name.lower().endswith(".nii.gz"):
        return ".nii.gz"

    return os.path.splitext(file_name)[1].lower()


def extract_adni_rid(path_value):
    """
    Extract ADNI RID from paths containing subject IDs like 002_S_0413.

    Returns:
        int RID if found, otherwise None.
    """
    match = re.search(
        r"\d{3}_S_(\d{4})",
        str(path_value),
    )

    if match:
        return int(match.group(1))

    return None


def extract_adni_subject_id(path_value):
    """Extract full ADNI subject ID such as 002_S_0413 from a path."""
    match = re.search(
        r"(\d{3}_S_\d{4})",
        str(path_value),
    )

    if match:
        return match.group(1)

    return None


def extract_oasis2_subject_id(path_value):
    """Extract OASIS-2 subject ID such as OAS2_0001 from a path."""
    match = re.search(
        r"(OAS2_\d{4})",
        str(path_value),
        re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    return None


def extract_visit(path_value):
    """
    Try to extract visit information from the path.

    Examples:
        bl, m06, m12, m18, m24, m36, m48, scmri, v1
    """
    match = re.search(
        r"[/\\_ ](bl|m06|m12|m18|m24|m36|m48|scmri|v\d+)[/\\_ ]",
        str(path_value),
        re.IGNORECASE,
    )

    if match:
        return match.group(1).lower()

    return ""


def find_first_column(columns, candidates):
    """Return the first matching column name from a candidate list."""
    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def load_adni_labels(csv_path):
    """
    Load ADNI labels from final_metadata.csv.

    Returns:
        Dictionary mapping RID to label.
    """
    if not csv_path.is_file():
        print(f"  WARNING: ADNI metadata not found: {csv_path}")
        print("           ADNI files will have no labels.")
        return {}

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.upper()

    rid_col = find_first_column(
        df.columns,
        [
            "RID",
            "PTID",
            "SUBJECT_ID",
        ],
    )

    label_col = find_first_column(
        df.columns,
        [
            "LABEL",
            "CONV_LABEL",
            "DX_LABEL",
            "DX",
        ],
    )

    if rid_col is None or label_col is None:
        print(f"  WARNING: Cannot find RID or LABEL in {csv_path.name}")
        print(f"           Columns found: {list(df.columns)}")
        return {}

    labels = {}

    for _, row in df.iterrows():
        try:
            rid = int(row[rid_col])
            label = str(row[label_col]).strip()
            labels[rid] = label
        except (ValueError, TypeError):
            continue

    print(f"  ADNI labels loaded: {len(labels)} subjects from {csv_path.name}")

    return labels


def load_oasis2_labels(csv_path):
    """
    Load OASIS-2 labels from oasis2_validation_ready.csv.

    Returns:
        Dictionary mapping OASIS-2 subject ID to label.
    """
    if not csv_path.is_file():
        print(f"  WARNING: OASIS-2 metadata not found: {csv_path}")
        print("           OASIS-2 files will have no labels.")
        return {}

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.upper()

    id_col = find_first_column(
        df.columns,
        [
            "SUBJECT_ID",
            "ID",
            "OAS_ID",
            "PTID",
        ],
    )

    label_col = find_first_column(
        df.columns,
        [
            "LABEL",
            "CONV_LABEL",
            "DX",
            "GROUP",
        ],
    )

    if id_col is None or label_col is None:
        print(f"  WARNING: Cannot find ID or LABEL in {csv_path.name}")
        print(f"           Columns found: {list(df.columns)}")
        return {}

    labels = {}

    for _, row in df.iterrows():
        subject_id = str(row[id_col]).strip().upper()
        label = str(row[label_col]).strip()
        labels[subject_id] = label

    print(f"  OASIS-2 labels loaded: {len(labels)} subjects from {csv_path.name}")

    return labels


def scan_adni_file(dir_path, file_name, full_path, file_extension, adni_labels):
    """Create one manifest record for an ADNI MRI file."""
    subject_id = extract_adni_subject_id(dir_path)
    rid_value = extract_adni_rid(dir_path)

    label = ""
    label_source = ""
    matched_label = False
    notes = ""

    if rid_value is not None and rid_value in adni_labels:
        label = adni_labels[rid_value]
        label_source = "final_metadata.csv"
        matched_label = True
    elif rid_value is None:
        notes = "RID not found in path"
    else:
        notes = f"RID {rid_value} not in final_metadata.csv"

    return {
        "dataset": "ADNI",
        "subject_id": subject_id or "",
        "rid": str(rid_value) if rid_value is not None else "",
        "session_id": "",
        "visit": extract_visit(dir_path),
        "scan_path": full_path,
        "file_name": file_name,
        "file_ext": file_extension,
        "label": label,
        "label_source": label_source,
        "matched_label": matched_label,
        "notes": notes,
    }


def scan_oasis2_file(dir_path, file_name, full_path, file_extension, oasis2_labels):
    """Create one manifest record for an OASIS-2 MRI file."""
    subject_id = extract_oasis2_subject_id(dir_path)

    if subject_id is None:
        subject_id = extract_oasis2_subject_id(file_name)

    label = ""
    label_source = ""
    matched_label = False
    notes = ""

    if subject_id and subject_id in oasis2_labels:
        label = oasis2_labels[subject_id]
        label_source = "oasis2_validation_ready.csv"
        matched_label = True
    elif subject_id:
        notes = f"{subject_id} not in oasis2_validation_ready.csv"
    else:
        notes = "Subject ID not found in path"

    return {
        "dataset": "OASIS2",
        "subject_id": subject_id or "",
        "rid": "",
        "session_id": "",
        "visit": extract_visit(dir_path),
        "scan_path": full_path,
        "file_name": file_name,
        "file_ext": file_extension,
        "label": label,
        "label_source": label_source,
        "matched_label": matched_label,
        "notes": notes,
    }


def scan_unknown_file(dir_path, file_name, full_path, file_extension):
    """Create one manifest record for an MRI file from an unknown dataset."""
    return {
        "dataset": "UNKNOWN",
        "subject_id": "",
        "rid": "",
        "session_id": "",
        "visit": extract_visit(dir_path),
        "scan_path": full_path,
        "file_name": file_name,
        "file_ext": file_extension,
        "label": "",
        "label_source": "",
        "matched_label": False,
        "notes": "Dataset not detected from path",
    }


def scan_folder(root_path, adni_labels, oasis2_labels):
    """
    Recursively scan one folder and return one record per MRI file found.
    """
    records = []
    dataset_hint = detect_dataset(root_path)

    for dir_path, dir_names, file_names in os.walk(root_path):
        dir_names[:] = [
            folder
            for folder in dir_names
            if folder not in SKIP_DIRS and not folder.startswith(".")
        ]

        for file_name in file_names:
            if not is_mri_file(file_name):
                continue

            full_path = os.path.join(dir_path, file_name)
            file_extension = get_file_extension(file_name)

            dataset = detect_dataset(dir_path)

            if dataset == "UNKNOWN":
                dataset = dataset_hint

            if dataset == "ADNI":
                record = scan_adni_file(
                    dir_path,
                    file_name,
                    full_path,
                    file_extension,
                    adni_labels,
                )

            elif dataset == "OASIS2":
                record = scan_oasis2_file(
                    dir_path,
                    file_name,
                    full_path,
                    file_extension,
                    oasis2_labels,
                )

            else:
                record = scan_unknown_file(
                    dir_path,
                    file_name,
                    full_path,
                    file_extension,
                )

            records.append(record)

    return records


def print_manifest_summary(df):
    print()
    print("=" * 65)
    print("  MANIFEST SUMMARY")
    print("=" * 65)

    total_files = len(df)
    matched_files = int(df["matched_label"].sum())
    unmatched_files = total_files - matched_files

    print(f"\n  Total MRI files found : {total_files}")

    print("\n  Files by dataset:")
    for dataset in ["ADNI", "OASIS2", "UNKNOWN"]:
        count = int((df["dataset"] == dataset).sum())
        print(f"    {dataset:<10}: {count}")

    print(f"\n  Matched to labels     : {matched_files}")
    print(f"  Unmatched             : {unmatched_files}")

    matched_df = df[df["matched_label"]]

    if len(matched_df) > 0:
        print("\n  Label distribution:")
        label_counts = matched_df["label"].value_counts()

        for label, count in label_counts.items():
            print(f"    {str(label):<20}: {count} files")

    print("\n  File type distribution:")
    extension_counts = df["file_ext"].value_counts()

    for extension, count in extension_counts.items():
        print(f"    {str(extension):<15}: {count} files")

    id_df = df[df["subject_id"] != ""]
    unique_subjects = id_df["subject_id"].nunique()

    print(f"\n  Unique subjects       : {unique_subjects}")

    if unique_subjects > 0:
        scan_counts = id_df.groupby("subject_id").size()
        multi_scan_subjects = int((scan_counts > 1).sum())

        print(f"  Subjects with >1 scan : {multi_scan_subjects}")

        if multi_scan_subjects > 0:
            print("  NOTE: Multiple scans per subject will be handled during preprocessing.")

    if total_files > 0:
        match_rate = matched_files / total_files * 100

        print()

        if match_rate < 50:
            print(f"  WARNING: Match rate is only {match_rate:.1f}%.")
            print("           Check whether folder subject IDs match final_metadata.csv.")
        elif match_rate < 80:
            print(f"  WARNING: Match rate is {match_rate:.1f}%. Review unmatched manifest.")
        else:
            print(f"  OK: Match rate is {match_rate:.1f}%.")


def save_manifest_outputs(df):
    section("Saving outputs")

    df.to_csv(
        OUTPUT_MANIFEST,
        index=False,
    )

    print(f"  Saved full manifest      : {OUTPUT_MANIFEST}")
    print(f"  Rows: {len(df)}")

    unmatched_df = df[~df["matched_label"]].copy()

    unmatched_df.to_csv(
        OUTPUT_UNMATCHED,
        index=False,
    )

    print(f"  Saved unmatched manifest : {OUTPUT_UNMATCHED}")
    print(f"  Rows: {len(unmatched_df)}")


def main():
    print("=" * 65)
    print("  BUILD MRI MANIFEST - Phase 2")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    section("Loading label dictionaries")

    adni_labels = load_adni_labels(ADNI_META_CSV)
    oasis2_labels = load_oasis2_labels(OASIS2_META_CSV)

    section("Scanning MRI folders")

    all_records = []

    for folder in MRI_SEARCH_FOLDERS:
        if not os.path.isdir(folder):
            print(f"  SKIP: Folder not found: {folder}")
            continue

        dataset = detect_dataset(folder)

        print(f"\n  Scanning: {folder}")
        print(f"  Dataset : {dataset}")

        records = scan_folder(
            folder,
            adni_labels,
            oasis2_labels,
        )

        all_records.extend(records)

        matched_count = sum(
            1
            for record in records
            if record["matched_label"]
        )

        print(f"  Found   : {len(records)} MRI files")
        print(f"  Matched : {matched_count}")

    if not all_records:
        print("\n[ERROR] No MRI files found in any search folder.")
        print("        Check that the folders above exist and contain MRI data.")
        sys.exit(1)

    manifest_df = pd.DataFrame(all_records)

    before_dedup = len(manifest_df)

    manifest_df = manifest_df.drop_duplicates(
        subset=["scan_path"],
    )

    duplicates_removed = before_dedup - len(manifest_df)

    if duplicates_removed > 0:
        print(f"\n  Removed {duplicates_removed} duplicate file paths.")

    print_manifest_summary(manifest_df)

    save_manifest_outputs(manifest_df)

    section("Next step")
    print("  Run: python MRI/check_mri_manifest.py")
    print("       to verify the manifest before CNN preprocessing.")
    print("=" * 65)


if __name__ == "__main__":
    main()