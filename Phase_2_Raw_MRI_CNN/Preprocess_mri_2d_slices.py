#!/usr/bin/env python3
"""
README - preprocess_mri_2d_slices.py
====================================

This script preprocesses ADNI MRI scans for Phase 2 CNN training.

It reads the MRI manifest, selects labelled ADNI subjects, loads one usable
MRI volume per subject, extracts axial slices from the hippocampal region, and
saves them as 224 x 224 PNG images.

Important
---------
This script does not train any model.

It only converts raw MRI files into 2D image slices that can be used later by
the CNN training script.

Processing logic
----------------
1. Load mri_manifest.csv.
2. Keep only labelled ADNI subjects.
3. Process one MRI volume per subject.
4. Prefer DICOM files because they were more reliable in this dataset.
5. Fall back to NIfTI files if DICOM is not available.
6. Extract 20 axial slices from 40% to 65% of the volume depth.
7. Normalise each slice and resize it to 224 x 224.
8. Save the slices into pMCI and sMCI folders.

Fixes included
--------------
1. DICOM is preferred over NIfTI when both are available.
2. 4D NIfTI volumes are collapsed to 3D before slice extraction.
3. Compressed DICOM transfer syntax is handled where possible.
4. Windows long paths are handled using the extended path prefix.

Run
---
python MRI/preprocess_mri_2d_slices.py

Next step
---------
python MRI/train_mri_cnn_2d.py
"""

import os
import re
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


# Configuration
N_SLICES = 20
IMAGE_SIZE = 224

SLICE_START_PCT = 0.40
SLICE_END_PCT = 0.65

MIN_DICOM_FOR_VOLUME = 30


# Paths
RESEARCH_ROOT = r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch"

METADATA_DIR = os.path.join(
    RESEARCH_ROOT,
    "data",
    "metadata",
)

SPLITS_DIR = os.path.join(
    RESEARCH_ROOT,
    "data",
    "splits",
)

MANIFEST_CSV = os.path.join(
    METADATA_DIR,
    "mri_manifest.csv",
)

OUTPUT_ROOT = os.path.join(
    RESEARCH_ROOT,
    "data",
    "preprocessed",
    "ADNI_2D_Slices",
)

OUTPUT_PMCI = os.path.join(
    OUTPUT_ROOT,
    "pMCI",
)

OUTPUT_SMCI = os.path.join(
    OUTPUT_ROOT,
    "sMCI",
)

SLICE_MANIFEST = os.path.join(
    METADATA_DIR,
    "mri_2d_slices_manifest.csv",
)

FAILURE_LOG_CSV = os.path.join(
    METADATA_DIR,
    "preprocess_failures.csv",
)

LOG_PATH = os.path.join(
    RESEARCH_ROOT,
    "results",
    "logs",
    "preprocess_mri_v2.log",
)


def make_long_path(path_value):
    """
    Return a Windows extended-length path if the path is very long.

    ADNI folder paths can exceed the normal Windows MAX_PATH limit.
    """
    path_value = str(path_value)

    if os.name == "nt" and len(path_value) > 250:
        absolute_path = os.path.abspath(path_value)

        if not absolute_path.startswith("\\\\?\\"):
            return "\\\\?\\" + absolute_path

    return path_value


def check_dependencies():
    """Check that the required MRI/image packages are installed."""
    required_packages = [
        ("pydicom", "pydicom"),
        ("nibabel", "nibabel"),
        ("Pillow", "PIL"),
    ]

    missing_packages = []

    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\n[ERROR] Missing packages: {missing_packages}")
        print(f"Install them using: pip install {' '.join(missing_packages)}")
        sys.exit(1)


def extract_date_from_path(path_value):
    """Extract YYYY-MM-DD from a path if present."""
    match = re.search(
        r"(\d{4}-\d{2}-\d{2})",
        str(path_value),
    )

    if match:
        return match.group(1)

    return ""


def select_session_rows(subject_df, file_extension):
    """
    Select rows for one subject and one file type.

    The script first tries to use the earliest dated session as baseline. If
    that session has too few DICOM slices, it falls back to the largest session.
    """
    selected_df = subject_df[subject_df["file_ext"] == file_extension].copy()

    if len(selected_df) == 0:
        return selected_df

    selected_df["_date"] = selected_df["scan_path"].apply(extract_date_from_path)

    dates = sorted(
        date
        for date in selected_df["_date"].unique()
        if date != ""
    )

    if not dates:
        return selected_df

    baseline_df = selected_df[selected_df["_date"] == dates[0]]

    if len(baseline_df) < MIN_DICOM_FOR_VOLUME and len(dates) > 1:
        session_sizes = selected_df.groupby("_date").size()
        best_date = session_sizes.idxmax()
        baseline_df = selected_df[selected_df["_date"] == best_date]

    return baseline_df


def load_dicom_volume(dicom_paths):
    """
    Load DICOM slices and stack them into a 3D volume.

    Returns:
        3D float32 array with shape H x W x D, or None if loading fails.
    """
    import pydicom

    slices = []

    for path in dicom_paths:
        try:
            dicom = pydicom.dcmread(
                make_long_path(path),
                force=True,
            )

            if hasattr(dicom, "file_meta"):
                transfer_syntax = getattr(
                    dicom.file_meta,
                    "TransferSyntaxUID",
                    None,
                )

                uncompressed_syntaxes = [
                    "1.2.840.10008.1.2",
                    "1.2.840.10008.1.2.1",
                    "1.2.840.10008.1.2.2",
                ]

                if transfer_syntax and str(transfer_syntax) not in uncompressed_syntaxes:
                    try:
                        dicom.decompress()
                    except Exception:
                        pass

            if not hasattr(dicom, "pixel_array"):
                continue

            pixel_array = dicom.pixel_array.astype(np.float32)
            instance_number = int(getattr(dicom, "InstanceNumber", 0))

            slices.append(
                (
                    instance_number,
                    pixel_array,
                )
            )

        except Exception:
            continue

    if len(slices) < MIN_DICOM_FOR_VOLUME:
        return None

    slices.sort(key=lambda item: item[0])

    volume = np.stack(
        [slice_data for _, slice_data in slices],
        axis=-1,
    )

    return volume.astype(np.float32)


def load_nifti_volume(nifti_path):
    """
    Load a NIfTI or Analyze volume and return a 3D array.

    4D volumes are reduced to the first timepoint.
    """
    import nibabel as nib

    try:
        image = nib.load(make_long_path(nifti_path))
        data = np.asarray(image.dataobj, dtype=np.float32)

        while data.ndim > 3:
            data = data[..., 0]

        if data.ndim != 3:
            return None

        smallest_axis = int(np.argmin(data.shape))

        axes = list(range(3))
        axes.remove(smallest_axis)
        axes.append(smallest_axis)

        data = np.transpose(data, axes)

        if data.shape[2] < N_SLICES:
            return None

        return data

    except Exception:
        return None


def normalise_slice(slice_array):
    """
    Clip intensities to the 1st and 99th percentiles and scale to uint8.
    """
    slice_array = slice_array.astype(np.float32)

    lower = np.percentile(slice_array, 1)
    upper = np.percentile(slice_array, 99)

    if upper <= lower:
        return np.zeros(slice_array.shape, dtype=np.uint8)

    slice_array = (slice_array - lower) / (upper - lower)
    slice_array = np.clip(slice_array * 255.0, 0, 255)

    return slice_array.astype(np.uint8)


def resize_and_save(slice_array, save_path, image_size):
    """Resize a 2D slice and save it as PNG."""
    from PIL import Image

    image = Image.fromarray(
        slice_array,
        mode="L",
    )

    image = image.resize(
        (
            image_size,
            image_size,
        ),
        Image.LANCZOS,
    )

    image.save(save_path)


def extract_slice_indices(depth, n_slices, start_pct, end_pct):
    """
    Return evenly spaced slice indices from the selected volume depth range.
    """
    start = int(depth * start_pct)
    end = int(depth * end_pct)

    end = max(
        end,
        start + 1,
    )

    indices = np.linspace(
        start,
        end - 1,
        n_slices,
        dtype=int,
    )

    return np.clip(
        indices,
        0,
        depth - 1,
    )


def load_subject_volume(subject_df):
    """
    Load one usable MRI volume for a subject.

    Priority:
    1. DICOM
    2. NIfTI
    3. Analyze .hdr
    """
    volume = None
    used_extension = None

    dicom_rows = select_session_rows(
        subject_df,
        ".dcm",
    )

    if len(dicom_rows) >= MIN_DICOM_FOR_VOLUME:
        dicom_paths = dicom_rows["scan_path"].tolist()
        volume = load_dicom_volume(dicom_paths)

        if volume is not None:
            used_extension = ".dcm"

    if volume is None:
        for extension in [
            ".nii.gz",
            ".nii",
        ]:
            nifti_rows = subject_df[subject_df["file_ext"] == extension]

            if len(nifti_rows) == 0:
                continue

            nifti_rows = nifti_rows.copy()
            nifti_rows["_date"] = nifti_rows["scan_path"].apply(extract_date_from_path)

            dates = sorted(
                date
                for date in nifti_rows["_date"].unique()
                if date
            )

            if dates:
                best_path = nifti_rows[nifti_rows["_date"] == dates[0]].iloc[0]["scan_path"]
            else:
                best_path = nifti_rows.iloc[0]["scan_path"]

            volume = load_nifti_volume(best_path)

            if volume is not None:
                used_extension = extension
                break

    if volume is None:
        hdr_rows = subject_df[subject_df["file_ext"] == ".hdr"]

        if len(hdr_rows) > 0:
            best_path = hdr_rows.iloc[0]["scan_path"]
            volume = load_nifti_volume(best_path)

            if volume is not None:
                used_extension = ".hdr"

    return volume, used_extension


def process_subject(subject_id, subject_df, label, output_dir):
    """
    Process one subject and save 2D MRI slices.

    Returns:
        saved_rows, error
    """
    volume, used_extension = load_subject_volume(subject_df)

    if volume is None:
        available_extensions = subject_df["file_ext"].value_counts().to_dict()
        return None, f"Could not load volume. Extensions available: {available_extensions}"

    if volume.ndim != 3:
        return None, f"Volume not 3D after loading: shape={volume.shape}"

    depth = volume.shape[2]

    if depth < N_SLICES:
        return None, f"Volume depth {depth} < N_SLICES {N_SLICES}"

    slice_indices = extract_slice_indices(
        depth,
        N_SLICES,
        SLICE_START_PCT,
        SLICE_END_PCT,
    )

    safe_subject_id = (
        str(subject_id)
        .replace("/", "_")
        .replace("\\", "_")
    )

    saved_rows = []

    for slice_number, slice_index in enumerate(slice_indices):
        raw_slice = volume[:, :, slice_index]
        normalised_slice = normalise_slice(raw_slice)

        file_name = f"{safe_subject_id}_slice_{slice_number:03d}.png"
        save_path = os.path.join(
            output_dir,
            file_name,
        )

        resize_and_save(
            normalised_slice,
            save_path,
            IMAGE_SIZE,
        )

        saved_rows.append(
            {
                "subject_id": subject_id,
                "label": label,
                "label_class": "pMCI" if str(label) in ["1", "1.0"] else "sMCI",
                "slice_path": save_path,
                "slice_index": slice_number,
                "volume_depth": depth,
                "volume_shape": str(volume.shape),
                "loaded_from": used_extension,
            }
        )

    return saved_rows, None


def load_manifest():
    if not os.path.isfile(MANIFEST_CSV):
        print(f"[ERROR] {MANIFEST_CSV} not found.")
        print("Run build_mri_manifest.py first.")
        sys.exit(1)

    df = pd.read_csv(
        MANIFEST_CSV,
        low_memory=False,
    )

    df["matched_label"] = (
        df["matched_label"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )

    return df


def get_labelled_adni_subjects(manifest_df):
    adni_df = manifest_df[
        (manifest_df["dataset"] == "ADNI")
        & manifest_df["matched_label"]
    ].copy()

    return adni_df


def load_existing_slice_manifest():
    if not os.path.isfile(SLICE_MANIFEST):
        return set(), None

    existing_df = pd.read_csv(SLICE_MANIFEST)
    existing_subjects = set(existing_df["subject_id"].unique())

    return existing_subjects, existing_df


def save_log(log_lines):
    os.makedirs(
        os.path.dirname(LOG_PATH),
        exist_ok=True,
    )

    with open(LOG_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(log_lines))


def main():
    check_dependencies()

    os.makedirs(OUTPUT_PMCI, exist_ok=True)
    os.makedirs(OUTPUT_SMCI, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    log_lines = []

    def log(message):
        print(message)
        log_lines.append(message)

    log("=" * 65)
    log("  PREPROCESS MRI - 2D AXIAL SLICES")
    log(f"  Run at      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  N_SLICES    : {N_SLICES}")
    log(f"  IMAGE_SIZE  : {IMAGE_SIZE} x {IMAGE_SIZE}")
    log(f"  SLICE_RANGE : {SLICE_START_PCT * 100:.0f}% to {SLICE_END_PCT * 100:.0f}% of volume depth")
    log("=" * 65)

    log("\n-- Loading manifest ---------------------------------------")

    manifest_df = load_manifest()

    adni_df = get_labelled_adni_subjects(manifest_df)
    n_subjects = adni_df["subject_id"].nunique()

    log(f"  Total manifest rows  : {len(manifest_df)}")
    log(f"  ADNI labelled rows   : {len(adni_df)}")
    log(f"  Unique ADNI subjects : {n_subjects}")

    if n_subjects == 0:
        log("[ERROR] No labelled ADNI subjects found.")
        save_log(log_lines)
        sys.exit(1)

    existing_subjects, existing_df = load_existing_slice_manifest()

    if existing_subjects:
        log(f"\n  Existing slice manifest found: {len(existing_subjects)} subjects already processed.")
        log("  These subjects will be skipped.")
        log("  Delete mri_2d_slices_manifest.csv to reprocess everything.")

    log("\n-- Processing subjects ------------------------------------")

    all_slice_rows = []
    failure_rows = []

    successful_count = 0
    skipped_count = 0
    failed_count = 0

    subjects = adni_df["subject_id"].unique()
    total_subjects = len(subjects)

    for index, subject_id in enumerate(subjects):
        if subject_id in existing_subjects:
            skipped_count += 1
            continue

        subject_df = adni_df[adni_df["subject_id"] == subject_id]

        label = subject_df["label"].iloc[0]

        output_dir = OUTPUT_PMCI if str(label) in ["1", "1.0"] else OUTPUT_SMCI

        if (index + 1) % 50 == 0 or index == 0:
            label_name = "pMCI" if str(label) in ["1", "1.0"] else "sMCI"
            log(f"  [{index + 1:>4}/{total_subjects}] {subject_id} ({label_name})")

        rows, error = process_subject(
            subject_id,
            subject_df,
            label,
            output_dir,
        )

        if error:
            failed_count += 1

            failure_rows.append(
                {
                    "subject_id": subject_id,
                    "label": label,
                    "error": error,
                }
            )

        else:
            successful_count += 1
            all_slice_rows.extend(rows)

    if existing_subjects and existing_df is not None:
        if all_slice_rows:
            new_df = pd.DataFrame(all_slice_rows)

            combined_df = pd.concat(
                [
                    existing_df,
                    new_df,
                ],
                ignore_index=True,
            )

        else:
            combined_df = existing_df

    else:
        combined_df = pd.DataFrame(all_slice_rows) if all_slice_rows else pd.DataFrame()

    log("\n-- Saving files -------------------------------------------")

    if len(combined_df) > 0:
        combined_df.to_csv(
            SLICE_MANIFEST,
            index=False,
        )

        log(f"  Slice manifest saved : {SLICE_MANIFEST}")
        log(f"  Total slice rows     : {len(combined_df)}")

        class_counts = (
            combined_df
            .drop_duplicates("subject_id")["label_class"]
            .value_counts()
        )

        for class_name, count in class_counts.items():
            log(f"  {class_name}: {count} subjects")

    else:
        log("  WARNING: No slices were saved. Check the failure log.")

    if failure_rows:
        failure_df = pd.DataFrame(failure_rows)

        failure_df.to_csv(
            FAILURE_LOG_CSV,
            index=False,
        )

        log(f"  Failure log saved    : {FAILURE_LOG_CSV}")
        log(f"  Failed subjects      : {len(failure_rows)}")

        log("\n  Failure breakdown:")

        for error_text, count in failure_df["error"].value_counts().items():
            log(f"    {count:>4}x {error_text[:80]}")

    log("\n" + "=" * 65)
    log("  PREPROCESSING COMPLETE")
    log("=" * 65)

    log(f"  Total subjects          : {total_subjects}")
    log(f"  Skipped already done    : {skipped_count}")
    log(f"  Newly attempted         : {successful_count + failed_count}")
    log(f"  Successful              : {successful_count}")
    log(f"  Failed                  : {failed_count}")
    log(f"  Total slices saved      : {len(combined_df)}")
    log(f"  Slice output folder     : {OUTPUT_ROOT}")

    log("")

    if failed_count == 0:
        log("  All subjects processed successfully.")
    elif failed_count < total_subjects * 0.10:
        log(f"  Failure rate {failed_count / total_subjects * 100:.1f}% is acceptable for CNN training.")
    else:
        log(f"  Failure rate {failed_count / total_subjects * 100:.1f}%. Review the failure log before training.")

    log("")
    log("  Next: python MRI/train_mri_cnn_2d.py")
    log("=" * 65)

    save_log(log_lines)


if __name__ == "__main__":
    main()