#!/usr/bin/env python3
"""
README - validate_oasis2_mri.py
===============================

This script performs external MRI validation on the OASIS-2 dataset using the
trained CNN v3 model.

It applies best_resnet18_v3.pt to OASIS-2 MRI data and checks whether the MRI
CNN trained on ADNI can generalise to an external dataset.

Important
---------
This script supports two cases:

1. Preprocessed OASIS-2 slices already exist.
   In this case, the script directly runs inference.

2. Raw OASIS-2 MRI files are available.
   In this case, the script preprocesses the raw NIfTI MRI files into 2D axial
   slices first, then runs inference.

The input format for CNN inference must match CNN v3 training:
    channel 1 = previous slice
    channel 2 = current slice
    channel 3 = next slice

Run
---
python validate_oasis2_mri.py

Paper framing
-------------
External MRI validation on OASIS-2 assesses whether CNN features trained on
ADNI generalise across datasets with different scanners, protocols, and label
definitions.
"""

import glob
import os
import re
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore")
matplotlib.use("Agg")


# Paths
DATA_ROOT = r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch"

OASIS2_CANDIDATES = [
    r"C:\Users\ASUS\Desktop\Research Resources\DATA\OAS2_RAW_PART1",
    r"C:\Users\ASUS\Desktop\Research Resources\DATA\OAS2_RAW_PART2",
]

OASIS2_LABELS = os.path.join(
    DATA_ROOT,
    "data",
    "metadata",
    "oasis2_validation_ready.csv",
)

PREPROC_OUT = os.path.join(
    DATA_ROOT,
    "data",
    "preprocessed",
    "OASIS2_2D_Slices",
)

MODEL_PATH = os.path.join(
    DATA_ROOT,
    "models",
    "best_resnet18_v3.pt",
)

OUT_METRICS = os.path.join(
    DATA_ROOT,
    "results",
    "metrics",
)

OUT_PLOTS = os.path.join(
    DATA_ROOT,
    "plots",
    "oasis2",
)

os.makedirs(OUT_METRICS, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)


# Settings
SLICES_PER_SUBJECT = 20
AXIAL_LO = 0.40
AXIAL_HI = 0.65
IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42
DEVICE = torch.device("cpu")


TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


def section(title):
    print()
    print("-" * 65)
    print(f"  {title}")
    print("-" * 65)


def print_start_message():
    print("=" * 65)
    print("  OASIS-2 MRI EXTERNAL VALIDATION")
    print(f"  Run at : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Model  : CNN v3, ResNet-18")
    print("  Input  : triplet axial slices")
    print("=" * 65)


def load_oasis2_labels():
    """
    Load OASIS-2 subject labels from oasis2_validation_ready.csv.
    """
    section("Loading OASIS-2 labels")

    if not os.path.exists(OASIS2_LABELS):
        raise FileNotFoundError(f"Labels not found: {OASIS2_LABELS}")

    labels_df = pd.read_csv(OASIS2_LABELS)

    print(f"  Shape   : {labels_df.shape}")
    print(f"  Columns : {list(labels_df.columns)}")

    labels_df.columns = labels_df.columns.str.strip()

    subject_candidates = [
        "subject_id",
        "SubjectID",
        "SUBJECT_ID",
        "Subject ID",
        "ID",
        "PTID",
        "id",
    ]

    label_candidates = [
        "label",
        "Label",
        "LABEL",
        "CONV_LABEL",
        "DX",
        "conv",
    ]

    subject_col = next(
        (
            column
            for column in subject_candidates
            if column in labels_df.columns
        ),
        None,
    )

    label_col = next(
        (
            column
            for column in label_candidates
            if column in labels_df.columns
        ),
        None,
    )

    if subject_col is None:
        raise KeyError(
            f"No subject ID column found. Available columns: {list(labels_df.columns)}"
        )

    if label_col is None:
        raise KeyError(
            f"No label column found. Available columns: {list(labels_df.columns)}"
        )

    labels_df = labels_df.rename(
        columns={
            subject_col: "subject_id",
            label_col: "label",
        }
    )

    labels_df["subject_id"] = labels_df["subject_id"].astype(str).str.strip()
    labels_df["label"] = labels_df["label"].astype(int)

    n_pmci = int((labels_df["label"] == 1).sum())
    n_smci = int((labels_df["label"] == 0).sum())

    print(f"  Subjects  : {len(labels_df)}")
    print(f"  pMCI-like : {n_pmci}")
    print(f"  sMCI-like : {n_smci}")

    return labels_df


def preprocessed_slices_exist():
    """
    Check whether OASIS-2 preprocessed PNG slices already exist.
    """
    if not os.path.isdir(PREPROC_OUT):
        return False, 0

    png_files = glob.glob(
        os.path.join(
            PREPROC_OUT,
            "**",
            "*.png",
        ),
        recursive=True,
    )

    return len(png_files) > 0, len(png_files)


def find_oasis2_raw_root():
    """
    Search common folders for raw OASIS-2 MRI files.
    """
    section("Checking for MRI data")

    exists, n_slices = preprocessed_slices_exist()

    if exists:
        print(f"  Preprocessed slices found: {n_slices} PNGs")
        print(f"  Folder: {PREPROC_OUT}")
        return None, True

    print(f"  No preprocessed slices found at: {PREPROC_OUT}")
    print("  Searching for raw OASIS-2 MRI data...")

    for candidate in OASIS2_CANDIDATES:
        if not os.path.isdir(candidate):
            continue

        nii_files = glob.glob(
            os.path.join(
                candidate,
                "**",
                "*.nii*",
            ),
            recursive=True,
        )

        dcm_files = glob.glob(
            os.path.join(
                candidate,
                "**",
                "*.dcm",
            ),
            recursive=True,
        )

        mgz_files = glob.glob(
            os.path.join(
                candidate,
                "**",
                "*.mgz",
            ),
            recursive=True,
        )

        if nii_files or dcm_files or mgz_files:
            if nii_files:
                file_type = "NIfTI"
            elif dcm_files:
                file_type = "DICOM"
            else:
                file_type = "MGZ"

            print(f"  Found raw data: {candidate}")
            print(f"  Detected type : {file_type}")

            return candidate, False

        print(f"  Folder found but no MRI files: {candidate}")

    print()
    print("  OASIS-2 MRI data was not found in the candidate folders.")
    print("  Set OASIS2_CANDIDATES manually at the top of this script.")
    print()
    print("  Expected preprocessed folder structure:")
    print(f"    {PREPROC_OUT}/pMCI-like/<subject>_slice_000.png")
    print(f"    {PREPROC_OUT}/sMCI-like/<subject>_slice_000.png")

    raise SystemExit(0)


def normalise_oasis_id(subject_id):
    """
    Convert OAS2_0001_MR1 to OAS2_0001.
    """
    match = re.match(
        r"(OAS2_\d+)",
        str(subject_id).upper().strip(),
    )

    if match:
        return match.group(1)

    return str(subject_id).strip()


def load_nifti_data(nifti_path):
    """
    Load a NIfTI file and return a 3D numpy array.
    """
    import nibabel as nib

    image = nib.load(nifti_path)

    try:
        image = nib.as_closest_canonical(image)
    except Exception:
        pass

    data = np.asarray(
        image.get_fdata(),
        dtype=np.float32,
    )

    if data.ndim == 4:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(f"Unexpected NIfTI shape: {data.shape}")

    return data


def normalise_mri_slice(slice_array):
    """
    Percentile-normalise one MRI slice to uint8.
    """
    low, high = np.percentile(
        slice_array,
        [
            1,
            99,
        ],
    )

    if high > low:
        slice_array = np.clip(
            (slice_array - low) / (high - low),
            0,
            1,
        )
    else:
        slice_array = np.zeros_like(slice_array)

    return (
        slice_array * 255
    ).astype(np.uint8)


def extract_subject_id_from_path(path_value):
    """
    Extract OAS2_xxxx style subject ID from a file path.
    """
    for part in Path(path_value).parts:
        if re.match(
            r"OAS2_\d+",
            part,
            re.IGNORECASE,
        ):
            return part

    return None


def get_axial_slice_indices(depth):
    """
    Select axial slice indices from the hippocampal window.
    """
    start = int(AXIAL_LO * depth)
    end = int(AXIAL_HI * depth)

    if end <= start:
        end = start + 1

    candidate_indices = list(
        range(
            start,
            end,
        )
    )

    if len(candidate_indices) <= SLICES_PER_SUBJECT:
        return candidate_indices[:SLICES_PER_SUBJECT]

    selected = np.linspace(
        start,
        end - 1,
        SLICES_PER_SUBJECT,
        dtype=int,
    )

    return selected.tolist()


def preprocess_oasis2_mri(oasis2_root, labels_df):
    """
    Preprocess raw OASIS-2 NIfTI files into 2D axial PNG slices.
    """
    section("Preprocessing OASIS-2 MRI into 2D slices")

    print(f"  Source: {oasis2_root}")
    print(f"  Output: {PREPROC_OUT}")

    label_by_subject = {
        normalise_oasis_id(row.subject_id): int(row.label)
        for row in labels_df.itertuples(index=False)
    }

    nifti_files = (
        glob.glob(
            os.path.join(
                oasis2_root,
                "**",
                "*.nii.gz",
            ),
            recursive=True,
        )
        + glob.glob(
            os.path.join(
                oasis2_root,
                "**",
                "*.nii",
            ),
            recursive=True,
        )
    )

    print(f"  NIfTI files found: {len(nifti_files)}")

    processed = 0
    failed = 0

    for nifti_path in sorted(nifti_files):
        subject_raw = extract_subject_id_from_path(nifti_path)

        if subject_raw is None:
            continue

        subject_id = normalise_oasis_id(subject_raw)

        if subject_id not in label_by_subject:
            continue

        label = label_by_subject[subject_id]
        label_name = "pMCI-like" if label == 1 else "sMCI-like"

        output_dir = os.path.join(
            PREPROC_OUT,
            label_name,
        )

        os.makedirs(output_dir, exist_ok=True)

        existing_slices = glob.glob(
            os.path.join(
                output_dir,
                f"{subject_id}_slice_*.png",
            )
        )

        if len(existing_slices) >= SLICES_PER_SUBJECT:
            processed += 1
            continue

        try:
            volume = load_nifti_data(nifti_path)
            depth = volume.shape[2]

            slice_indices = get_axial_slice_indices(depth)

            for slice_number, z_index in enumerate(slice_indices):
                slice_array = volume[:, :, z_index]
                slice_uint8 = normalise_mri_slice(slice_array)

                image = Image.fromarray(slice_uint8).resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                    ),
                    Image.BILINEAR,
                )

                output_path = os.path.join(
                    output_dir,
                    f"{subject_id}_slice_{slice_number:03d}.png",
                )

                image.save(output_path)

            processed += 1

            if processed % 5 == 0:
                print(f"  Preprocessed subjects: {processed}")

        except Exception as error:
            print(f"  FAIL: {subject_id}: {error}")
            failed += 1

    print(f"  Preprocessing complete: {processed} subjects, {failed} failures")


def build_slice_manifest():
    """
    Build subject-wise slice lists from preprocessed OASIS-2 PNG slices.
    """
    section("Building slice manifest")

    subject_slices = {}
    subject_label_map = {}

    for label_name in [
        "pMCI-like",
        "sMCI-like",
    ]:
        label_value = 1 if label_name == "pMCI-like" else 0

        class_dir = os.path.join(
            PREPROC_OUT,
            label_name,
        )

        if not os.path.isdir(class_dir):
            continue

        image_paths = glob.glob(
            os.path.join(
                class_dir,
                "*.png",
            )
        )

        for image_path in image_paths:
            file_name = os.path.basename(image_path)

            match = re.match(
                r"^(.+)_slice_(\d+)\.png$",
                file_name,
            )

            if not match:
                continue

            subject_id = match.group(1)
            slice_index = int(match.group(2))

            if subject_id not in subject_slices:
                subject_slices[subject_id] = []
                subject_label_map[subject_id] = label_value

            subject_slices[subject_id].append(
                (
                    slice_index,
                    image_path,
                )
            )

    for subject_id in subject_slices:
        subject_slices[subject_id].sort(key=lambda item: item[0])

    all_subjects = sorted(subject_slices.keys())

    print(f"  Subjects with slices : {len(all_subjects)}")
    print(f"  pMCI-like            : {sum(1 for subject in all_subjects if subject_label_map[subject] == 1)}")
    print(f"  sMCI-like            : {sum(1 for subject in all_subjects if subject_label_map[subject] == 0)}")

    if len(all_subjects) == 0:
        raise RuntimeError("No preprocessed OASIS-2 slices found.")

    return subject_slices, subject_label_map, all_subjects


def match_labels_to_slices(labels_df, all_subjects):
    """
    Match label-file subject IDs with available slice subject IDs.
    """
    label_ids = set(
        labels_df["subject_id"].astype(str).tolist()
    )

    slice_ids = set(all_subjects)

    matched_ids = label_ids & slice_ids
    unmatched_ids = label_ids - slice_ids

    print(f"  Label file IDs    : {len(label_ids)}")
    print(f"  Matched to slices : {len(matched_ids)}")

    if unmatched_ids:
        print(f"  Unmatched IDs     : {len(unmatched_ids)}")
        print(f"  First few         : {sorted(unmatched_ids)[:5]}")

    matched_df = labels_df[
        labels_df["subject_id"].isin(matched_ids)
    ].copy()

    n_pmci = int((matched_df["label"] == 1).sum())
    n_smci = int((matched_df["label"] == 0).sum())

    print(
        f"  Final evaluation set: {len(matched_df)} "
        f"pMCI-like={n_pmci} sMCI-like={n_smci}"
    )

    return matched_df


def build_triplet_dataframe(labels_matched, subject_slices, subject_label_map):
    """
    Build previous/current/next triplet records for CNN v3 inference.
    """
    records = []

    for subject_id in labels_matched["subject_id"].tolist():
        slices = subject_slices.get(
            subject_id,
            [],
        )

        if not slices:
            continue

        paths = [
            path
            for _, path in slices
        ]

        label = subject_label_map[subject_id]
        n_slices = len(paths)

        for index in range(n_slices):
            records.append(
                {
                    "subject_id": subject_id,
                    "label": label,
                    "path_prev": paths[max(0, index - 1)],
                    "path_curr": paths[index],
                    "path_next": paths[min(n_slices - 1, index + 1)],
                }
            )

    triplet_df = pd.DataFrame(records)

    print(f"  Triplets built: {len(triplet_df)}")

    if len(triplet_df) == 0:
        raise RuntimeError("No triplets were created. Check slice names and labels.")

    return triplet_df


def load_cnn_model():
    """
    Load trained CNN v3 model.
    """
    section("Loading CNN v3 model")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Run train_mri_cnn_2d_v3.py first."
        )

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        512,
        2,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
    )

    model.eval()
    model.to(DEVICE)

    print(f"  Loaded epoch : {checkpoint.get('epoch', '?')}")
    print(f"  Val AUC      : {checkpoint.get('val_auc', 0):.4f}")

    return model


class TripletDataset(Dataset):
    """
    Dataset that loads previous, current, and next slices as RGB channels.
    """

    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        previous_slice = np.asarray(
            Image.open(row["path_prev"]).convert("L")
        )

        current_slice = np.asarray(
            Image.open(row["path_curr"]).convert("L")
        )

        next_slice = np.asarray(
            Image.open(row["path_next"]).convert("L")
        )

        image_array = np.stack(
            [
                previous_slice,
                current_slice,
                next_slice,
            ],
            axis=2,
        ).astype(np.uint8)

        image = Image.fromarray(
            image_array,
            mode="RGB",
        )

        return self.transform(image), row["subject_id"]


def run_inference(model, triplet_df, subject_label_map):
    """
    Run CNN inference on OASIS-2 triplet slices.
    """
    section("Running inference")

    dataset = TripletDataset(
        triplet_df,
        TRANSFORM,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    slice_probs = {}
    slice_labels = {}

    with torch.no_grad():
        for images, subject_ids in loader:
            images = images.to(DEVICE)

            probabilities = torch.softmax(
                model(images),
                dim=1,
            )[:, 1].cpu().numpy()

            for probability, subject_id in zip(probabilities, subject_ids):
                if subject_id not in slice_probs:
                    slice_probs[subject_id] = []
                    slice_labels[subject_id] = subject_label_map[subject_id]

                slice_probs[subject_id].append(
                    float(probability)
                )

    print(f"  Inference complete: {len(slice_probs)} subjects")

    return slice_probs, slice_labels


def bootstrap_auc_ci(y_true, y_prob):
    """
    Bootstrap AUC confidence interval.
    """
    rng = np.random.default_rng(SEED)
    auc_values = []

    for _ in range(1000):
        indices = rng.integers(
            0,
            len(y_true),
            len(y_true),
        )

        if len(np.unique(y_true[indices])) < 2:
            continue

        auc_values.append(
            roc_auc_score(
                y_true[indices],
                y_prob[indices],
            )
        )

    low, high = np.percentile(
        auc_values,
        [
            2.5,
            97.5,
        ],
    )

    return low, high


def evaluate_probabilities(name, y_true, y_prob):
    """
    Evaluate one subject-level aggregation strategy.
    """
    if len(np.unique(y_true)) < 2:
        raise ValueError("Only one class is present. AUC cannot be computed.")

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    auc_pr = average_precision_score(
        y_true,
        y_prob,
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    ci_low, ci_high = bootstrap_auc_ci(
        y_true,
        y_prob,
    )

    print(f"\n  [{name} aggregation]")
    print(f"  AUC-ROC     : {auc:.4f}  [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  AUC-PR      : {auc_pr:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}")

    return {
        "aggregation": name,
        "auc": auc,
        "auc_pr": auc_pr,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "y_prob": y_prob,
    }


def evaluate_subject_level(slice_probs, slice_labels):
    """
    Aggregate slice-level probabilities to subject-level probabilities.
    """
    section("Subject-level evaluation")

    subject_ids = list(slice_probs.keys())

    y_true = np.asarray(
        [
            slice_labels[subject_id]
            for subject_id in subject_ids
        ]
    )

    y_prob_median = np.asarray(
        [
            np.median(slice_probs[subject_id])
            for subject_id in subject_ids
        ]
    )

    y_prob_mean = np.asarray(
        [
            np.mean(slice_probs[subject_id])
            for subject_id in subject_ids
        ]
    )

    print(
        f"  n={len(y_true)} "
        f"pMCI-like={int(y_true.sum())} "
        f"sMCI-like={int((y_true == 0).sum())}"
    )

    median_result = evaluate_probabilities(
        "median",
        y_true,
        y_prob_median,
    )

    mean_result = evaluate_probabilities(
        "mean",
        y_true,
        y_prob_mean,
    )

    return subject_ids, y_true, y_prob_median, y_prob_mean, median_result, mean_result


def save_prediction_table(subject_ids, y_true, y_prob_median, y_prob_mean):
    """
    Save subject-level OASIS-2 MRI predictions.
    """
    output_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "y_true": y_true,
            "y_prob_median": y_prob_median,
            "y_prob_mean": y_prob_mean,
            "y_pred": (y_prob_median >= 0.5).astype(int),
        }
    )

    output_path = os.path.join(
        OUT_METRICS,
        "oasis2_mri_validation.csv",
    )

    output_df.to_csv(
        output_path,
        index=False,
    )

    print(f"\n  Saved: {output_path}")


def save_validation_plot(y_true, best_result):
    """
    Save ROC curve and internal-vs-external comparison plot.
    """
    section("Saving validation plots")

    final_auc = best_result["auc"]
    final_low = best_result["ci_low"]
    final_high = best_result["ci_high"]
    best_prob = best_result["y_prob"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        best_prob,
    )

    axes[0].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"OASIS-2 CNN AUC={final_auc:.3f} [{final_low:.3f}, {final_high:.3f}]",
    )

    axes[0].plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        "k--",
        alpha=0.3,
    )

    axes[0].set_xlabel("1 - Specificity")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_title("OASIS-2 External Validation - CNN v3 ROC")
    axes[0].legend(
        loc="lower right",
        fontsize=9,
    )
    axes[0].grid(alpha=0.3)

    labels = [
        "ADNI\ninternal\nClinical",
        "ADNI\ninternal\nCNN v3",
        "OASIS-2\nexternal\nClinical",
        "OASIS-2\nexternal\nCNN v3",
    ]

    aucs = [
        0.821,
        0.710,
        0.469,
        final_auc,
    ]

    ci_low = [
        0.751,
        0.606,
        0.312,
        final_low,
    ]

    ci_high = [
        0.885,
        0.816,
        0.619,
        final_high,
    ]

    colors = [
        "steelblue",
        "darkorange",
        "steelblue",
        "darkorange",
    ]

    lower_error = [
        auc - low
        for auc, low in zip(aucs, ci_low)
    ]

    upper_error = [
        high - auc
        for auc, high in zip(aucs, ci_high)
    ]

    bars = axes[1].bar(
        labels,
        aucs,
        color=colors,
        alpha=0.85,
        yerr=[
            lower_error,
            upper_error,
        ],
        capsize=5,
        hatch=[
            "",
            "",
            "///",
            "///",
        ],
        error_kw={
            "elinewidth": 1.5,
        },
    )

    axes[1].axhline(
        0.5,
        linestyle=":",
        color="gray",
        alpha=0.5,
    )

    axes[1].set_ylim(
        [
            0.3,
            1.0,
        ]
    )

    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Internal ADNI vs External OASIS-2 Validation")
    axes[1].grid(
        axis="y",
        alpha=0.3,
    )

    for bar, auc in zip(bars, aucs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            auc + 0.018,
            f"{auc:.3f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        "OASIS-2 External MRI Validation - CNN v3",
        fontsize=12,
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUT_PLOTS,
        "oasis2_mri_validation.png",
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {output_path}")


def print_final_summary(y_true, best_result):
    """
    Print final validation interpretation.
    """
    final_auc = best_result["auc"]
    final_low = best_result["ci_low"]
    final_high = best_result["ci_high"]

    print()
    print("=" * 65)
    print("  OASIS-2 MRI EXTERNAL VALIDATION COMPLETE")
    print("=" * 65)

    print(f"  n subjects : {len(y_true)}")
    print(f"  pMCI-like  : {int(y_true.sum())}")
    print(f"  sMCI-like  : {int((y_true == 0).sum())}")

    print("\n  Result:")
    print(f"  OASIS-2 CNN AUC : {final_auc:.4f} [{final_low:.3f}, {final_high:.3f}]")
    print("  ADNI CNN AUC    : 0.710 [0.606, 0.816] internal")
    print("  OASIS-2 Clin.   : 0.469 [0.312, 0.619] previous clinical validation")

    print("\n  Interpretation:")

    if final_auc >= 0.65:
        print("  CNN generalises to OASIS-2 better than the clinical cross-dataset model.")
        print("  MRI CNN features appear more transferable than clinical features.")

    elif final_auc >= 0.55:
        print("  CNN shows partial generalisation to OASIS-2.")
        print("  It performs better than the clinical OASIS-2 result but below ADNI internal testing.")
        print("  Domain shift from scanner, protocol, and slice extraction may limit transfer.")

    else:
        print("  CNN does not clearly generalise to OASIS-2.")
        print("  Possible causes include scanner differences, acquisition protocol differences,")
        print("  age distribution shift, and OASIS-2 labels not being identical to ADNI labels.")

    print("\n  Paper framing:")
    print("  External MRI validation on OASIS-2 evaluates cross-dataset transfer.")
    print("  It should be interpreted with scanner, protocol, and label-definition shift in mind.")

    print("=" * 65)


def main():
    print_start_message()

    labels_df = load_oasis2_labels()

    oasis2_root, preprocessed_available = find_oasis2_raw_root()

    if not preprocessed_available and oasis2_root is not None:
        preprocess_oasis2_mri(
            oasis2_root,
            labels_df,
        )

    subject_slices, subject_label_map, all_subjects = build_slice_manifest()

    labels_matched = match_labels_to_slices(
        labels_df,
        all_subjects,
    )

    triplet_df = build_triplet_dataframe(
        labels_matched,
        subject_slices,
        subject_label_map,
    )

    model = load_cnn_model()

    slice_probs, slice_labels = run_inference(
        model,
        triplet_df,
        subject_label_map,
    )

    subject_ids, y_true, y_prob_median, y_prob_mean, median_result, mean_result = (
        evaluate_subject_level(
            slice_probs,
            slice_labels,
        )
    )

    best_result = median_result

    save_prediction_table(
        subject_ids,
        y_true,
        y_prob_median,
        y_prob_mean,
    )

    save_validation_plot(
        y_true,
        best_result,
    )

    print_final_summary(
        y_true,
        best_result,
    )


if __name__ == "__main__":
    main()