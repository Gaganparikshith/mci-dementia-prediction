#!/usr/bin/env python3
"""
README - gradcam_mri_2d.py
==========================

This script generates Grad-CAM visual explanations for the Phase 2 CNN v3
model.

It loads the trained ResNet-18 v3 checkpoint, uses the same triplet axial input
format used during training, and creates Grad-CAM heatmaps from the last
convolutional block of the model.

Main purpose
------------
The goal of this script is to show which MRI regions influenced the CNN
predictions for pMCI and sMCI subjects.

Important
---------
The input format must match CNN v3 training.

CNN v3 used triplet axial input:
    channel 1 = previous slice
    channel 2 = current slice
    channel 3 = next slice

Grad-CAM is generated using:
    target layer = model.layer4

The heatmap is overlaid on the center slice.

Processing logic
----------------
1. Load the trained ResNet-18 v3 checkpoint.
2. Load the same test subjects used in CNN v3.
3. Load subject-level CNN predictions.
4. Build triplet inputs from preprocessed MRI slices.
5. Compute Grad-CAM for each test subject.
6. Generate grids for correct and incorrect predictions.
7. Generate average class-wise heatmaps.
8. Generate a 4-subject panel for paper/report use.

Run
---
python gradcam_mri_2d.py

Next step
---------
python validate_oasis2_mri.py
"""

import glob
import json
import os
import re
import time

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


matplotlib.use("Agg")


# Paths
DATA_ROOT = r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch"

SLICE_ROOT = os.path.join(
    DATA_ROOT,
    "data",
    "preprocessed",
    "ADNI_2D_Slices",
)

MODEL_PATH = os.path.join(
    DATA_ROOT,
    "models",
    "best_resnet18_v3.pt",
)

SPLIT_JSON = os.path.join(
    DATA_ROOT,
    "results",
    "metrics",
    "cnn2d_v2_subject_split.json",
)

CNN_PREDICTIONS = os.path.join(
    DATA_ROOT,
    "results",
    "metrics",
    "cnn2d_v3_subject_predictions.csv",
)

OUT_DIR = os.path.join(
    DATA_ROOT,
    "plots",
    "gradcam",
)

os.makedirs(OUT_DIR, exist_ok=True)


# Settings
DEVICE = torch.device("cpu")
CENTER_SLICE_INDEX = 9


# Same normalization used during CNN v3 training
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
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
    print("  GRAD-CAM - ResNet-18 v3")
    print(f"  Run at : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Layer  : model.layer4")
    print("  Input  : triplet axial slices")
    print("=" * 65)


def load_model():
    """
    Load the trained CNN v3 checkpoint.
    """
    section("Loading model")

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

    epoch = checkpoint.get("epoch", "?")
    val_auc = checkpoint.get("val_auc", 0)

    print(f"  Checkpoint epoch : {epoch}")
    print(f"  Validation AUC   : {val_auc:.4f}")

    return model


def load_test_subjects():
    """
    Load the test subjects from the saved CNN v2/v3 subject split.
    """
    section("Loading test subjects")

    if not os.path.isfile(SPLIT_JSON):
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_JSON}\n"
            "Run train_mri_cnn_2d_v2.py first."
        )

    with open(SPLIT_JSON, "r", encoding="utf-8") as file:
        split = json.load(file)

    test_subjects = set(
        str(subject).strip()
        for subject in split["test_subjects"]
    )

    return test_subjects


def scan_test_subject_slices(test_subjects):
    """
    Scan pMCI and sMCI slice folders and keep only test subjects.
    """
    subject_slices = {}
    subject_labels = {}

    class_mapping = [
        ("pMCI", 1),
        ("sMCI", 0),
    ]

    for class_name, label_value in class_mapping:
        class_dir = os.path.join(
            SLICE_ROOT,
            class_name,
        )

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing slice folder: {class_dir}")

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

            if subject_id not in test_subjects:
                continue

            slice_index = int(match.group(2))

            if subject_id not in subject_slices:
                subject_slices[subject_id] = []
                subject_labels[subject_id] = label_value

            subject_slices[subject_id].append(
                (
                    slice_index,
                    image_path,
                )
            )

    for subject_id in subject_slices:
        subject_slices[subject_id].sort(key=lambda item: item[0])

    print(f"  Test subjects loaded : {len(subject_slices)}")
    print(f"  pMCI                 : {sum(1 for value in subject_labels.values() if value == 1)}")
    print(f"  sMCI                 : {sum(1 for value in subject_labels.values() if value == 0)}")

    return subject_slices, subject_labels


def load_predictions():
    """
    Load subject-level CNN predictions.
    """
    if not os.path.isfile(CNN_PREDICTIONS):
        raise FileNotFoundError(
            f"Prediction file not found: {CNN_PREDICTIONS}\n"
            "Run train_mri_cnn_2d_v3.py first."
        )

    prediction_df = pd.read_csv(CNN_PREDICTIONS)

    prediction_map = {}

    for row in prediction_df.itertuples(index=False):
        prediction_map[row.subject_id] = {
            "y_prob": row.y_prob,
            "y_pred": row.y_pred,
            "y_true": row.y_true,
        }

    return prediction_map


class GradCAM:
    """
    Grad-CAM implementation for ResNet-18.

    Target layer:
        model.layer4

    The input tensor is set to require gradients so that Grad-CAM works even
    when most of the model parameters are frozen.
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_handle = self.model.layer4.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = self.model.layer4.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute(self, image_tensor, class_index=1):
        """
        Compute Grad-CAM for one input image.

        Args:
            image_tensor:
                Tensor with shape 3 x 224 x 224.

            class_index:
                1 for pMCI.
                0 for sMCI.

        Returns:
            224 x 224 Grad-CAM heatmap normalized to [0, 1].
        """
        self.model.eval()

        x = image_tensor.unsqueeze(0).to(DEVICE)
        x.requires_grad_(True)

        output = self.model(x)

        self.model.zero_grad()

        score = output[0, class_index]
        score.backward()

        weights = self.gradients.mean(
            dim=[
                2,
                3,
            ],
            keepdim=True,
        )

        cam = (
            weights * self.activations
        ).sum(dim=1).squeeze()

        cam = torch.relu(cam).detach().cpu().numpy()

        cam = cv2.resize(
            cam,
            (
                224,
                224,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


def load_triplet_tensor(slices, center_index):
    """
    Load previous, current, and next slices as a 3-channel image tensor.
    """
    paths = [
        path
        for _, path in slices
    ]

    n_slices = len(paths)
    index = center_index

    previous_slice = np.asarray(
        Image.open(paths[max(0, index - 1)]).convert("L")
    )

    current_slice = np.asarray(
        Image.open(paths[index]).convert("L")
    )

    next_slice = np.asarray(
        Image.open(paths[min(n_slices - 1, index + 1)]).convert("L")
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

    return TRANSFORM(image), current_slice


def overlay_heatmap(gray_image, cam, alpha=0.45):
    """
    Overlay the Grad-CAM heatmap on a grayscale MRI slice.
    """
    base = Image.fromarray(gray_image).resize(
        (
            224,
            224,
        )
    )

    base = np.asarray(base)

    base_rgb = np.stack(
        [
            base,
            base,
            base,
        ],
        axis=2,
    )

    heatmap = (
        cm.jet(cam)[:, :, :3] * 255
    ).astype(np.uint8)

    overlay = (
        alpha * heatmap + (1 - alpha) * base_rgb
    ).astype(np.uint8)

    return overlay


def compute_gradcam_for_subjects(subject_slices, subject_labels, prediction_map, gradcam):
    """
    Compute Grad-CAM overlays for all test subjects.
    """
    section("Computing Grad-CAM maps")

    subject_cam_data = {}

    for subject_id in sorted(subject_slices.keys()):
        slices = subject_slices[subject_id]
        label = subject_labels[subject_id]

        center_index = min(
            CENTER_SLICE_INDEX,
            len(slices) - 1,
        )

        prediction = prediction_map.get(
            subject_id,
            {},
        )

        probability = prediction.get(
            "y_prob",
            0.5,
        )

        y_pred = int(probability >= 0.5)
        correct = y_pred == label

        try:
            image_tensor, gray_image = load_triplet_tensor(
                slices,
                center_index,
            )

            cam = gradcam.compute(
                image_tensor,
                class_index=label,
            )

            overlay = overlay_heatmap(
                gray_image,
                cam,
            )

            subject_cam_data[subject_id] = {
                "cam": cam,
                "overlay": overlay,
                "gray": gray_image,
                "label": label,
                "prob": probability,
                "correct": correct,
            }

        except Exception as error:
            print(f"  WARNING: {subject_id}: {error}")

    print(f"  Completed    : {len(subject_cam_data)} subjects")
    print(f"  pMCI correct : {sum(1 for item in subject_cam_data.values() if item['label'] == 1 and item['correct'])}")
    print(f"  pMCI wrong   : {sum(1 for item in subject_cam_data.values() if item['label'] == 1 and not item['correct'])}")
    print(f"  sMCI correct : {sum(1 for item in subject_cam_data.values() if item['label'] == 0 and item['correct'])}")
    print(f"  sMCI wrong   : {sum(1 for item in subject_cam_data.values() if item['label'] == 0 and not item['correct'])}")

    return subject_cam_data


def make_grid(title, subjects_data, filename, n_cols=4):
    """
    Save a grid of Grad-CAM overlays.
    """
    n_subjects = len(subjects_data)

    if n_subjects == 0:
        print(f"  SKIP: {title} - no subjects")
        return

    n_cols = min(
        n_cols,
        n_subjects,
    )

    n_rows = (
        n_subjects + n_cols - 1
    ) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            n_cols * 3.2,
            n_rows * 3.5,
        ),
    )

    axes = np.asarray(axes).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for index, (subject_id, data) in enumerate(subjects_data):
        axis = axes[index]

        label_name = "pMCI" if data["label"] == 1 else "sMCI"
        marker = "correct" if data["correct"] else "wrong"

        axis.imshow(data["overlay"])

        axis.set_title(
            f"{label_name} p={data['prob']:.2f}\n{marker}",
            fontsize=8,
            pad=3,
        )

        axis.axis("off")

    fig.suptitle(
        title,
        fontsize=12,
        y=1.01,
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUT_DIR,
        filename,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {output_path}")


def split_subjects_by_class_and_correctness(subject_cam_data):
    """
    Split Grad-CAM results into four groups.
    """
    pmci_correct = [
        (subject_id, data)
        for subject_id, data in subject_cam_data.items()
        if data["label"] == 1 and data["correct"]
    ]

    pmci_wrong = [
        (subject_id, data)
        for subject_id, data in subject_cam_data.items()
        if data["label"] == 1 and not data["correct"]
    ]

    smci_correct = [
        (subject_id, data)
        for subject_id, data in subject_cam_data.items()
        if data["label"] == 0 and data["correct"]
    ]

    smci_wrong = [
        (subject_id, data)
        for subject_id, data in subject_cam_data.items()
        if data["label"] == 0 and not data["correct"]
    ]

    return pmci_correct, pmci_wrong, smci_correct, smci_wrong


def save_gradcam_grids(subject_cam_data):
    """
    Save Grad-CAM grids for correct and incorrect pMCI/sMCI subjects.
    """
    section("Generating Grad-CAM grids")

    pmci_correct, pmci_wrong, smci_correct, smci_wrong = (
        split_subjects_by_class_and_correctness(subject_cam_data)
    )

    make_grid(
        "Grad-CAM: correctly classified pMCI subjects",
        pmci_correct[:8],
        "gradcam_pMCI_correct.png",
    )

    make_grid(
        "Grad-CAM: correctly classified sMCI subjects",
        smci_correct[:8],
        "gradcam_sMCI_correct.png",
    )

    make_grid(
        "Grad-CAM: misclassified pMCI subjects",
        pmci_wrong[:8],
        "gradcam_pMCI_wrong.png",
    )

    make_grid(
        "Grad-CAM: misclassified sMCI subjects",
        smci_wrong[:8],
        "gradcam_sMCI_wrong.png",
    )

    return pmci_correct, pmci_wrong, smci_correct, smci_wrong


def save_average_heatmaps(subject_cam_data):
    """
    Save mean Grad-CAM heatmap for pMCI and sMCI.
    """
    section("Generating average class heatmaps")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    class_settings = [
        (1, "pMCI converter", "Reds"),
        (0, "sMCI stable", "Blues"),
    ]

    for axis, (label_value, label_name, color_map) in zip(axes, class_settings):
        cams = [
            data["cam"]
            for data in subject_cam_data.values()
            if data["label"] == label_value
        ]

        if not cams:
            axis.axis("off")
            axis.set_title(f"No {label_name} subjects")
            continue

        cams = np.stack(cams)
        mean_cam = cams.mean(axis=0)

        image = axis.imshow(
            mean_cam,
            cmap=color_map,
            vmin=0,
            vmax=1,
        )

        axis.set_title(
            f"Mean Grad-CAM - {label_name}\n(n={len(cams)})",
            fontsize=10,
        )

        axis.axis("off")

        plt.colorbar(
            image,
            ax=axis,
            fraction=0.046,
        )

    fig.suptitle(
        "Average Grad-CAM Activation Map by Class",
        fontsize=12,
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUT_DIR,
        "gradcam_average_heatmap.png",
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {output_path}")


def save_paper_panel(pmci_correct, smci_correct):
    """
    Save a 4-subject panel for report or paper use.
    """
    section("Generating paper figure panel")

    panel_subjects = pmci_correct[:2] + smci_correct[:2]

    if len(panel_subjects) != 4:
        print(
            "  SKIP: Need at least 2 correct pMCI and 2 correct sMCI examples "
            f"(pMCI={len(pmci_correct)}, sMCI={len(smci_correct)})"
        )
        return

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7),
    )

    labels = [
        "pMCI",
        "pMCI",
        "sMCI",
        "sMCI",
    ]

    for column, (subject_id, data) in enumerate(panel_subjects):
        label_name = labels[column]

        axes[0, column].imshow(
            data["gray"],
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        axes[0, column].set_title(
            f"{label_name}\np={data['prob']:.2f}",
            fontsize=9,
        )

        axes[0, column].axis("off")

        axes[1, column].imshow(
            data["overlay"],
        )

        axes[1, column].set_title(
            "Grad-CAM",
            fontsize=9,
        )

        axes[1, column].axis("off")

    axes[0, 0].set_ylabel(
        "Original",
        fontsize=10,
    )

    axes[1, 0].set_ylabel(
        "Grad-CAM",
        fontsize=10,
    )

    fig.suptitle(
        "Grad-CAM Visualisation: pMCI vs sMCI Correct Predictions",
        fontsize=12,
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUT_DIR,
        "gradcam_sample_panel.png",
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {output_path}")


def save_colorbar():
    """
    Save a standalone Grad-CAM colorbar reference.
    """
    fig, axis = plt.subplots(
        figsize=(4, 1),
    )

    fig.subplots_adjust(bottom=0.5)

    colorbar = matplotlib.colorbar.ColorbarBase(
        axis,
        cmap=cm.jet,
        orientation="horizontal",
        norm=matplotlib.colors.Normalize(
            vmin=0,
            vmax=1,
        ),
    )

    colorbar.set_label(
        "Grad-CAM activation: 0 = low, 1 = high",
        fontsize=9,
    )

    output_path = os.path.join(
        OUT_DIR,
        "gradcam_colorbar.png",
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {output_path}")


def print_final_summary():
    print()
    print("=" * 65)
    print("  GRAD-CAM COMPLETE")
    print("=" * 65)

    print(f"  Output folder: {OUT_DIR}")

    print("\n  Files generated:")

    for file_name in sorted(os.listdir(OUT_DIR)):
        print(f"    {file_name}")

    print("\n  Interpretation guide:")
    print("    Red/yellow regions show higher Grad-CAM activation.")
    print("    These are the image regions the CNN used more strongly.")
    print("    For pMCI, hippocampal or temporal activation is expected.")
    print("    For sMCI, activation may appear more diffuse.")

    print("\n  Next:")
    print("    validate_oasis2_mri.py")
    print("=" * 65)


def main():
    print_start_message()

    model = load_model()

    test_subjects = load_test_subjects()

    subject_slices, subject_labels = scan_test_subject_slices(
        test_subjects,
    )

    prediction_map = load_predictions()

    gradcam = GradCAM(
        model,
    )

    subject_cam_data = compute_gradcam_for_subjects(
        subject_slices,
        subject_labels,
        prediction_map,
        gradcam,
    )

    pmci_correct, pmci_wrong, smci_correct, smci_wrong = save_gradcam_grids(
        subject_cam_data,
    )

    save_average_heatmaps(
        subject_cam_data,
    )

    save_paper_panel(
        pmci_correct,
        smci_correct,
    )

    save_colorbar()

    print_final_summary()


if __name__ == "__main__":
    main()