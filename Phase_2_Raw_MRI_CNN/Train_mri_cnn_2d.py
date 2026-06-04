#!/usr/bin/env python3
"""
README - train_mri_cnn_2d_v3.py
================================

This script trains the Phase 2 MRI CNN model using preprocessed ADNI 2D MRI
slices.

Version 3 is a controlled experiment. It uses the same subject split as v2,
but freezes the full ResNet-18 backbone and trains only the final fully
connected layer. This reduces overfitting because only 1,026 parameters are
trainable.

Main idea
---------
Each MRI input is created as a 3-channel axial triplet:

channel 1 = previous slice
channel 2 = current slice
channel 3 = next slice

This gives the model limited inter-slice spatial context without needing a
new preprocessing pipeline or a full 3D CNN.

Changes from v2
---------------
1. FC-only training:
   All ResNet-18 layers are frozen except the final classifier.

2. Triplet input:
   Consecutive axial slices are loaded as a 3-channel image.

3. Brain-safe augmentation:
   Small rotation, brightness/contrast jitter, and mild Gaussian noise.
   Horizontal flip is not used because it can swap left/right anatomy.

4. Subject-level aggregation:
   Mean, median, and top-5 mean probabilities are compared on the test set.

Run
---
python train_mri_cnn_2d_v3.py

Next step
---------
python extract_mri_embeddings.py
"""

import glob
import json
import os
import re
import time

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


matplotlib.use("Agg")


# Paths
DATA_ROOT = r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch"

SLICE_ROOT = os.path.join(
    DATA_ROOT,
    "data",
    "preprocessed",
    "ADNI_2D_Slices",
)

SPLIT_JSON = os.path.join(
    DATA_ROOT,
    "results",
    "metrics",
    "cnn2d_v2_subject_split.json",
)

OUT_MODEL = os.path.join(
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
    "cnn2d_v3",
)

os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
os.makedirs(OUT_METRICS, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)


# Hyperparameters
EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

DEVICE = torch.device("cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


def section(title):
    print()
    print("-" * 65)
    print(f"  {title}")
    print("-" * 65)


def print_start_message():
    print("=" * 65)
    print("  TRAIN MRI CNN v3 - ResNet-18")
    print(f"  Run at  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  LR      : {LR}")
    print("  Frozen  : all layers except final FC layer")
    print("  Input   : 3-channel axial triplet")
    print("=" * 65)


def build_subject_manifest():
    """
    Build subject-wise slice lists from the preprocessed PNG folders.

    Expected folder structure:
        ADNI_2D_Slices/pMCI/*.png
        ADNI_2D_Slices/sMCI/*.png
    """
    section("Building subject manifest")

    subject_slices = {}
    subject_labels = {}

    class_mapping = [
        ("pMCI", 1),
        ("sMCI", 0),
    ]

    for label_name, label_value in class_mapping:
        class_dir = os.path.join(SLICE_ROOT, label_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        png_files = glob.glob(
            os.path.join(
                class_dir,
                "*.png",
            )
        )

        for image_path in png_files:
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
                subject_labels[subject_id] = label_value

            subject_slices[subject_id].append(
                (
                    slice_index,
                    image_path,
                )
            )

    for subject_id in subject_slices:
        subject_slices[subject_id].sort(key=lambda item: item[0])

    all_subjects = sorted(subject_slices.keys())

    n_pmci = sum(
        1
        for subject in all_subjects
        if subject_labels[subject] == 1
    )

    n_smci = sum(
        1
        for subject in all_subjects
        if subject_labels[subject] == 0
    )

    print(f"  Total subjects : {len(all_subjects)}")
    print(f"  pMCI subjects  : {n_pmci}")
    print(f"  sMCI subjects  : {n_smci}")

    return subject_slices, subject_labels, all_subjects


def build_triplet_dataframe(subject_slices, subject_labels, all_subjects):
    """
    Create one record per slice using previous/current/next slice paths.
    """
    records = []

    for subject_id in all_subjects:
        slices = subject_slices[subject_id]
        paths = [
            path
            for _, path in slices
        ]

        label = subject_labels[subject_id]
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

    print(f"  Total triplets : {len(triplet_df)}")
    print("  This should match the total number of preprocessed slices.")

    return triplet_df


def load_subject_split(triplet_df):
    """
    Load the same subject split used in CNN v2 for fair comparison.
    """
    section("Loading Phase 2 subject split")

    if not os.path.exists(SPLIT_JSON):
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_JSON}\n"
            "Run train_mri_cnn_2d_v2.py first because it creates this split file."
        )

    with open(SPLIT_JSON, "r", encoding="utf-8") as file:
        split = json.load(file)

    train_subjects = set(
        str(subject).strip()
        for subject in split["train_subjects"]
    )

    val_subjects = set(
        str(subject).strip()
        for subject in split["val_subjects"]
    )

    test_subjects = set(
        str(subject).strip()
        for subject in split["test_subjects"]
    )

    train_df = triplet_df[
        triplet_df["subject_id"].isin(train_subjects)
    ].reset_index(drop=True)

    val_df = triplet_df[
        triplet_df["subject_id"].isin(val_subjects)
    ].reset_index(drop=True)

    test_df = triplet_df[
        triplet_df["subject_id"].isin(test_subjects)
    ].reset_index(drop=True)

    print_split_summary("Train", train_df)
    print_split_summary("Val", val_df)
    print_split_summary("Test", test_df)

    return train_df, val_df, test_df


def print_split_summary(name, split_df):
    subject_count = split_df["subject_id"].nunique()
    triplet_count = len(split_df)

    pmci_subjects = split_df[split_df["label"] == 1]["subject_id"].nunique()
    smci_subjects = split_df[split_df["label"] == 0]["subject_id"].nunique()

    print(
        f"  {name:<5}: {subject_count:>3} subjects "
        f"({triplet_count:>5} triplets) "
        f"pMCI={pmci_subjects} sMCI={smci_subjects}"
    )


class GaussianNoise:
    """Add mild Gaussian noise after ToTensor."""

    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),
            transforms.ToTensor(),
            GaussianNoise(std=0.02),
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

    eval_transform = transforms.Compose(
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

    return train_transform, eval_transform


class TripletDataset(Dataset):
    """
    Dataset that loads previous, current, and next axial slices as RGB channels.
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

        label = int(row["label"])

        return self.transform(image), label


def create_dataloaders(train_df, val_df, test_df):
    train_transform, eval_transform = get_transforms()

    train_labels = train_df["label"].values
    class_counts = np.bincount(train_labels)

    sample_weights = (
        1.0 / class_counts
    )[train_labels]

    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_dataset = TripletDataset(
        train_df,
        train_transform,
    )

    val_dataset = TripletDataset(
        val_df,
        eval_transform,
    )

    test_dataset = TripletDataset(
        test_df,
        eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader, class_counts


def build_model():
    section("Model")

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(
        512,
        2,
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    frozen_params = total_params - trainable_params

    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")
    print(f"  Frozen params    : {frozen_params:,}")
    print(f"  Trainable percent: {100 * trainable_params / total_params:.3f}%")
    print("  Mode             : linear probe on pretrained features")

    model.to(DEVICE)

    return model


def create_training_objects(model, class_counts):
    class_weights = torch.tensor(
        [
            1.0 / class_counts[0],
            1.0 / class_counts[1],
        ],
        dtype=torch.float32,
    )

    class_weights = class_weights / class_weights.sum()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE),
    )

    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    return criterion, optimizer, scheduler


def run_epoch(model, loader, criterion, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    probabilities = []
    labels_all = []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(
                logits,
                labels,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            total_loss += loss.item() * len(labels)

            probabilities.extend(
                batch_probabilities.detach().cpu().tolist()
            )

            labels_all.extend(
                labels.detach().cpu().tolist()
            )

    average_loss = total_loss / len(labels_all)

    if len(set(labels_all)) > 1:
        auc = roc_auc_score(
            labels_all,
            probabilities,
        )
    else:
        auc = 0.5

    return average_loss, auc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    section("Training")

    print(
        f"  {'Ep':>3}  {'TrLoss':>8}  {'VaLoss':>8}  "
        f"{'TrAUC':>7}  {'VaAUC':>7}  {'Time':>6}"
    )

    print("  " + "-" * 55)

    best_val_auc = 0.0
    best_epoch = 0
    patience_count = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_auc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            train=True,
        )

        val_loss, val_auc = run_epoch(
            model,
            val_loader,
            criterion,
            train=False,
        )

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"  {epoch:>3}  "
            f"{train_loss:>8.4f}  "
            f"{val_loss:>8.4f}  "
            f"{train_auc:>7.4f}  "
            f"{val_auc:>7.4f}  "
            f"{elapsed:>5.0f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "tr_loss": train_loss,
                "va_loss": val_loss,
                "tr_auc": train_auc,
                "va_auc": val_auc,
            }
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "config": {
                        "frozen": "all_except_fc",
                        "input": "triplet_axial",
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                    },
                },
                OUT_MODEL,
            )

            print(f"  Best val AUC updated: {best_val_auc:.4f} - saved")

        else:
            patience_count += 1

            if patience_count >= PATIENCE:
                print(f"\n  Early stopping triggered. Patience={PATIENCE}")
                break

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        os.path.join(
            OUT_METRICS,
            "cnn2d_v3_training_history.csv",
        ),
        index=False,
    )

    return best_epoch, best_val_auc, history


def collect_subject_probabilities(model, test_loader, test_df):
    model.eval()

    slice_probs = {}
    slice_labels = {}

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)

            probabilities = torch.softmax(
                model(images),
                dim=1,
            )[:, 1].cpu().numpy()

            start = batch_index * BATCH_SIZE
            end = start + len(probabilities)

            rows = test_df.iloc[start:end]

            for probability, (_, row) in zip(probabilities, rows.iterrows()):
                subject_id = row["subject_id"]

                if subject_id not in slice_probs:
                    slice_probs[subject_id] = []
                    slice_labels[subject_id] = int(row["label"])

                slice_probs[subject_id].append(float(probability))

    return slice_probs, slice_labels


def bootstrap_auc_ci(y_true, y_prob, n_boot=1000):
    rng = np.random.default_rng(SEED)

    boot_aucs = []

    for _ in range(n_boot):
        sample_indices = rng.integers(
            0,
            len(y_true),
            len(y_true),
        )

        if len(np.unique(y_true[sample_indices])) < 2:
            continue

        boot_auc = roc_auc_score(
            y_true[sample_indices],
            y_prob[sample_indices],
        )

        boot_aucs.append(boot_auc)

    ci_low, ci_high = np.percentile(
        boot_aucs,
        [
            2.5,
            97.5,
        ],
    )

    return ci_low, ci_high


def evaluate_aggregation(name, y_true, y_prob):
    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    auc_pr = average_precision_score(
        y_true,
        y_prob,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    ci_low, ci_high = bootstrap_auc_ci(
        y_true,
        y_prob,
        n_boot=1000,
    )

    print(f"\n  [{name}]")
    print(f"  AUC-ROC     : {auc:.4f}  [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  AUC-PR      : {auc_pr:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}")

    return {
        "name": name,
        "auc": auc,
        "ci_lo": ci_low,
        "ci_hi": ci_high,
        "auc_pr": auc_pr,
        "f1": f1,
        "sens": sensitivity,
        "spec": specificity,
        "y_prob": y_prob,
    }


def evaluate_test_set(model, test_loader, test_df, best_epoch, best_val_auc):
    section("Test evaluation")

    print(f"  Best epoch : {best_epoch}")
    print(f"  Val AUC    : {best_val_auc:.4f}")

    checkpoint = torch.load(
        OUT_MODEL,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
    )

    slice_probs, slice_labels = collect_subject_probabilities(
        model,
        test_loader,
        test_df,
    )

    subject_ids = list(slice_probs.keys())

    y_true = np.asarray(
        [
            slice_labels[subject_id]
            for subject_id in subject_ids
        ]
    )

    prob_mean = np.asarray(
        [
            np.mean(slice_probs[subject_id])
            for subject_id in subject_ids
        ]
    )

    prob_median = np.asarray(
        [
            np.median(slice_probs[subject_id])
            for subject_id in subject_ids
        ]
    )

    prob_top5 = np.asarray(
        [
            np.mean(
                sorted(
                    slice_probs[subject_id],
                    reverse=True,
                )[:5]
            )
            for subject_id in subject_ids
        ]
    )

    print(
        f"\n  Subjects in test set: {len(subject_ids)} "
        f"(pMCI={int(y_true.sum())}, sMCI={int((y_true == 0).sum())})"
    )

    results = [
        evaluate_aggregation(
            "mean",
            y_true,
            prob_mean,
        ),
        evaluate_aggregation(
            "median",
            y_true,
            prob_median,
        ),
        evaluate_aggregation(
            "top-5 mean",
            y_true,
            prob_top5,
        ),
    ]

    best_aggregation = max(
        results,
        key=lambda result: result["auc"],
    )

    print(
        f"\n  Best aggregation: {best_aggregation['name']} "
        f"AUC={best_aggregation['auc']:.4f}"
    )

    return subject_ids, y_true, results, best_aggregation


def save_results(subject_ids, y_true, results, best_aggregation, best_epoch, best_val_auc):
    result_rows = []

    for result in results:
        result_rows.append(
            {
                "phase": "Phase2_CNN_v3",
                "aggregation": result["name"],
                "AUC_ROC": result["auc"],
                "AUC_ROC_CI_lo": result["ci_lo"],
                "AUC_ROC_CI_hi": result["ci_hi"],
                "AUC_PR": result["auc_pr"],
                "F1": result["f1"],
                "Sensitivity": result["sens"],
                "Specificity": result["spec"],
                "best_epoch": best_epoch,
                "best_val_auc": best_val_auc,
                "version": "v3",
            }
        )

    results_df = pd.DataFrame(result_rows)

    results_df.to_csv(
        os.path.join(
            OUT_METRICS,
            "cnn2d_v3_test_results.csv",
        ),
        index=False,
    )

    prediction_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "y_true": y_true,
            "y_prob": best_aggregation["y_prob"],
            "y_pred": (best_aggregation["y_prob"] >= 0.5).astype(int),
            "aggregation": best_aggregation["name"],
        }
    )

    prediction_df.to_csv(
        os.path.join(
            OUT_METRICS,
            "cnn2d_v3_subject_predictions.csv",
        ),
        index=False,
    )

    print(f"\n  Results saved to: {OUT_METRICS}")


def save_training_curves(history, best_epoch):
    epochs = [
        item["epoch"]
        for item in history
    ]

    train_loss = [
        item["tr_loss"]
        for item in history
    ]

    val_loss = [
        item["va_loss"]
        for item in history
    ]

    train_auc = [
        item["tr_auc"]
        for item in history
    ]

    val_auc = [
        item["va_auc"]
        for item in history
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
    )

    axes[0].plot(
        epochs,
        train_loss,
        label="Train Loss",
        color="steelblue",
    )

    axes[0].plot(
        epochs,
        val_loss,
        label="Val Loss",
        color="orange",
    )

    axes[0].axvline(
        best_epoch,
        linestyle="--",
        color="red",
        alpha=0.6,
        label=f"Best epoch {best_epoch}",
    )

    axes[0].set_title("Loss - CNN v3 FC-only")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(
        epochs,
        train_auc,
        label="Train AUC",
        color="steelblue",
    )

    axes[1].plot(
        epochs,
        val_auc,
        label="Val AUC",
        color="orange",
    )

    axes[1].axvline(
        best_epoch,
        linestyle="--",
        color="red",
        alpha=0.6,
    )

    axes[1].axhline(
        0.821,
        linestyle=":",
        color="green",
        alpha=0.6,
        label="Clinical 0.821",
    )

    axes[1].axhline(
        0.689,
        linestyle=":",
        color="purple",
        alpha=0.6,
        label="CNN v2 0.689",
    )

    axes[1].set_ylim(
        [
            0.4,
            1.05,
        ]
    )

    axes[1].set_title("AUC - CNN v3")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "CNN v3 - FC-only, Triplet Axial Input",
        fontsize=12,
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUT_PLOTS,
        "cnn2d_v3_training_curves.png",
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {output_path}")


def save_roc_plot(y_true, results):
    fig, ax = plt.subplots(
        figsize=(6, 6),
    )

    colors = {
        "mean": "steelblue",
        "median": "darkorange",
        "top-5 mean": "seagreen",
    }

    for result in results:
        fpr, tpr, _ = roc_curve(
            y_true,
            result["y_prob"],
        )

        ax.plot(
            fpr,
            tpr,
            lw=2,
            color=colors[result["name"]],
            label=(
                f"{result['name']} "
                f"AUC={result['auc']:.3f} "
                f"[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]"
            ),
        )

    ax.plot(
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

    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("CNN v3 - Subject-Level ROC by Aggregation")
    ax.legend(
        loc="lower right",
        fontsize=9,
    )

    ax.grid(alpha=0.3)

    output_path = os.path.join(
        OUT_PLOTS,
        "cnn2d_v3_roc_curves.png",
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {output_path}")


def save_version_comparison_plot(results):
    versions = [
        "v1\nwrong split\nfull FT",
        "v2\nfull split\nlayer4+fc",
        "Clinical\nPhase 1",
    ]

    aucs = [
        0.582,
        0.689,
        0.821,
    ]

    ci_low = [
        0.443,
        0.579,
        0.751,
    ]

    ci_high = [
        0.706,
        0.793,
        0.885,
    ]

    for result in results:
        versions.append(
            f"v3\n{result['name']}"
        )

        aucs.append(result["auc"])
        ci_low.append(result["ci_lo"])
        ci_high.append(result["ci_hi"])

    colors = [
        "#d9534f",
        "#f0ad4e",
        "#5cb85c",
    ] + ["#5bc0de"] * len(results)

    lower_error = [
        auc - low
        for auc, low in zip(aucs, ci_low)
    ]

    upper_error = [
        high - auc
        for auc, high in zip(aucs, ci_high)
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4),
    )

    bars = ax.bar(
        versions,
        aucs,
        color=colors,
        alpha=0.85,
        yerr=[
            lower_error,
            upper_error,
        ],
        capsize=5,
        error_kw={
            "elinewidth": 1.5,
        },
    )

    ax.set_ylim(
        [
            0.4,
            1.0,
        ]
    )

    ax.set_ylabel("AUC-ROC")
    ax.set_title("CNN Performance Across Versions")
    ax.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, auc in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            auc + 0.015,
            f"{auc:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    fig.tight_layout()

    output_path = os.path.join(
        OUT_PLOTS,
        "cnn2d_version_comparison.png",
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {output_path}")


def save_plots(history, best_epoch, y_true, results):
    section("Saving plots")

    try:
        save_training_curves(
            history,
            best_epoch,
        )

        save_roc_plot(
            y_true,
            results,
        )

        save_version_comparison_plot(
            results,
        )

    except Exception as error:
        print(f"  Plot error: {error}")


def print_final_summary(best_epoch, best_val_auc, results, best_aggregation):
    print()
    print("=" * 65)
    print("  PHASE 2 CNN v3 COMPLETE")
    print("=" * 65)

    print(f"  Best epoch : {best_epoch}")
    print(f"  Val AUC    : {best_val_auc:.4f}")
    print("  Trainable  : final FC layer only, 1,026 parameters")
    print("  Input      : triplet axial channels")

    print("\n  Aggregation comparison:")

    for result in results:
        marker = " <- best" if result["name"] == best_aggregation["name"] else ""

        print(
            f"    {result['name']:<12} "
            f"AUC={result['auc']:.4f} "
            f"[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]"
            f"{marker}"
        )

    print("\n  Version history using the same test cohort:")
    print("    v1  full fine-tune, 442 subjects : 0.582 [0.443, 0.706]")
    print("    v2  layer4+fc, 701 subjects      : 0.689 [0.579, 0.793]")

    print(
        f"    v3  fc-only, triplet input       : "
        f"{best_aggregation['auc']:.3f} "
        f"[{best_aggregation['ci_lo']:.3f}, {best_aggregation['ci_hi']:.3f}]"
    )

    print("    Clinical Phase 1                 : 0.821 [0.751, 0.885]")

    print("\n  Decision gate:")
    print("    If AUC >= 0.70, lock Phase 2 and proceed to fusion.")
    print("    If AUC < 0.70, still lock Phase 2 as an honest result and proceed.")

    print("\n  Next:")
    print("    extract_mri_embeddings.py")
    print("    train_fusion_clinical_mri.py")
    print("=" * 65)


def main():
    print_start_message()

    subject_slices, subject_labels, all_subjects = build_subject_manifest()

    triplet_df = build_triplet_dataframe(
        subject_slices,
        subject_labels,
        all_subjects,
    )

    train_df, val_df, test_df = load_subject_split(
        triplet_df,
    )

    train_loader, val_loader, test_loader, class_counts = create_dataloaders(
        train_df,
        val_df,
        test_df,
    )

    model = build_model()

    criterion, optimizer, scheduler = create_training_objects(
        model,
        class_counts,
    )

    best_epoch, best_val_auc, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
    )

    subject_ids, y_true, results, best_aggregation = evaluate_test_set(
        model,
        test_loader,
        test_df,
        best_epoch,
        best_val_auc,
    )

    save_results(
        subject_ids,
        y_true,
        results,
        best_aggregation,
        best_epoch,
        best_val_auc,
    )

    save_plots(
        history,
        best_epoch,
        y_true,
        results,
    )

    print_final_summary(
        best_epoch,
        best_val_auc,
        results,
        best_aggregation,
    )


if __name__ == "__main__":
    main()