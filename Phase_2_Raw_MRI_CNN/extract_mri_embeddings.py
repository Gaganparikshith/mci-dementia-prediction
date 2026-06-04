#!/usr/bin/env python3
"""
README - extract_mri_embeddings.py
==================================

This script extracts subject-level MRI embeddings for Phase 3 fusion.

It loads the trained ResNet-18 v3 checkpoint, removes the final classification
layer, and uses the backbone to generate 512-dimensional embeddings from MRI
slice triplets.

Important
---------
The input format must match CNN v3 training.

CNN v3 used 3-channel axial triplets:
    channel 1 = previous slice
    channel 2 = current slice
    channel 3 = next slice

Using single-channel slices here would create a mismatch with the trained
model setup.

Processing logic
----------------
1. Scan the preprocessed pMCI and sMCI slice folders.
2. Build ordered slice lists for each subject.
3. Create triplet inputs using previous, current, and next slices.
4. Load best_resnet18_v3.pt.
5. Replace the final FC layer with Identity.
6. Extract 512-dimensional embeddings for every triplet.
7. Average triplet embeddings to get one embedding per subject.
8. Save subject-level embeddings to CSV.
Run
---
python extract_mri_embeddings.py

Next step
---------
python train_fusion_clinical_mri.py
"""

import glob
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


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

OUT_DIR = os.path.join(
    DATA_ROOT,
    "data",
    "processed",
)

OUT_CSV = os.path.join(
    OUT_DIR,
    "mri_embeddings_resnet18.csv",
)

os.makedirs(OUT_DIR, exist_ok=True)


# Settings
BATCH_SIZE = 32
EMBEDDING_DIM = 512
DEVICE = torch.device("cpu")


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
    print("  EXTRACT MRI EMBEDDINGS - ResNet-18 v3")
    print(f"  Run at : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Model  : best_resnet18_v3.pt")
    print("  Input  : triplet axial slices")
    print("=" * 65)


def scan_subject_slices():
    """
    Scan pMCI and sMCI slice folders and build ordered slice lists per subject.
    """
    section("Scanning MRI slices")

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
            raise FileNotFoundError(f"Missing folder: {class_dir}")

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
        for subject_id in all_subjects
        if subject_labels[subject_id] == 1
    )

    n_smci = sum(
        1
        for subject_id in all_subjects
        if subject_labels[subject_id] == 0
    )

    print(f"  Subjects : {len(all_subjects)}")
    print(f"  pMCI     : {n_pmci}")
    print(f"  sMCI     : {n_smci}")

    if len(all_subjects) == 0:
        raise RuntimeError("No MRI slices found. Run preprocess_mri_2d_slices.py first.")

    return subject_slices, subject_labels, all_subjects


def build_triplet_dataframe(subject_slices, subject_labels, all_subjects):
    """
    Build triplet records using previous, current, and next axial slices.
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

    print(f"  Triplets : {len(triplet_df)}")

    return triplet_df


def load_resnet_backbone():
    """
    Load the trained ResNet-18 v3 checkpoint and remove the classifier layer.
    """
    section("Loading ResNet-18 v3 model")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Run train_mri_cnn_2d_v3.py first."
        )

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        EMBEDDING_DIM,
        2,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        epoch = checkpoint.get("epoch", "?")
        val_auc = checkpoint.get("val_auc", "?")

        if isinstance(val_auc, float):
            print(f"  Checkpoint loaded: epoch={epoch}, val_AUC={val_auc:.4f}")
        else:
            print("  Checkpoint loaded.")

    else:
        model.load_state_dict(checkpoint)
        print("  State dictionary loaded.")

    model.fc = nn.Identity()

    model.eval()
    model.to(DEVICE)

    print(f"  Final FC replaced with Identity")
    print(f"  Output embedding size: {EMBEDDING_DIM}")

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


def create_dataloader(triplet_df):
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

    return loader


def extract_embeddings(model, loader, triplet_df, all_subjects):
    """
    Extract triplet-level embeddings and group them by subject.
    """
    section("Extracting MRI embeddings")

    print(
        f"  {len(all_subjects)} subjects x about 20 triplets "
        f"x {EMBEDDING_DIM} dimensions"
    )

    subject_embeddings = {}

    start_time = time.time()

    with torch.no_grad():
        for batch_index, (images, subject_ids) in enumerate(loader):
            images = images.to(DEVICE)

            embeddings = model(images).cpu().numpy()

            for embedding, subject_id in zip(embeddings, subject_ids):
                if subject_id not in subject_embeddings:
                    subject_embeddings[subject_id] = []

                subject_embeddings[subject_id].append(embedding)

            if (batch_index + 1) % 50 == 0:
                completed = min(
                    (batch_index + 1) * BATCH_SIZE,
                    len(triplet_df),
                )

                percent = completed / len(triplet_df) * 100

                print(
                    f"  [{percent:5.1f}%] "
                    f"triplets: {completed}/{len(triplet_df)} "
                    f"elapsed: {time.time() - start_time:.0f}s"
                )

    print(f"\n  Done in {time.time() - start_time:.1f}s")

    return subject_embeddings


def aggregate_subject_embeddings(subject_embeddings, subject_labels, all_subjects):
    """
    Average triplet embeddings to create one 512-dimensional vector per subject.
    """
    section("Aggregating embeddings to subject level")

    rows = []

    for subject_id in all_subjects:
        if subject_id not in subject_embeddings:
            print(f"  WARNING: No embeddings found for {subject_id}. Skipped.")
            continue

        mean_embedding = np.stack(
            subject_embeddings[subject_id],
        ).mean(axis=0)

        rows.append(
            [
                subject_id,
                subject_labels[subject_id],
                *mean_embedding.tolist(),
            ]
        )

    embedding_columns = [
        f"emb_{index:03d}"
        for index in range(EMBEDDING_DIM)
    ]

    output_df = pd.DataFrame(
        rows,
        columns=[
            "subject_id",
            "label",
            *embedding_columns,
        ],
    )

    print(f"  Subjects embedded : {len(output_df)}")
    print(f"  pMCI              : {(output_df['label'] == 1).sum()}")
    print(f"  sMCI              : {(output_df['label'] == 0).sum()}")
    print(f"  Embedding shape   : ({len(output_df)}, {EMBEDDING_DIM})")

    return output_df


def save_embeddings(output_df):
    section("Saving embeddings")

    output_df.to_csv(
        OUT_CSV,
        index=False,
    )

    print(f"  Saved: {OUT_CSV}")
    print(f"  Shape: {output_df.shape}")


def print_final_summary(output_df):
    print()
    print("=" * 65)
    print("  EMBEDDING EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"  Rows    : {len(output_df)}")
    print("  Columns : subject_id, label, emb_000 to emb_511")
    print("  Next    : train_fusion_clinical_mri.py")
    print("=" * 65)


def main():
    print_start_message()

    subject_slices, subject_labels, all_subjects = scan_subject_slices()

    triplet_df = build_triplet_dataframe(
        subject_slices,
        subject_labels,
        all_subjects,
    )

    model = load_resnet_backbone()

    loader = create_dataloader(
        triplet_df,
    )

    subject_embeddings = extract_embeddings(
        model,
        loader,
        triplet_df,
        all_subjects,
    )

    output_df = aggregate_subject_embeddings(
        subject_embeddings,
        subject_labels,
        all_subjects,
    )

    save_embeddings(
        output_df,
    )

    print_final_summary(
        output_df,
    )


if __name__ == "__main__":
    main()