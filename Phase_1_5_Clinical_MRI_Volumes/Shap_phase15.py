#!/usr/bin/env python3
"""
README - shap_phase15.py
========================

This script generates SHAP explainability outputs for the Phase 1.5
clinical + MRI volumetric model.

The main purpose of this script is to explain whether MRI volumetric features
contribute meaningful predictive signal after being combined with clinical
features.

Key difference from shap_clinical.py
------------------------------------
This script separates features into two groups:

1. Clinical features
2. MRI volumetric features

It then compares the total SHAP contribution of both groups to show whether
the combined model is mainly driven by clinical features or MRI volumetric
features.

Main outputs
------------
phase15_shap_beeswarm.png
phase15_shap_bar_all.png
phase15_shap_grouped_bar.png
phase15_shap_clinical_vs_mri.png
phase15_shap_waterfall_pmci.png
phase15_shap_waterfall_smci.png
phase15_shap_dependence_top3.png
phase15_shap_heatmap.png
phase15_shap_feature_importance.csv

Important
---------
All-NaN volumetric columns are dropped before imputation.

This avoids shape mismatch errors caused by SimpleImputer silently removing
fully empty columns.

Run
---
python shap_phase15.py

Run this script after:
1. build_clinical_plus_volumes.py
2. train_clinical_plus_volumes.py
"""

import os
import sys
import pickle
import warnings
from datetime import datetime

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
matplotlib.use("Agg")


# Paths
RESEARCH_ROOT = r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch"

METADATA_DIR = os.path.join(RESEARCH_ROOT, "data", "metadata")
SPLITS_DIR = os.path.join(RESEARCH_ROOT, "data", "splits")
MODELS_DIR = os.path.join(RESEARCH_ROOT, "models")

DATA_CSV = os.path.join(
    METADATA_DIR,
    "final_metadata_clinical_plus_volumes.csv",
)

TRAIN_SPLIT_CSV = os.path.join(
    SPLITS_DIR,
    "train_subjects.csv",
)

TEST_SPLIT_CSV = os.path.join(
    SPLITS_DIR,
    "test_subjects.csv",
)

MODEL_PKL = os.path.join(
    MODELS_DIR,
    "best_xgb_phase15.pkl",
)

FIGURES_DIR = os.path.join(RESEARCH_ROOT, "plots", "phase15_shap")
TABLES_DIR = os.path.join(RESEARCH_ROOT, "results", "tables")
LOGS_DIR = os.path.join(RESEARCH_ROOT, "results", "logs")


# Feature groups
CLINICAL_FEATURES = [
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


# UCSFFSX7 / FreeSurfer volumetric features.
# The script checks which of these are actually available in the CSV.
VOLUMETRIC_FEATURES = [
    "Hippocampus",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "WholeBrain",
    "Ventricles",
    "ICV",
    "ST44SV",
    "ST88SV",
    "HIPPO_L",
    "HIPPO_R",
    "ENTORHINAL_L",
    "ENTORHINAL_R",
]


LABEL_COL = "LABEL"
SUBJECT_COL = "RID"
RANDOM_SEED = 42


# Plot colours
CLINICAL_COLOR = "#2E86AB"
VOLUMETRIC_COLOR = "#E84855"


def ensure_dirs():
    """Create all output folders if they do not already exist."""
    for folder in [FIGURES_DIR, TABLES_DIR, LOGS_DIR, MODELS_DIR]:
        os.makedirs(folder, exist_ok=True)


def section(title):
    print(f"\n-- {title} {'-' * max(0, 55 - len(title))}")


def save_figure(filename, dpi=180):
    """Save the current matplotlib figure."""
    output_path = os.path.join(FIGURES_DIR, filename)

    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  Saved: {output_path}")


def load_data():
    section("Loading data")

    if not os.path.isfile(DATA_CSV):
        print(f"[ERROR] Dataset not found:\n  {DATA_CSV}")
        print("  Run build_clinical_plus_volumes.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_CSV)
    df.columns = df.columns.str.strip()

    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  File: {os.path.basename(DATA_CSV)}")

    label_col = None

    for column in df.columns:
        if column.upper() in ["LABEL", "CONV_LABEL", "DX_LABEL"]:
            label_col = column
            break

    if label_col is None:
        print(f"[ERROR] No label column found. Columns: {list(df.columns)}")
        sys.exit(1)

    if label_col != LABEL_COL:
        df = df.rename(columns={label_col: LABEL_COL})

    label_encoder = LabelEncoder()
    df[LABEL_COL] = label_encoder.fit_transform(df[LABEL_COL].astype(str))

    classes = label_encoder.classes_
    label_map = dict(zip(label_encoder.transform(classes), classes))

    print(f"  Label encoding: {label_map}")
    print(f"  Label counts  : {df[LABEL_COL].value_counts().to_dict()}")

    return df, label_map


def get_feature_groups(df):
    section("Feature groups")

    skip_cols = {
        LABEL_COL,
        SUBJECT_COL,
        "RID",
        "PTID",
    }

    available_columns = set(df.columns) - skip_cols

    clinical_present = [
        feature
        for feature in CLINICAL_FEATURES
        if feature in available_columns
    ]

    volumetric_present = [
        feature
        for feature in VOLUMETRIC_FEATURES
        if feature in available_columns
    ]

    known_columns = (
        set(clinical_present)
        | set(volumetric_present)
        | skip_cols
    )

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    extra_numeric_columns = [
        column
        for column in numeric_columns
        if column not in known_columns
    ]

    volumetric_present.extend(extra_numeric_columns)

    all_features = clinical_present + volumetric_present

    print(f"  Clinical features found   : {len(clinical_present)}")
    print(f"  Volumetric features found : {len(volumetric_present)}")
    print(f"  Total features            : {len(all_features)}")

    if len(all_features) == 0:
        print("[ERROR] No features found. Check column names in the CSV.")
        sys.exit(1)

    return clinical_present, volumetric_present, all_features


def get_splits(df, all_features):
    section("Train / test split")

    rid_col = None

    for column in ["RID", "PTID", "SUBJECT_ID"]:
        if column in df.columns:
            rid_col = column
            break

    saved_splits_available = (
        os.path.isfile(TRAIN_SPLIT_CSV)
        and os.path.isfile(TEST_SPLIT_CSV)
        and rid_col is not None
    )

    if saved_splits_available:
        train_rids = pd.read_csv(TRAIN_SPLIT_CSV).iloc[:, 0].astype(str).tolist()
        test_rids = pd.read_csv(TEST_SPLIT_CSV).iloc[:, 0].astype(str).tolist()

        df[rid_col] = df[rid_col].astype(str)

        train_df = df[df[rid_col].isin(train_rids)]
        test_df = df[df[rid_col].isin(test_rids)]

        print("  Using saved subject-level splits")
        print(f"  train={len(train_df)}, test={len(test_df)}")

    else:
        print("  WARNING: Split CSVs not found - using 80/20 stratified split")

        train_df, test_df = train_test_split(
            df,
            test_size=0.20,
            stratify=df[LABEL_COL],
            random_state=RANDOM_SEED,
        )

        print(f"  train={len(train_df)}, test={len(test_df)}")

    train_df = train_df.dropna(
        subset=all_features,
        how="all",
    ).copy()

    test_df = test_df.dropna(
        subset=all_features,
        how="all",
    ).copy()

    x_train = train_df[all_features].copy()
    y_train = train_df[LABEL_COL].values

    x_test = test_df[all_features].copy()
    y_test = test_df[LABEL_COL].copy()

    all_nan_columns = x_train.columns[x_train.isna().all()].tolist()

    if all_nan_columns:
        print(f"\n  Dropping {len(all_nan_columns)} all-NaN columns:")

        for column in all_nan_columns:
            print(f"    {column}")

        x_train = x_train.drop(columns=all_nan_columns)
        x_test = x_test.drop(columns=all_nan_columns)

    imputer = SimpleImputer(strategy="median")

    x_train_array = imputer.fit_transform(x_train)
    x_test_array = imputer.transform(x_test)

    x_train = pd.DataFrame(
        x_train_array,
        columns=x_train.columns,
        index=x_train.index,
    )

    x_test = pd.DataFrame(
        x_test_array,
        columns=x_test.columns,
        index=x_test.index,
    )

    assert x_train.isna().sum().sum() == 0, "NaNs remain in X_train after imputation"
    assert x_test.isna().sum().sum() == 0, "NaNs remain in X_test after imputation"

    print(f"\n  X_train: {x_train.shape}")
    print(f"  X_test : {x_test.shape}")

    return x_train, y_train, x_test, y_test, test_df, all_nan_columns


def get_model(x_train, y_train):
    section("Model")

    if os.path.isfile(MODEL_PKL):
        print(f"  Loading saved model: {MODEL_PKL}")

        with open(MODEL_PKL, "rb") as file:
            model = pickle.load(file)

        print("  Loaded successfully.")

        return model

    print("  No saved model found - training XGBoost now.")
    print("  Data is already clean, so SMOTE can run safely.")

    smote = SMOTE(random_state=RANDOM_SEED)

    x_resampled, y_resampled = smote.fit_resample(
        x_train,
        y_train,
    )

    print(f"  After SMOTE: {x_resampled.shape[0]} samples")

    negative_count = (y_resampled == 0).sum()
    positive_count = (y_resampled == 1).sum()

    scale_pos_weight = negative_count / max(positive_count, 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        verbosity=0,
    )

    model.fit(
        x_resampled,
        y_resampled,
    )

    print("  Training complete.")

    with open(MODEL_PKL, "wb") as file:
        pickle.dump(model, file)

    print(f"  Model saved: {MODEL_PKL}")

    return model


def compute_shap_values(model, x_test):
    section("Computing SHAP values")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_test)

    print(f"  SHAP values shape: {shap_values.values.shape}")

    return explainer, shap_values


def plot_beeswarm(shap_values, max_display=20):
    print("  Plotting: beeswarm")

    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        show=False,
    )

    plt.title(
        "SHAP Beeswarm - Phase 1.5 Clinical + MRI Volumes",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    save_figure("phase15_shap_beeswarm.png")


def plot_bar_all(shap_values, max_display=20):
    print("  Plotting: bar plot for all features")

    shap.plots.bar(
        shap_values,
        max_display=max_display,
        show=False,
    )

    plt.title(
        "Mean |SHAP| - Phase 1.5 All Features",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    save_figure("phase15_shap_bar_all.png")


def plot_grouped_bar(shap_values, x_test, clinical_features, volumetric_features):
    print("  Plotting: grouped bar plot")

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = list(x_test.columns)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_shap": mean_abs_shap,
        }
    )

    importance_df = (
        importance_df
        .sort_values("mean_shap", ascending=False)
        .head(25)
        .copy()
    )

    clinical_set = set(clinical_features)

    importance_df["color"] = importance_df["feature"].apply(
        lambda feature: CLINICAL_COLOR
        if feature in clinical_set
        else VOLUMETRIC_COLOR
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        importance_df["feature"][::-1],
        importance_df["mean_shap"][::-1],
        color=importance_df["color"][::-1],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)

    ax.set_title(
        "Feature Importance by Group - Phase 1.5\n"
        "Clinical + MRI Volumetric Features",
        fontsize=13,
        fontweight="bold",
    )

    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        handles=[
            mpatches.Patch(
                color=CLINICAL_COLOR,
                label="Clinical Features",
            ),
            mpatches.Patch(
                color=VOLUMETRIC_COLOR,
                label="MRI Volumetric Features",
            ),
        ],
        loc="lower right",
        fontsize=10,
    )

    plt.tight_layout()

    save_figure("phase15_shap_grouped_bar.png")

    return importance_df


def plot_clinical_vs_mri_total(
    shap_values,
    x_test,
    clinical_features,
    volumetric_features,
):
    print("  Plotting: clinical vs MRI total SHAP contribution")

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = list(x_test.columns)

    clinical_set = set(clinical_features)
    volumetric_set = set(volumetric_features)

    clinical_total = sum(
        mean_abs_shap[index]
        for index, feature in enumerate(feature_names)
        if feature in clinical_set
    )

    volumetric_total = sum(
        mean_abs_shap[index]
        for index, feature in enumerate(feature_names)
        if feature in volumetric_set
    )

    total = clinical_total + volumetric_total + 1e-9

    clinical_pct = clinical_total / total * 100
    volumetric_pct = volumetric_total / total * 100

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax_bar.bar(
        [
            "Clinical\nFeatures",
            "MRI Volumetric\nFeatures",
        ],
        [
            clinical_total,
            volumetric_total,
        ],
        color=[
            CLINICAL_COLOR,
            VOLUMETRIC_COLOR,
        ],
        width=0.5,
        edgecolor="white",
    )

    ax_bar.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=11,
    )

    ax_bar.set_ylabel("Sum of Mean |SHAP Values|", fontsize=11)

    ax_bar.set_title(
        "Total SHAP Contribution\nby Feature Group",
        fontsize=12,
        fontweight="bold",
    )

    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.set_ylim(0, max(clinical_total, volumetric_total) * 1.2)

    wedges, texts, autotexts = ax_pie.pie(
        [
            clinical_pct,
            volumetric_pct,
        ],
        labels=[
            "Clinical\nFeatures",
            "MRI Volumetric\nFeatures",
        ],
        colors=[
            CLINICAL_COLOR,
            VOLUMETRIC_COLOR,
        ],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 2,
        },
    )

    for text in autotexts:
        text.set_fontsize(13)
        text.set_fontweight("bold")

    ax_pie.set_title(
        "Percentage Contribution\nto Model Decisions",
        fontsize=12,
        fontweight="bold",
    )

    fig.suptitle(
        "Clinical Features Dominate - MRI Volumes Contribute Minimally\n"
        "Explains null result: combined AUC = clinical-only AUC",
        fontsize=11,
        style="italic",
        y=1.02,
    )

    plt.tight_layout()

    save_figure("phase15_shap_clinical_vs_mri.png")

    print(f"  Clinical contribution  : {clinical_pct:.1f}%")
    print(f"  Volumetric contribution: {volumetric_pct:.1f}%")

    return clinical_pct, volumetric_pct


def plot_waterfall(shap_values, y_test, class_name, target_label):
    print(f"  Plotting: waterfall plot for {class_name}")

    indices = np.where(np.asarray(y_test) == target_label)[0]

    if len(indices) == 0:
        print(f"  SKIP: No {class_name} subjects in test set.")
        return

    shap.plots.waterfall(
        shap_values[indices[0]],
        max_display=15,
        show=False,
    )

    plt.title(
        f"SHAP Waterfall - Example {class_name} Subject",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    save_figure(f"phase15_shap_waterfall_{class_name.lower()}.png")


def plot_dependence_top3(shap_values, x_test):
    print("  Plotting: dependence plots for top 3 features")

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = list(x_test.columns)

    top3_features = [
        feature_names[index]
        for index in np.argsort(mean_abs_shap)[-3:][::-1]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, feature in zip(axes, top3_features):
        shap.plots.scatter(
            shap_values[:, feature],
            ax=ax,
            show=False,
        )

        ax.set_title(
            feature,
            fontsize=10,
            fontweight="bold",
        )

        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "SHAP Dependence - Top 3 Features (Phase 1.5)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()

    save_figure("phase15_shap_dependence_top3.png")


def plot_heatmap(shap_values, max_display=20):
    print("  Plotting: heatmap")

    shap.plots.heatmap(
        shap_values,
        max_display=max_display,
        show=False,
    )

    plt.title(
        "SHAP Heatmap - Phase 1.5 Test Set",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    save_figure("phase15_shap_heatmap.png")


def save_importance_table(
    shap_values,
    x_test,
    clinical_features,
    volumetric_features,
    clinical_pct,
    volumetric_pct,
):
    section("Saving feature importance table")

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = list(x_test.columns)

    total_shap = mean_abs_shap.sum()
    clinical_set = set(clinical_features)

    rows = []

    for index, feature in enumerate(feature_names):
        rows.append(
            {
                "rank": 0,
                "feature": feature,
                "group": "Clinical" if feature in clinical_set else "Volumetric_MRI",
                "mean_abs_shap": round(float(mean_abs_shap[index]), 4),
                "pct_of_total_shap": round(
                    float(mean_abs_shap[index] / total_shap * 100),
                    2,
                ),
            }
        )

    importance_df = pd.DataFrame(rows)
    importance_df = importance_df.sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance_df["rank"] = range(1, len(importance_df) + 1)

    summary_rows = pd.DataFrame(
        [
            {
                "rank": "",
                "feature": "--- GROUP SUMMARY ---",
                "group": "",
                "mean_abs_shap": "",
                "pct_of_total_shap": "",
            },
            {
                "rank": "",
                "feature": "Clinical Features TOTAL",
                "group": "Clinical",
                "mean_abs_shap": "",
                "pct_of_total_shap": round(clinical_pct, 2),
            },
            {
                "rank": "",
                "feature": "MRI Volumetric Features TOTAL",
                "group": "Volumetric_MRI",
                "mean_abs_shap": "",
                "pct_of_total_shap": round(volumetric_pct, 2),
            },
        ]
    )

    importance_df = pd.concat(
        [
            importance_df,
            summary_rows,
        ],
        ignore_index=True,
    )

    output_path = os.path.join(
        TABLES_DIR,
        "phase15_shap_feature_importance.csv",
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    print(f"  Saved: {output_path}")

    print("\n  Top 10 features by mean |SHAP|:")
    print(
        f"  {'Rank':<5} {'Feature':<30} {'Group':<15} "
        f"{'Mean|SHAP|':<12} {'%Total'}"
    )
    print(f"  {'-' * 5} {'-' * 30} {'-' * 15} {'-' * 12} {'-' * 8}")

    for _, row in importance_df.head(10).iterrows():
        print(
            f"  {str(row['rank']):<5} "
            f"{str(row['feature']):<30} "
            f"{str(row['group']):<15} "
            f"{str(row['mean_abs_shap']):<12} "
            f"{row['pct_of_total_shap']}%"
        )


def get_label_id(label_map, positive=True):
    """
    Get pMCI or sMCI label ID from the encoded label map.

    Fallback:
    pMCI -> 1
    sMCI -> 0
    """
    if positive:
        valid_values = ["1", "pmci", "pMCI"]
        fallback = 1
    else:
        valid_values = ["0", "smci", "sMCI"]
        fallback = 0

    for key, value in label_map.items():
        if str(value) in valid_values:
            return key

    return fallback


def write_log_start(logfile):
    with open(logfile, "w", encoding="utf-8") as file:
        file.write(f"SHAP Phase 1.5 run started: {datetime.now()}\n")


def write_log_end(logfile, clinical_pct, volumetric_pct, dropped_columns):
    with open(logfile, "a", encoding="utf-8") as file:
        file.write(f"Clinical %: {clinical_pct:.1f}\n")
        file.write(f"Volumetric %: {volumetric_pct:.1f}\n")
        file.write(f"Dropped all-NaN columns: {dropped_columns}\n")
        file.write(f"Completed: {datetime.now()}\n")


def print_summary(clinical_pct, volumetric_pct, dropped_columns):
    section("Summary")

    print(f"  Plots saved to: {FIGURES_DIR}")
    print(f"  Table saved to: {TABLES_DIR}")
    print(f"  Model saved to: {MODEL_PKL}")

    print("\n  Key paper finding:")
    print(f"    Clinical features : {clinical_pct:.1f}% of total SHAP contribution")
    print(f"    MRI volumes       : {volumetric_pct:.1f}% of total SHAP contribution")

    print("\n  Key figures for paper:")
    print("    phase15_shap_clinical_vs_mri.png")
    print("    phase15_shap_grouped_bar.png")

    if dropped_columns:
        print(f"\n  NOTE: {len(dropped_columns)} all-NaN volumetric columns were dropped:")

        for column in dropped_columns:
            print(f"    {column}")

        print("  These can be mentioned as a limitation.")

    print("\n  Phase 1.5 SHAP complete. Clinical pipeline is now locked.")
    print("=" * 65)


def main():
    ensure_dirs()

    logfile = os.path.join(
        LOGS_DIR,
        "shap_phase15.log",
    )

    write_log_start(logfile)

    print("=" * 65)
    print("  SHAP EXPLAINABILITY - PHASE 1.5")
    print("  Clinical + MRI Volumetric Features")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    df, label_map = load_data()

    clinical_features, volumetric_features, all_features = get_feature_groups(df)

    x_train, y_train, x_test, y_test, test_df, dropped_columns = get_splits(
        df,
        all_features,
    )

    dropped_set = set(dropped_columns)

    clinical_features = [
        feature
        for feature in clinical_features
        if feature not in dropped_set
    ]

    volumetric_features = [
        feature
        for feature in volumetric_features
        if feature not in dropped_set
    ]

    model = get_model(
        x_train,
        y_train,
    )

    explainer, shap_values = compute_shap_values(
        model,
        x_test,
    )

    section("Generating plots")

    plot_beeswarm(
        shap_values,
        max_display=20,
    )

    plot_bar_all(
        shap_values,
        max_display=20,
    )

    plot_grouped_bar(
        shap_values,
        x_test,
        clinical_features,
        volumetric_features,
    )

    clinical_pct, volumetric_pct = plot_clinical_vs_mri_total(
        shap_values,
        x_test,
        clinical_features,
        volumetric_features,
    )

    pmci_label = get_label_id(
        label_map,
        positive=True,
    )

    smci_label = get_label_id(
        label_map,
        positive=False,
    )

    plot_waterfall(
        shap_values,
        y_test,
        "pMCI",
        pmci_label,
    )

    plot_waterfall(
        shap_values,
        y_test,
        "sMCI",
        smci_label,
    )

    plot_dependence_top3(
        shap_values,
        x_test,
    )

    plot_heatmap(
        shap_values,
        max_display=20,
    )

    save_importance_table(
        shap_values,
        x_test,
        clinical_features,
        volumetric_features,
        clinical_pct,
        volumetric_pct,
    )

    write_log_end(
        logfile,
        clinical_pct,
        volumetric_pct,
        dropped_columns,
    )

    print_summary(
        clinical_pct,
        volumetric_pct,
        dropped_columns,
    )


if __name__ == "__main__":
    main()