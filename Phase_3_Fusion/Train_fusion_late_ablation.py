#!/usr/bin/env python3
"""
README - train_fusion_late_ablation.py
======================================

This script performs the final Phase 3 fusion ablation.

It compares clinical-only, MRI-only, early fusion, late fusion, and
uncertain-case fusion strategies on the same held-out test cohort.

The goal is to check whether MRI information adds useful predictive value
beyond the Phase 1 clinical XGBoost model.

Strategies tested
-----------------
A. Clinical-only XGBoost
   Baseline clinical model.

B. MRI-only CNN v3 probability
   Uses subject-level CNN v3 probabilities.

C. Early fusion with XGBoost
   Clinical features + PCA-reduced MRI embeddings.

D. Early fusion with Logistic Regression
   Clinical features + PCA-reduced MRI embeddings.

E. Late fusion
   Weighted average of clinical and MRI probabilities.

F. Uncertain-case fusion
   MRI adjusts the clinical prediction only when clinical confidence is low.

Inputs
------
final_metadata.csv
    Clinical metadata and labels.

mri_embeddings_resnet18.csv
    Subject-level 512-dimensional MRI embeddings.

cnn2d_v2_subject_split.json
    Same train/validation/test subject split used in Phase 2.

cnn2d_v3_subject_predictions.csv
    Subject-level CNN v3 MRI probabilities.

Run
---
python train_fusion_late_ablation.py

Next step
---------
gradcam_mri_2d.py
streamlit_app.py
paper draft
"""

import json
import os
import pickle
import re
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")
matplotlib.use("Agg")


# Paths
DATA_ROOT = r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch"

CLINICAL_CSV = os.path.join(
    DATA_ROOT,
    "data",
    "metadata",
    "final_metadata.csv",
)

EMBED_CSV = os.path.join(
    DATA_ROOT,
    "data",
    "processed",
    "mri_embeddings_resnet18.csv",
)

SPLIT_JSON = os.path.join(
    DATA_ROOT,
    "results",
    "metrics",
    "cnn2d_v2_subject_split.json",
)

CNN_PREDS = os.path.join(
    DATA_ROOT,
    "results",
    "metrics",
    "cnn2d_v3_subject_predictions.csv",
)

OUT_METRICS = os.path.join(
    DATA_ROOT,
    "results",
    "metrics",
)

OUT_PLOTS = os.path.join(
    DATA_ROOT,
    "plots",
    "fusion",
)

OUT_MODELS = os.path.join(
    DATA_ROOT,
    "models",
)

os.makedirs(OUT_METRICS, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)
os.makedirs(OUT_MODELS, exist_ok=True)


# Settings
SEED = 42
PCA_COMPONENTS = [4, 8, 16, 32]
LATE_FUSION_WEIGHTS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

np.random.seed(SEED)


def section(title):
    print()
    print("-" * 65)
    print(f"  {title}")
    print("-" * 65)


def print_start_message():
    print("=" * 65)
    print("  PHASE 3 FINAL ABLATION - EARLY AND LATE FUSION")
    print(f"  Run at : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


def extract_rid(value):
    """
    Extract numeric RID from ADNI-style subject IDs.

    Examples:
        002_S_0413 -> 413
        413        -> 413
    """
    value = str(value).strip()

    match = re.match(
        r"^\d+_S_(\d+)$",
        value,
    )

    if match:
        return int(match.group(1))

    try:
        return int(float(value))
    except Exception:
        return np.nan


def load_clinical_data():
    """
    Load clinical metadata and prepare RID for merging.
    """
    clinical_df = pd.read_csv(CLINICAL_CSV)

    clinical_df = clinical_df.rename(
        columns={
            "LABEL": "label",
        }
    )

    clinical_df["RID_int"] = clinical_df["RID"].apply(extract_rid)

    return clinical_df


def load_mri_embeddings():
    """
    Load subject-level MRI embeddings and prepare RID for merging.
    """
    embedding_df = pd.read_csv(EMBED_CSV)

    embedding_df["subject_id"] = (
        embedding_df["subject_id"]
        .astype(str)
        .str.strip()
    )

    embedding_df["RID_int"] = embedding_df["subject_id"].apply(extract_rid)

    embedding_columns = [
        column
        for column in embedding_df.columns
        if column.startswith("emb_")
    ]

    return embedding_df, embedding_columns


def merge_clinical_and_embeddings(clinical_df, embedding_df, embedding_columns):
    """
    Merge clinical features with MRI embeddings on RID.
    """
    metadata_columns = {
        "RID",
        "RID_int",
        "label",
        "subject_id",
    }

    clinical_feature_columns = [
        column
        for column in clinical_df.columns
        if column not in metadata_columns
    ]

    merged_df = clinical_df[
        [
            "RID_int",
            "label",
            *clinical_feature_columns,
        ]
    ].merge(
        embedding_df[
            [
                "RID_int",
                "subject_id",
                *embedding_columns,
            ]
        ],
        on="RID_int",
        how="inner",
    )

    print(
        f"\n  Merged subjects : {len(merged_df)} "
        f"pMCI={(merged_df['label'] == 1).sum()} "
        f"sMCI={(merged_df['label'] == 0).sum()}"
    )

    return merged_df, clinical_feature_columns


def split_rids(subject_list):
    """
    Convert split subject IDs into integer RID values.
    """
    rid_set = set()

    for subject_id in subject_list:
        rid = extract_rid(subject_id)

        if not np.isnan(rid):
            rid_set.add(int(rid))

    return rid_set


def load_phase2_split(merged_df):
    """
    Load the saved Phase 2 subject split and apply it to the merged dataframe.
    """
    with open(SPLIT_JSON, "r", encoding="utf-8") as file:
        split = json.load(file)

    train_rids = split_rids(split["train_subjects"])
    val_rids = split_rids(split["val_subjects"])
    test_rids = split_rids(split["test_subjects"])

    train_df = merged_df[merged_df["RID_int"].isin(train_rids)].copy()
    val_df = merged_df[merged_df["RID_int"].isin(val_rids)].copy()
    test_df = merged_df[merged_df["RID_int"].isin(test_rids)].copy()

    print(
        f"  Train={len(train_df)} "
        f"Val={len(val_df)} "
        f"Test={len(test_df)}"
    )

    return train_df, val_df, test_df


def prepare_feature_matrices(train_df, val_df, test_df, clinical_columns, embedding_columns):
    """
    Prepare imputed and scaled clinical and MRI embedding matrices.
    """
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    clinical_imputer = SimpleImputer(strategy="median")
    clinical_scaler = StandardScaler()

    x_clin_train = clinical_scaler.fit_transform(
        clinical_imputer.fit_transform(train_df[clinical_columns].values)
    )

    x_clin_val = clinical_scaler.transform(
        clinical_imputer.transform(val_df[clinical_columns].values)
    )

    x_clin_test = clinical_scaler.transform(
        clinical_imputer.transform(test_df[clinical_columns].values)
    )

    mri_imputer = SimpleImputer(strategy="median")
    mri_scaler = StandardScaler()

    x_mri_train_raw = mri_scaler.fit_transform(
        mri_imputer.fit_transform(train_df[embedding_columns].values)
    )

    x_mri_val_raw = mri_scaler.transform(
        mri_imputer.transform(val_df[embedding_columns].values)
    )

    x_mri_test_raw = mri_scaler.transform(
        mri_imputer.transform(test_df[embedding_columns].values)
    )

    return {
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "x_clin_train": x_clin_train,
        "x_clin_val": x_clin_val,
        "x_clin_test": x_clin_test,
        "x_mri_train_raw": x_mri_train_raw,
        "x_mri_val_raw": x_mri_val_raw,
        "x_mri_test_raw": x_mri_test_raw,
    }


def load_cnn_test_probabilities(test_df):
    """
    Load CNN v3 subject probabilities and align them to the fusion test set.
    """
    cnn_df = pd.read_csv(CNN_PREDS)

    cnn_df["RID_int"] = cnn_df["subject_id"].apply(extract_rid)

    cnn_probability_map = dict(
        zip(
            cnn_df["RID_int"],
            cnn_df["y_prob"],
        )
    )

    cnn_probabilities = np.array(
        [
            cnn_probability_map.get(rid, 0.5)
            for rid in test_df["RID_int"]
        ]
    )

    matched_count = sum(
        1
        for rid in test_df["RID_int"]
        if rid in cnn_probability_map
    )

    print(f"  CNN v3 probabilities matched to test: {matched_count}/{len(test_df)}")

    return cnn_probabilities


def get_xgboost_params(y_train):
    """
    Build XGBoost parameter dictionary using training class balance.
    """
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / max(positive_count, 1)

    return {
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "random_state": SEED,
        "n_jobs": -1,
    }


def train_xgb_model(x_train, y_train, x_val, y_val, xgb_params):
    """
    Train XGBoost with validation early stopping, then refit using train + val.
    """
    first_model = xgb.XGBClassifier(
        **{
            **xgb_params,
            "n_estimators": 500,
            "early_stopping_rounds": 30,
        }
    )

    first_model.fit(
        x_train,
        y_train,
        eval_set=[
            (
                x_val,
                y_val,
            )
        ],
        verbose=False,
    )

    best_n_estimators = first_model.best_iteration + 1

    final_model = xgb.XGBClassifier(
        **{
            **xgb_params,
            "n_estimators": best_n_estimators,
        }
    )

    final_model.fit(
        np.vstack(
            [
                x_train,
                x_val,
            ]
        ),
        np.concatenate(
            [
                y_train,
                y_val,
            ]
        ),
        verbose=False,
    )

    return final_model


def train_lr_model(x_train, y_train):
    """
    Train balanced Logistic Regression.
    """
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=SEED,
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


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

    ci_low, ci_high = np.percentile(
        auc_values,
        [
            2.5,
            97.5,
        ],
    )

    return ci_low, ci_high


def evaluate_strategy(y_true, y_prob, label):
    """
    Evaluate one fusion strategy.
    """
    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    auc = roc_auc_score(
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

    return {
        "label": label,
        "AUC": auc,
        "CI_lo": ci_low,
        "CI_hi": ci_high,
        "F1": f1,
        "Sens": sensitivity,
        "Spec": specificity,
        "y_prob": y_prob,
    }


def log_result(result, all_results):
    """
    Print and store one result row.
    """
    all_results.append(result)

    print(
        f"  {result['label']:<50} "
        f"AUC={result['AUC']:.4f} "
        f"[{result['CI_lo']:.3f},{result['CI_hi']:.3f}] "
        f"Sens={result['Sens']:.3f} "
        f"Spec={result['Spec']:.3f}"
    )


def run_baselines(matrices, cnn_prob_test, xgb_params, all_results):
    """
    Run clinical-only and MRI-only baselines.
    """
    section("A. Baselines")

    clinical_model = train_xgb_model(
        matrices["x_clin_train"],
        matrices["y_train"],
        matrices["x_clin_val"],
        matrices["y_val"],
        xgb_params,
    )

    clinical_prob = clinical_model.predict_proba(
        matrices["x_clin_test"]
    )[:, 1]

    log_result(
        evaluate_strategy(
            matrices["y_test"],
            clinical_prob,
            "Clinical-only XGBoost",
        ),
        all_results,
    )

    log_result(
        evaluate_strategy(
            matrices["y_test"],
            cnn_prob_test,
            "MRI-only CNN v3 direct probability",
        ),
        all_results,
    )

    model_path = os.path.join(
        OUT_MODELS,
        "ablation_xgb_clinical.pkl",
    )

    with open(model_path, "wb") as file:
        pickle.dump(
            clinical_model,
            file,
        )

    return clinical_model, clinical_prob


def run_early_fusion(matrices, xgb_params, all_results):
    """
    Run early fusion using PCA-reduced MRI embeddings.
    """
    section("C/D. Early fusion PCA ablation")

    for n_components in PCA_COMPONENTS:
        pca = PCA(
            n_components=n_components,
            random_state=SEED,
        )

        x_mri_train = pca.fit_transform(
            matrices["x_mri_train_raw"]
        )

        x_mri_val = pca.transform(
            matrices["x_mri_val_raw"]
        )

        x_mri_test = pca.transform(
            matrices["x_mri_test_raw"]
        )

        explained_variance = pca.explained_variance_ratio_.sum()

        x_fusion_train = np.hstack(
            [
                matrices["x_clin_train"],
                x_mri_train,
            ]
        )

        x_fusion_val = np.hstack(
            [
                matrices["x_clin_val"],
                x_mri_val,
            ]
        )

        x_fusion_test = np.hstack(
            [
                matrices["x_clin_test"],
                x_mri_test,
            ]
        )

        xgb_fusion_model = train_xgb_model(
            x_fusion_train,
            matrices["y_train"],
            x_fusion_val,
            matrices["y_val"],
            xgb_params,
        )

        xgb_prob = xgb_fusion_model.predict_proba(
            x_fusion_test
        )[:, 1]

        log_result(
            evaluate_strategy(
                matrices["y_test"],
                xgb_prob,
                f"Early-XGB PCA={n_components:2d} ({explained_variance:.0%} var)",
            ),
            all_results,
        )

        lr_fusion_model = train_lr_model(
            x_fusion_train,
            matrices["y_train"],
        )

        lr_prob = lr_fusion_model.predict_proba(
            x_fusion_test
        )[:, 1]

        log_result(
            evaluate_strategy(
                matrices["y_test"],
                lr_prob,
                f"Early-LR PCA={n_components:2d} ({explained_variance:.0%} var)",
            ),
            all_results,
        )

        mri_lr_model = train_lr_model(
            x_mri_train,
            matrices["y_train"],
        )

        mri_lr_prob = mri_lr_model.predict_proba(
            x_mri_test
        )[:, 1]

        log_result(
            evaluate_strategy(
                matrices["y_test"],
                mri_lr_prob,
                f"MRI-PCA-LR PCA={n_components:2d}",
            ),
            all_results,
        )


def run_late_fusion(y_test, clinical_prob, cnn_prob_test, all_results):
    """
    Run late fusion using weighted clinical and MRI probabilities.
    """
    section("E. Late fusion weight sweep")

    for clinical_weight in LATE_FUSION_WEIGHTS:
        mri_weight = 1 - clinical_weight

        late_prob = (
            clinical_weight * clinical_prob
            + mri_weight * cnn_prob_test
        )

        log_result(
            evaluate_strategy(
                y_test,
                late_prob,
                f"Late fusion w_clin={clinical_weight:.2f} w_mri={mri_weight:.2f}",
            ),
            all_results,
        )


def run_uncertain_case_fusion(y_test, clinical_prob, cnn_prob_test, all_results):
    """
    Adjust clinical probabilities only for uncertain clinical predictions.
    """
    section("F. Uncertain-case MRI correction")

    configs = [
        (
            0.35,
            0.65,
            0.20,
        ),
        (
            0.35,
            0.65,
            0.30,
        ),
        (
            0.40,
            0.60,
            0.20,
        ),
    ]

    for lower_threshold, upper_threshold, mri_weight in configs:
        uncertain_prob = clinical_prob.copy()

        uncertain_mask = (
            (clinical_prob >= lower_threshold)
            & (clinical_prob <= upper_threshold)
        )

        uncertain_prob[uncertain_mask] = (
            (1 - mri_weight) * clinical_prob[uncertain_mask]
            + mri_weight * cnn_prob_test[uncertain_mask]
        )

        adjusted_count = int(uncertain_mask.sum())

        log_result(
            evaluate_strategy(
                y_test,
                uncertain_prob,
                (
                    f"Uncertain [{lower_threshold},{upper_threshold}] "
                    f"w_mri={mri_weight} ({adjusted_count} adjusted)"
                ),
            ),
            all_results,
        )


def save_ranked_results(all_results):
    """
    Save all ablation results sorted by test AUC.
    """
    section("Ranked results by test AUC")

    results_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key != "y_prob"
            }
            for result in all_results
        ]
    )

    results_df = (
        results_df
        .sort_values(
            "AUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        f"\n  {'Rank':>4}  {'AUC':>6}  {'CI':>14}  "
        f"{'Sens':>5}  {'Spec':>5}  Strategy"
    )

    print("  " + "-" * 75)

    for index, row in results_df.iterrows():
        marker = " <- BEST" if index == 0 else ""

        print(
            f"  {index + 1:>4}  "
            f"{row['AUC']:.4f}  "
            f"[{row['CI_lo']:.3f},{row['CI_hi']:.3f}]  "
            f"{row['Sens']:.3f}  "
            f"{row['Spec']:.3f}  "
            f"{row['label']}{marker}"
        )

    output_path = os.path.join(
        OUT_METRICS,
        "fusion_ablation_ranked.csv",
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"\n  Saved: {output_path}")

    return results_df


def get_plot_color(label):
    """
    Assign plot color based on fusion strategy.
    """
    if "Clinical-only" in label:
        return "steelblue"

    if "Late" in label:
        return "seagreen"

    if "Uncertain" in label:
        return "mediumorchid"

    if "Early-XGB" in label:
        return "darkorange"

    if "Early-LR" in label:
        return "coral"

    return "gray"


def save_ablation_plot(results_df):
    """
    Save AUC comparison plot for the top fusion strategies.
    """
    section("Saving ablation plot")

    try:
        top_results = results_df.head(12).iloc[::-1]

        colors = [
            get_plot_color(label)
            for label in top_results["label"]
        ]

        lower_error = (
            top_results["AUC"] - top_results["CI_lo"]
        ).values

        upper_error = (
            top_results["CI_hi"] - top_results["AUC"]
        ).values

        fig, axis = plt.subplots(
            figsize=(10, 6),
        )

        bars = axis.barh(
            top_results["label"],
            top_results["AUC"],
            color=colors,
            alpha=0.85,
            xerr=[
                lower_error,
                upper_error,
            ],
            capsize=4,
            error_kw={
                "elinewidth": 1.2,
            },
        )

        axis.axvline(
            0.855,
            linestyle="--",
            color="steelblue",
            alpha=0.5,
            linewidth=1.5,
            label="Clinical-only baseline 0.855",
        )

        axis.set_xlabel("AUC-ROC")
        axis.set_title("Phase 3 Fusion Ablation - Top Strategies")
        axis.set_xlim(
            [
                0.5,
                1.0,
            ]
        )

        for bar, auc in zip(bars, top_results["AUC"]):
            axis.text(
                auc + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}",
                va="center",
                fontsize=8,
            )

        axis.legend(
            handles=[
                Patch(color="steelblue", label="Clinical baseline"),
                Patch(color="seagreen", label="Late fusion"),
                Patch(color="mediumorchid", label="Uncertain-case"),
                Patch(color="darkorange", label="Early-XGB"),
                Patch(color="coral", label="Early-LR"),
            ],
            fontsize=8,
            loc="lower right",
        )

        axis.grid(
            axis="x",
            alpha=0.3,
        )

        fig.tight_layout()

        output_path = os.path.join(
            OUT_PLOTS,
            "fusion_ablation_comparison.png",
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"  Saved: {output_path}")

    except Exception as error:
        print(f"  Plot warning: {error}")


def print_final_verdict(results_df, all_results):
    """
    Print final Phase 3 decision.
    """
    best_result = results_df.iloc[0]

    baseline_auc = next(
        result["AUC"]
        for result in all_results
        if result["label"] == "Clinical-only XGBoost"
    )

    delta = best_result["AUC"] - baseline_auc

    print()
    print("=" * 65)
    print("  PHASE 3 ABLATION COMPLETE")
    print("=" * 65)

    print(f"  Clinical-only baseline : AUC={baseline_auc:.4f}")
    print(f"  Best strategy          : {best_result['label']}")
    print(
        f"  Best AUC               : "
        f"{best_result['AUC']:.4f} "
        f"[{best_result['CI_lo']:.3f}, {best_result['CI_hi']:.3f}]"
    )
    print(f"  Delta vs clinical-only : {delta:+.4f}")

    print()

    if delta > 0.005:
        print("  Fusion improves over clinical-only.")
        print("  Use the best fusion strategy as the final Phase 3 model.")
        print("  Lock Phase 3 with this result.")

    else:
        print("  Clinical-only remains the dominant model.")
        print("  Lock Phase 3 as an honest negative fusion result.")
        print()
        print("  Paper conclusion:")
        print("  MRI embeddings capture moderate signal, but they do not provide")
        print("  meaningful incremental benefit over the clinical model in fusion.")
        print("  Clinical assessments remain the strongest predictive modality.")

    print()
    print("  Next:")
    print("    gradcam_mri_2d.py")
    print("    streamlit_app.py")
    print("    paper draft")
    print("=" * 65)


def main():
    print_start_message()

    section("Loading and merging Phase 3 data")

    clinical_df = load_clinical_data()

    embedding_df, embedding_columns = load_mri_embeddings()

    merged_df, clinical_columns = merge_clinical_and_embeddings(
        clinical_df,
        embedding_df,
        embedding_columns,
    )

    train_df, val_df, test_df = load_phase2_split(
        merged_df,
    )

    matrices = prepare_feature_matrices(
        train_df,
        val_df,
        test_df,
        clinical_columns,
        embedding_columns,
    )

    cnn_prob_test = load_cnn_test_probabilities(
        test_df,
    )

    xgb_params = get_xgboost_params(
        matrices["y_train"],
    )

    all_results = []

    _, clinical_prob = run_baselines(
        matrices,
        cnn_prob_test,
        xgb_params,
        all_results,
    )

    run_early_fusion(
        matrices,
        xgb_params,
        all_results,
    )

    run_late_fusion(
        matrices["y_test"],
        clinical_prob,
        cnn_prob_test,
        all_results,
    )

    run_uncertain_case_fusion(
        matrices["y_test"],
        clinical_prob,
        cnn_prob_test,
        all_results,
    )

    results_df = save_ranked_results(
        all_results,
    )

    save_ablation_plot(
        results_df,
    )

    print_final_verdict(
        results_df,
        all_results,
    )


if __name__ == "__main__":
    main()