#!/usr/bin/env python3
"""
README - validate_oasis2.py
===========================

This script performs external validation of the Phase 1 clinical XGBoost model
on the OASIS-2 dataset.

It loads the trained ADNI Phase 1 XGBoost model, aligns the available OASIS-2
features to the ADNI E4 feature format, predicts pMCI-like vs sMCI-like
conversion risk, and reports external validation performance.

Important
---------
OASIS-2 does not contain all ADNI E4 clinical features.

Only PTGENDER, PTEDUCAT, and MMSE_BL are available directly. The remaining
features are filled as NaN and handled by the trained model pipeline imputer.

This means the OASIS-2 result mainly reflects a feature-mismatch experiment,
not a complete model generalisation test.

Expected result
---------------
AUC-ROC around 0.47, close to chance level.

Reason:
Only 3/18 ADNI E4 features are available in OASIS-2.

Framing:
This should be reported as a feature-mismatch finding, not as a complete model
failure.

Run
---
python validate_oasis2.py

Run this script after:
1. train_clinical.py
2. build_oasis2_metadata.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


warnings.filterwarnings("ignore")
matplotlib.use("Agg")


# Paths
BASE_DIR = Path(r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch")

OASIS2_SEARCH_PATHS = [
    Path(r"C:\Users\ASUS\Desktop\Research Resources\MRI DATA\oasis2_validation_ready.csv"),
    Path(r"C:\Users\ASUS\Desktop\Research Resources\DATA\oasis2_validation_ready.csv"),
    BASE_DIR / "data" / "metadata" / "oasis2_validation_ready.csv",
    BASE_DIR / "outputs" / "predictions" / "oasis2_validation_ready.csv",
    BASE_DIR / "oasis2_validation_ready.csv",
]

MODEL_PATH = BASE_DIR / "best_xgb.pkl"

RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots" / "oasis2"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Settings
RANDOM_STATE = 42

PLOT_KW = {
    "dpi": 150,
    "bbox_inches": "tight",
}


# E4 feature set used during Phase 1 training
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

OASIS2_AVAILABLE_FEATURES = [
    "PTGENDER",
    "PTEDUCAT",
    "MMSE_BL",
]


def section(title):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC_ROC": roc_auc_score(y_true, y_prob),
        "AUC_PR": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Brier": brier_score_loss(y_true, y_prob),
        "N_pMCI": int(y_true.sum()),
        "N_sMCI": int((y_true == 0).sum()),
    }


def bootstrap_ci(y_true, y_prob, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)

    values = {
        "AUC_ROC": [],
        "AUC_PR": [],
        "F1": [],
        "Sensitivity": [],
        "Specificity": [],
    }

    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)

        sample_y = y_true[sample_idx]
        sample_prob = y_prob[sample_idx]

        if len(np.unique(sample_y)) < 2:
            continue

        sample_metrics = calculate_metrics(sample_y, sample_prob)

        for metric_name in values:
            values[metric_name].append(sample_metrics[metric_name])

    alpha = (1 - ci) / 2

    return {
        metric_name: (
            float(np.mean(metric_values)),
            float(np.percentile(metric_values, alpha * 100)),
            float(np.percentile(metric_values, (1 - alpha) * 100)),
        )
        for metric_name, metric_values in values.items()
    }


def find_oasis2_file():
    for path in OASIS2_SEARCH_PATHS:
        if path.exists():
            return path

    print("\n  OASIS-2 validation file not found.")
    print("  Searched:")

    for path in OASIS2_SEARCH_PATHS:
        print(f"    {path}")

    print("\n  Expected file: oasis2_validation_ready.csv")
    print("  Run build_oasis2_metadata.py first.")

    raise FileNotFoundError("oasis2_validation_ready.csv not found")


def load_oasis2_data():
    section("validate_oasis2.py - External Validation")

    oasis2_path = find_oasis2_file()

    print(f"\n  OASIS-2 file: {oasis2_path}")

    section("STEP 1 - Loading OASIS-2 data")

    df = pd.read_csv(oasis2_path)
    df.columns = df.columns.str.strip()

    print(f"  Loaded : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    return df


def find_first_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def label_oasis2_from_cdr(df):
    """
    Label OASIS-2 subjects using CDR trajectory.

    pMCI-like:
        baseline CDR = 0.5 and any follow-up CDR >= 1.0

    sMCI-like:
        baseline CDR = 0.5 and follow-up CDR stays <= 0.5
    """
    print("  Labelling OASIS-2 subjects from CDR trajectory...")

    cdr_col = find_first_column(
        df,
        [
            "CDR",
            "BL_CDR",
            "CDGLOBAL",
            "CDR_GLOBAL",
            "CDR_GLOBAL_BL",
            "cdr",
        ],
    )

    if cdr_col is None:
        raise KeyError(f"No CDR column found. Available columns: {list(df.columns)}")

    id_col = find_first_column(
        df,
        [
            "SUBJECT_ID",
            "ID",
            "Subject",
            "SUBJECT",
            "subject_id",
            "RID",
            "PTID",
        ],
    )

    if id_col is None:
        raise KeyError(f"No subject ID column found. Available columns: {list(df.columns)}")

    visit_col = find_first_column(
        df,
        [
            "Visit",
            "VISIT",
            "session",
            "SESSION",
            "VISCODE",
        ],
    )

    print(f"  CDR column    : {cdr_col}")
    print(f"  Subject column: {id_col}")
    print(f"  Visit column  : {visit_col}")

    df = df.copy()
    df[cdr_col] = pd.to_numeric(df[cdr_col], errors="coerce")

    if visit_col:
        df = df.sort_values([id_col, visit_col])

    labelled_rows = []

    for subject_id, subject_rows in df.groupby(id_col):
        subject_rows = subject_rows.reset_index(drop=True)

        baseline_cdr = subject_rows[cdr_col].iloc[0]

        if baseline_cdr != 0.5:
            continue

        if len(subject_rows) > 1:
            followup_cdr = subject_rows[cdr_col].iloc[1:]
        else:
            followup_cdr = pd.Series([], dtype=float)

        if (followup_cdr >= 1.0).any():
            label = 1
        elif (followup_cdr <= 0.5).all():
            label = 0
        else:
            continue

        labelled_rows.append(
            {
                id_col: subject_id,
                "CONV_LABEL": label,
                "bl_CDR": baseline_cdr,
                "n_visits": len(subject_rows),
                "max_CDR": subject_rows[cdr_col].max(),
            }
        )

    labelled_df = pd.DataFrame(labelled_rows)

    print(f"\n  Labelled subjects: {len(labelled_df)}")
    print(f"  pMCI-like (1)   : {(labelled_df['CONV_LABEL'] == 1).sum()}")
    print(f"  sMCI-like (0)   : {(labelled_df['CONV_LABEL'] == 0).sum()}")

    first_visit_df = df.groupby(id_col).first().reset_index()

    labelled_df = labelled_df.merge(
        first_visit_df,
        on=id_col,
        how="left",
    )

    return labelled_df


def prepare_labels(oasis_raw):
    section("STEP 2 - Labelling pMCI-like / sMCI-like")

    if "LABEL" in oasis_raw.columns:
        print("  LABEL column found - using existing labels")

        oasis_labelled = oasis_raw.copy()
        oasis_labelled["CONV_LABEL"] = oasis_labelled["LABEL"].astype(int)

    else:
        print("  No LABEL column found - deriving labels from CDR trajectory")
        oasis_labelled = label_oasis2_from_cdr(oasis_raw)

    if len(oasis_labelled) == 0:
        raise ValueError("No subjects labelled. Check LABEL/CDR values.")

    y_oasis = oasis_labelled["CONV_LABEL"].values.astype(int)

    return oasis_labelled, y_oasis


def encode_gender(series):
    if series.dtype == object:
        return (
            series.astype(str)
            .str.upper()
            .str.strip()
            .map(
                {
                    "M": 0,
                    "F": 1,
                    "MALE": 0,
                    "FEMALE": 1,
                    "0": 0,
                    "1": 1,
                    "2": 1,
                }
            )
        )

    numeric_values = pd.to_numeric(series, errors="coerce")
    unique_values = set(numeric_values.dropna().unique())

    if unique_values.issubset({0, 1}):
        print("  PTGENDER: detected 0/1 encoding - kept as-is")
        return numeric_values

    if unique_values.issubset({1, 2}):
        print("  PTGENDER: detected 1/2 encoding - remapped to 0/1")
        return numeric_values.map({1: 0, 2: 1})

    print(f"  PTGENDER: warning, unexpected values {unique_values} - kept numeric")
    return numeric_values


def align_features(df, e4_features):
    """
    Align OASIS-2 columns to the ADNI E4 feature order.

    Missing features are filled with NaN so that the trained pipeline imputer
    can handle them.
    """
    column_map = {
        "M/F": "PTGENDER",
        "Gender": "PTGENDER",
        "GENDER": "PTGENDER",
        "SEX": "PTGENDER",
        "Educ": "PTEDUCAT",
        "EDUC": "PTEDUCAT",
        "MMSE": "MMSE_BL",
        "mmse": "MMSE_BL",
    }

    df = df.rename(columns=column_map).copy()

    if "PTGENDER" in df.columns:
        df["PTGENDER"] = encode_gender(df["PTGENDER"])

    availability = {}
    feature_values = {}

    for feature in e4_features:
        if feature in df.columns:
            values = pd.to_numeric(df[feature], errors="coerce").values
            valid_count = int(pd.notna(values).sum())

            availability[feature] = valid_count / len(df)
            feature_values[feature] = values

        else:
            availability[feature] = 0.0
            feature_values[feature] = np.full(len(df), np.nan)

    x_oasis = np.column_stack(
        [
            feature_values[feature]
            for feature in e4_features
        ]
    )

    return x_oasis, availability


def report_feature_availability(oasis_labelled):
    section("STEP 3 - Feature availability in OASIS-2")

    x_oasis, availability = align_features(
        oasis_labelled,
        E4_FEATURES,
    )

    print(f"\n  {'Feature':<28} {'OASIS-2 Coverage':>18}  Status")
    print("  " + "-" * 68)

    for feature in E4_FEATURES:
        coverage = availability[feature]

        if coverage > 0.10:
            status = "available"
        else:
            status = "ABSENT - median imputed"

        print(f"  {feature:<28} {coverage * 100:>16.1f}%  {status}")

    n_available = sum(
        1
        for coverage in availability.values()
        if coverage > 0.10
    )

    n_absent = len(E4_FEATURES) - n_available

    print(f"\n  Available : {n_available} / {len(E4_FEATURES)}")
    print(f"  Absent    : {n_absent} / {len(E4_FEATURES)}")
    print(f"\n  KEY FINDING: Only {n_available}/18 features available.")
    print("  This is a feature-mismatch result expected for the paper.")

    return x_oasis, availability, n_available


def load_xgboost_model():
    section("STEP 4 - Loading trained XGBoost model")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run train_clinical.py first."
        )

    pipeline = joblib.load(MODEL_PATH)

    print(f"  Loaded: {MODEL_PATH}")
    print(f"  Pipeline steps: {list(pipeline.named_steps.keys())}")

    return pipeline


def predict_oasis2(pipeline, x_oasis, y_oasis):
    section("STEP 5 - Predicting on OASIS-2")

    y_prob = pipeline.predict_proba(x_oasis)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"  Subjects      : {len(y_oasis)}")
    print(f"  pMCI-like (1) : {y_oasis.sum()}")
    print(f"  sMCI-like (0) : {(y_oasis == 0).sum()}")
    print(f"  Prevalence    : {y_oasis.mean() * 100:.1f}%")

    print(f"\n  Predicted pMCI at threshold 0.5: {y_pred.sum()}")
    print(f"  Probability range: [{y_prob.min():.3f}, {y_prob.max():.3f}]")
    print(f"  Probability mean : {y_prob.mean():.3f}")

    return y_prob, y_pred


def evaluate_oasis2(y_oasis, y_prob):
    section("STEP 6 - Evaluation metrics")

    try:
        result = calculate_metrics(y_oasis, y_prob)

        print(f"\n  {'Metric':<20} {'Value':>8}")
        print("  " + "-" * 30)

        for metric_name, value in result.items():
            if metric_name.startswith("N_"):
                print(f"  {metric_name:<20} {int(value):>8}")
            else:
                print(f"  {metric_name:<20} {value:>8.3f}")

        return result

    except Exception as error:
        print(f"  Metrics failed: {error}")
        return {}


def save_bootstrap_results(y_oasis, y_prob):
    section("STEP 7 - Bootstrap 95% CI")

    rows = []
    ci = {}

    try:
        ci = bootstrap_ci(
            y_oasis,
            y_prob,
            n_boot=1000,
        )

        print(f"\n  {'Metric':<16} {'Mean':>8}  {'95% CI':>20}")
        print("  " + "-" * 48)

        for metric_name, (mean_value, ci_low, ci_high) in ci.items():
            print(
                f"  {metric_name:<16} {mean_value:>8.3f}  "
                f"[{ci_low:.3f}, {ci_high:.3f}]"
            )

            rows.append(
                {
                    "metric": metric_name,
                    "mean": mean_value,
                    "CI_lo": ci_low,
                    "CI_hi": ci_high,
                }
            )

        print(
            f"\n  AUC-ROC = {ci['AUC_ROC'][0]:.3f} "
            f"[{ci['AUC_ROC'][1]:.3f}, {ci['AUC_ROC'][2]:.3f}]"
        )

        print("  Expected: around 0.47, essentially chance level")

    except Exception as error:
        print(f"  Bootstrap failed: {error}")

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "oasis2_bootstrap_ci.csv",
        index=False,
    )

    print("\n  Saved: oasis2_bootstrap_ci.csv")

    return ci, rows


def save_predictions(oasis_labelled, y_prob, y_pred):
    section("STEP 8 - Saving predictions")

    prediction_df = oasis_labelled.copy()
    prediction_df["y_prob"] = y_prob
    prediction_df["y_pred"] = y_pred

    prediction_df.to_csv(
        RESULTS_DIR / "oasis2_scored_predictions.csv",
        index=False,
    )

    print("  Saved: oasis2_scored_predictions.csv")


def run_threshold_analysis(y_oasis, y_prob):
    print("\n  OASIS-2 threshold analysis")
    print(f"  {'Threshold':>10} {'Sens':>7} {'Spec':>7} {'F1':>7} {'Pred+':>7}")
    print("  " + "-" * 42)

    thresholds = [
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
    ]

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_oasis,
            y_pred,
            labels=[0, 1],
        ).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = f1_score(y_oasis, y_pred, zero_division=0)

        print(
            f"  {threshold:10.2f} "
            f"{sensitivity:7.3f} "
            f"{specificity:7.3f} "
            f"{f1:7.3f} "
            f"{y_pred.sum():7d}"
        )


def save_roc_pr_plot(y_oasis, y_prob):
    section("STEP 9 - ROC and PR curves")

    try:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

        fpr, tpr, _ = roc_curve(y_oasis, y_prob)
        auc_value = roc_auc_score(y_oasis, y_prob)

        ax_roc.plot(
            fpr,
            tpr,
            color="#DC2626",
            lw=2,
            label=f"OASIS-2 AUC={auc_value:.3f}",
        )

        ax_roc.plot(
            [0, 1],
            [0, 1],
            "k--",
            lw=1,
            label="Chance",
        )

        ax_roc.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="ROC - OASIS-2 External Validation",
        )

        ax_roc.legend(fontsize=9)
        ax_roc.grid(alpha=0.3)

        precision, recall, _ = precision_recall_curve(y_oasis, y_prob)
        ap_value = average_precision_score(y_oasis, y_prob)
        baseline = y_oasis.mean()

        ax_pr.plot(
            recall,
            precision,
            color="#DC2626",
            lw=2,
            label=f"OASIS-2 AP={ap_value:.3f}",
        )

        ax_pr.axhline(
            baseline,
            color="k",
            linestyle="--",
            lw=1,
            label=f"Baseline={baseline:.2f}",
        )

        ax_pr.set(
            xlabel="Recall",
            ylabel="Precision",
            title="PR Curve - OASIS-2 External Validation",
        )

        ax_pr.legend(fontsize=9)
        ax_pr.grid(alpha=0.3)

        plt.suptitle(
            f"External Validation: ADNI XGBoost -> OASIS-2\n"
            f"n={len(y_oasis)} | pMCI-like={y_oasis.sum()} | "
            f"sMCI-like={(y_oasis == 0).sum()}",
            fontsize=11,
            fontweight="bold",
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / "oasis2_roc_pr.png",
            **PLOT_KW,
        )

        plt.close()

        print("  Saved: oasis2_roc_pr.png")

    except Exception as error:
        print(f"  Plot failed: {error}")


def save_feature_mismatch_plot(availability):
    section("STEP 10 - Feature mismatch visualisation")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        features = E4_FEATURES
        coverage_values = [
            availability[feature] * 100
            for feature in features
        ]

        colors = [
            "#16A34A" if coverage > 10 else "#DC2626"
            for coverage in coverage_values
        ]

        bars = ax.barh(
            features,
            coverage_values,
            color=colors,
            alpha=0.85,
        )

        ax.axvline(
            10,
            color="black",
            linestyle="--",
            lw=1.5,
            label="10% threshold",
        )

        ax.set_xlabel("Feature Coverage in OASIS-2 (%)")

        ax.set_title(
            "Feature Availability: ADNI E4 Features in OASIS-2\n"
            "Red = absent/imputed with ADNI median, Green = available",
            fontsize=11,
        )

        ax.legend(fontsize=9)
        ax.set_xlim(0, 110)
        ax.grid(axis="x", alpha=0.3)

        for bar, coverage in zip(bars, coverage_values):
            label = f"{coverage:.0f}%" if coverage > 1 else "0%"

            ax.text(
                max(coverage + 1, 2),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=8,
            )

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR / "oasis2_feature_mismatch.png",
            **PLOT_KW,
        )

        plt.close()

        print("  Saved: oasis2_feature_mismatch.png")

    except Exception as error:
        print(f"  Feature mismatch plot failed: {error}")


def print_final_report(y_oasis, ci, ci_rows, n_available):
    section("OASIS-2 Validation Complete")

    if ci_rows:
        auc_string = (
            f"{ci['AUC_ROC'][0]:.3f} "
            f"[{ci['AUC_ROC'][1]:.3f}, {ci['AUC_ROC'][2]:.3f}]"
        )
    else:
        auc_string = "N/A"

    print(
        f"""
  Results summary:
  ------------------------------------------------------------
  Subjects       : {len(y_oasis)}
                   pMCI-like={y_oasis.sum()}, sMCI-like={(y_oasis == 0).sum()}
  AUC-ROC        : {auc_string}
  Features used  : {n_available}/18
                   Remaining features median-imputed from ADNI

  Paper framing:
  ------------------------------------------------------------
  External validation on OASIS-2 produced near chance-level
  performance. This result reflects feature mismatch rather than
  direct model failure, because only {n_available} of the 18 ADNI
  training features are available in OASIS-2.

  Outputs:
    results/oasis2_bootstrap_ci.csv
    results/oasis2_scored_predictions.csv
    plots/oasis2_roc_pr.png
    plots/oasis2_feature_mismatch.png

  Clinical pipeline is now locked.

  Next:
    shap_phase15.py
    IEEE Access paper draft
"""
    )


def main():
    oasis_raw = load_oasis2_data()

    oasis_labelled, y_oasis = prepare_labels(oasis_raw)

    x_oasis, availability, n_available = report_feature_availability(
        oasis_labelled,
    )

    pipeline = load_xgboost_model()

    y_prob, y_pred = predict_oasis2(
        pipeline,
        x_oasis,
        y_oasis,
    )

    evaluate_oasis2(
        y_oasis,
        y_prob,
    )

    ci, ci_rows = save_bootstrap_results(
        y_oasis,
        y_prob,
    )

    save_predictions(
        oasis_labelled,
        y_prob,
        y_pred,
    )

    run_threshold_analysis(
        y_oasis,
        y_prob,
    )

    save_roc_pr_plot(
        y_oasis,
        y_prob,
    )

    save_feature_mismatch_plot(
        availability,
    )

    print_final_report(
        y_oasis,
        ci,
        ci_rows,
        n_available,
    )


if __name__ == "__main__":
    main()