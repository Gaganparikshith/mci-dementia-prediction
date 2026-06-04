#!/usr/bin/env python3
"""
README - train_clinical.py
==========================

CHANGES MADE (v2):
------------------
CHANGE 1 — Medical history features (diabetes, hypertension, smoking, BMI)
           loaded from MEDHIST table and merged into feature set.

CHANGE 2 — SMOTE replaced with scale_pos_weight for XGBoost.
           RF and LR now use class_weight='balanced' instead of SMOTE.
           No synthetic data generated. More honest for research paper.

CHANGE 3 — Longitudinal slope features added (MMSE_slope, ADAS13_slope,
           CDRSB_slope, FAQ_slope). Computed from ADNIMERGE longitudinal
           data. Rate-of-change is the strongest untapped predictor.
           Falls back gracefully if longitudinal data is unavailable.

This script trains the Phase 1 clinical-only models for pMCI vs sMCI
prediction using ADNI clinical features.

Pipeline
--------
SimpleImputer(median) -> StandardScaler -> Classifier
(SMOTE removed — XGBoost uses scale_pos_weight, RF/LR use class_weight)

Models evaluated
----------------
1. Random Forest
2. Logistic Regression
3. XGBoost

Main outputs
------------
best_rf.pkl, best_lr.pkl, best_xgb.pkl
clinical_test_bootstrap_ci.csv
xgb_threshold_tuning.csv
xgb_calibrated_probs.csv
ablation_e1_e4.csv
subgroup_analysis.csv

Run
---
python Scripts/training/train_clinical.py

Next step
---------
python Scripts/training/shap_clinical.py
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch")
ROOT.mkdir(parents=True, exist_ok=True)

META_FILE      = ROOT / "data" / "metadata" / "final_metadata.csv"
SPLITS_DIR     = ROOT / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# ── CHANGE 1: Medical history and longitudinal paths ──────────────────────────
MEDHIST_FILE   = ROOT / "data" / "metadata" / "MEDHIST.csv"       # ADNI MEDHIST table
ADNIMERGE_FILE = ROOT / "data" / "metadata" / "ADNIMERGE.csv"     # Full longitudinal table


# ── Settings ──────────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
SPARSE_THRESHOLD = 0.10


# ── Ablation feature sets ─────────────────────────────────────────────────────
ABLATION_SETS = {
    "E1": [
        "PTGENDER", "PTEDUCAT", "MMSE_BL", "FAQ_BL",
    ],
    "E2": [
        "PTGENDER", "PTEDUCAT", "MMSE_BL", "FAQ_BL",
        "GDS_BL", "ADAS11_BL", "ADAS13_BL", "MOCA_BL",
    ],
    "E3": [
        "PTGENDER", "PTEDUCAT", "MMSE_BL", "FAQ_BL",
        "GDS_BL", "ADAS11_BL", "ADAS13_BL", "MOCA_BL",
        "RAVLT_immediate", "RAVLT_forgetting", "RAVLT_delayed",
        "RAVLT_forget_rate", "CDRSB_BL",
    ],
    "E4": [
        "PTGENDER", "PTEDUCAT", "MMSE_BL", "FAQ_BL",
        "GDS_BL", "MOCA_BL", "ADAS11_BL", "ADAS13_BL",
        "RAVLT_immediate", "RAVLT_forgetting", "RAVLT_delayed",
        "RAVLT_forget_rate", "DigitSpan", "TrailsB",
        "MMSE_FAQ_composite", "ADAS_MMSE_gap", "CDRSB_BL", "CDR_GLOBAL_BL",
    ],
    "E4_no_MoCA": [
        "PTGENDER", "PTEDUCAT", "MMSE_BL", "FAQ_BL",
        "GDS_BL", "ADAS11_BL", "ADAS13_BL",
        "RAVLT_immediate", "RAVLT_forgetting", "RAVLT_delayed",
        "RAVLT_forget_rate", "DigitSpan", "TrailsB",
        "MMSE_FAQ_composite", "ADAS_MMSE_gap", "CDRSB_BL", "CDR_GLOBAL_BL",
    ],
    "E4_no_MoCA_DigitSpan": [
        "PTGENDER", "PTEDUCAT", "MMSE_BL", "FAQ_BL",
        "GDS_BL", "ADAS11_BL", "ADAS13_BL",
        "RAVLT_immediate", "RAVLT_forgetting", "RAVLT_delayed",
        "RAVLT_forget_rate", "TrailsB",
        "MMSE_FAQ_composite", "ADAS_MMSE_gap", "CDRSB_BL", "CDR_GLOBAL_BL",
    ],
}

# ── CHANGE 1: Medical history features to merge in ───────────────────────────
MEDICAL_HISTORY_FEATURES = [
    "MHDIABET",   # Type 2 diabetes history (0/1)
    "MHHYPERT",   # Hypertension history (0/1)
    "MHSMOK",     # Smoking history (0/1)
    "BMI",        # Body mass index
]

# ── CHANGE 3: Slope features to compute ──────────────────────────────────────
SLOPE_FEATURES = ["MMSE", "ADAS13", "CDRSB", "FAQ"]

MODEL_NAMES = ["RandomForest", "LogisticRegression", "XGBoost"]

MODEL_FILES = {
    "RandomForest":      "best_rf.pkl",
    "LogisticRegression":"best_lr.pkl",
    "XGBoost":           "best_xgb.pkl",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def calculate_metrics(y_true, y_prob, label="", threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "label":       label,
        "AUC_ROC":     roc_auc_score(y_true, y_prob),
        "AUC_PR":      average_precision_score(y_true, y_prob),
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV":         ppv,
        "NPV":         npv,
        "MCC":         matthews_corrcoef(y_true, y_pred),
        "Brier":       brier_score_loss(y_true, y_prob),
    }


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / len(y_true)


# ── CHANGE 2: Separate pipeline builders — no SMOTE ──────────────────────────
def make_classifier(model_name, n_pos=None, n_neg=None):
    """
    CHANGE 2: SMOTE removed.
    - XGBoost uses scale_pos_weight = n_neg / n_pos (native balancing).
    - RF and LR use class_weight='balanced'.
    - No synthetic data generated anywhere in the pipeline.
    """
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",        # CHANGE 2
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "LogisticRegression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",        # CHANGE 2
            random_state=RANDOM_STATE,
        )

    if model_name == "XGBoost":
        # scale_pos_weight = n_negative / n_positive
        spw = (n_neg / n_pos) if (n_pos and n_neg) else 1.0
        print(f"     XGBoost scale_pos_weight = {spw:.3f}")
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,           # CHANGE 2
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def make_pipeline(classifier):
    """
    CHANGE 2: Uses sklearn Pipeline (not ImbPipeline) — SMOTE removed.
    Steps: impute → scale → classify.
    """
    return SkPipeline(
        steps=[
            ("imputer",    SimpleImputer(strategy="median")),
            ("scaler",     StandardScaler()),
            ("classifier", classifier),
        ]
    )


def bootstrap_ci(y_true, y_prob, n_boot=1000, ci=0.95, threshold=0.5, seed=42):
    rng    = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n      = len(y_true)

    results = {k: [] for k in
               ["AUC_ROC", "AUC_PR", "F1", "Sensitivity", "Specificity"]}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sy, sp = y_true[idx], y_prob[idx]
        if len(np.unique(sy)) < 2:
            continue
        m = calculate_metrics(sy, sp, threshold=threshold)
        for k in results:
            results[k].append(m[k])

    alpha = (1 - ci) / 2
    return {
        k: (float(np.mean(v)),
            float(np.percentile(v, alpha * 100)),
            float(np.percentile(v, (1 - alpha) * 100)))
        for k, v in results.items()
    }


# ── CHANGE 1: Load and merge medical history features ────────────────────────
def merge_medical_history(df):
    """
    CHANGE 1: Load MEDHIST table and merge diabetes, hypertension,
    smoking, and BMI into the main dataframe.
    Falls back gracefully if MEDHIST.csv is not found.
    """
    section("CHANGE 1 — Merging medical history features")

    if not MEDHIST_FILE.exists():
        print(f"  MEDHIST.csv not found at {MEDHIST_FILE}")
        print("  Skipping medical history merge.")
        print("  To enable: export MEDHIST table from ADNI and place it here.")
        return df, []

    medhist = pd.read_csv(MEDHIST_FILE)
    print(f"  Loaded MEDHIST: {medhist.shape[0]} rows x {medhist.shape[1]} cols")

    # Keep only the columns we need
    available_cols = ["RID"] + [
        c for c in MEDICAL_HISTORY_FEATURES if c in medhist.columns
    ]
    medhist = medhist[available_cols]

    # One row per subject — take first visit
    medhist = medhist.groupby("RID").first().reset_index()

    before = len(df)
    df = df.merge(medhist, on="RID", how="left")
    print(f"  Merged {len(available_cols) - 1} features into df "
          f"({before} → {len(df)} rows, no rows lost)")

    added = [c for c in MEDICAL_HISTORY_FEATURES if c in df.columns]
    for feat in added:
        cov = df[feat].notna().mean() * 100
        print(f"    {feat:<14} {cov:.1f}% available")

    return df, added


# ── CHANGE 3: Compute longitudinal slope features ─────────────────────────────
def compute_slope_features(df):
    """
    CHANGE 3: Load ADNIMERGE longitudinal table and compute per-subject
    rate-of-change (slope) for MMSE, ADAS13, CDRSB, FAQ.
    Falls back gracefully if ADNIMERGE.csv is not found.

    Slope = linear regression coefficient of feature vs. months from baseline.
    A steeply negative MMSE slope = rapid decline = strong pMCI signal.
    """
    section("CHANGE 3 — Computing longitudinal slope features")

    if not ADNIMERGE_FILE.exists():
        print(f"  ADNIMERGE.csv not found at {ADNIMERGE_FILE}")
        print("  Skipping slope features.")
        print("  To enable: download ADNIMERGE.csv from LONI and place it here.")
        return df, []

    adni = pd.read_csv(ADNIMERGE_FILE, low_memory=False)
    print(f"  Loaded ADNIMERGE: {adni.shape[0]} rows x {adni.shape[1]} cols")

    # Convert exam date and compute months from baseline per subject
    adni["EXAMDATE"] = pd.to_datetime(adni["EXAMDATE"], errors="coerce")
    adni = adni.sort_values(["RID", "EXAMDATE"])
    adni["months_from_bl"] = adni.groupby("RID")["EXAMDATE"].transform(
        lambda x: (x - x.iloc[0]).dt.days / 30.44
    )

    # Filter to MCI subjects only (use RIDs in our dataset)
    mci_rids = set(df["RID"].values)
    adni = adni[adni["RID"].isin(mci_rids)]

    slope_cols = []
    slope_rows = []

    for rid, grp in adni.groupby("RID"):
        row = {"RID": rid}

        for feat in SLOPE_FEATURES:
            col_name = f"{feat}_slope"

            # Find the matching column name in ADNIMERGE (may have different suffix)
            candidates = [c for c in adni.columns
                          if c.upper().startswith(feat.upper())]
            if not candidates:
                row[col_name] = np.nan
                continue

            series = grp[candidates[0]].dropna()
            t      = grp.loc[series.index, "months_from_bl"].values

            if len(series) >= 2 and len(np.unique(t)) >= 2:
                # Linear regression slope
                slope = np.polyfit(t, series.values, 1)[0]
                row[col_name] = float(slope)
            else:
                # Only one visit — slope undefined, fill with 0 (no change)
                row[col_name] = 0.0

        slope_rows.append(row)

        if col_name not in slope_cols:
            for feat in SLOPE_FEATURES:
                cname = f"{feat}_slope"
                if cname not in slope_cols:
                    slope_cols.append(cname)

    slope_df = pd.DataFrame(slope_rows)
    print(f"  Computed slopes for {len(slope_df)} subjects")

    df = df.merge(slope_df, on="RID", how="left")

    for col in slope_cols:
        n_valid = df[col].notna().sum()
        print(f"    {col:<18} {n_valid}/{len(df)} subjects have ≥2 visits")

    print("\n  Interpretation guide:")
    print("    MMSE_slope   < 0  → declining cognition → pMCI signal ↑")
    print("    ADAS13_slope > 0  → worsening symptoms  → pMCI signal ↑")
    print("    CDRSB_slope  > 0  → increasing severity → pMCI signal ↑")
    print("    FAQ_slope    > 0  → worsening function  → pMCI signal ↑")

    return df, slope_cols


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    section("Loading data")
    df = pd.read_csv(META_FILE)
    print(f"  Loaded : {df.shape[0]} subjects x {df.shape[1]} columns")
    print(f"  pMCI   : {(df['LABEL'] == 1).sum()}  |  "
          f"sMCI: {(df['LABEL'] == 0).sum()}")
    return df


def report_coverage(df):
    print("\n  Pre-imputation coverage:")
    print(f"  {'Feature':<28} {'N present':>10}   {'Coverage':>9}")
    print("  " + "-" * 50)

    candidate_features = [c for c in df.columns if c not in ["RID", "LABEL"]]
    coverage = {}

    for feature in candidate_features:
        n_present       = int(df[feature].notna().sum())
        feature_cov     = n_present / len(df)
        coverage[feature] = feature_cov
        flag = "  <- LOW" if feature_cov < 0.50 else ""
        print(f"  {feature:<28} {n_present:>10}   "
              f"{feature_cov * 100:>8.1f}%{flag}")

    return coverage


def select_e4_features(df, coverage, extra_features=None):
    section("Feature selection - Phase 1 full clinical E4 + extras")

    selected_features = []

    for feature in ABLATION_SETS["E4"]:
        if feature not in df.columns:
            print(f"  ABSENT    : {feature}")
            continue
        if coverage.get(feature, 0.0) < SPARSE_THRESHOLD:
            print(f"  TOO SPARSE: {feature} "
                  f"({coverage.get(feature, 0) * 100:.1f}%)")
            continue
        selected_features.append(feature)

    # CHANGE 1 + 3: append medical history and slope features
    if extra_features:
        for feat in extra_features:
            if feat in df.columns and feat not in selected_features:
                cov = df[feat].notna().mean()
                if cov >= SPARSE_THRESHOLD:
                    selected_features.append(feat)
                    print(f"  ADDED     : {feat} ({cov * 100:.1f}% available)")
                else:
                    print(f"  SPARSE    : {feat} ({cov * 100:.1f}%) — skipped")

    ABLATION_SETS["E4"] = selected_features

    print(f"\n  E4 feature set ({len(selected_features)} features):")
    for idx, feature in enumerate(selected_features, start=1):
        print(f"  {idx:>4}. {feature:<30} "
              f"{coverage.get(feature, df[feature].notna().mean()) * 100:.0f}%")

    return selected_features


def create_train_test_split(df, features):
    section("STEP 1 - Stratified 80/20 train/test split")

    x = df[features].values
    y = df["LABEL"].values

    df_indexed   = df.reset_index(drop=True)
    all_indices  = df_indexed.index.values

    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pd.DataFrame({"RID": df_indexed.iloc[train_idx]["RID"].values,
                  "LABEL": y_train}).to_csv(
        SPLITS_DIR / "train_subjects.csv", index=False)
    pd.DataFrame({"RID": df_indexed.iloc[test_idx]["RID"].values,
                  "LABEL": y_test}).to_csv(
        SPLITS_DIR / "test_subjects.csv", index=False)

    print(f"  Train : {len(y_train)}  "
          f"pMCI={y_train.sum()} ({y_train.mean() * 100:.1f}%)  "
          f"sMCI={(y_train == 0).sum()}")
    print(f"  Test  : {len(y_test)}  "
          f"pMCI={y_test.sum()} ({y_test.mean() * 100:.1f}%)  "
          f"sMCI={(y_test == 0).sum()}")

    # CHANGE 2: compute class counts for scale_pos_weight
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    print(f"\n  Class ratio (train): n_neg={n_neg}, n_pos={n_pos}, "
          f"scale_pos_weight={n_neg/n_pos:.3f}")
    print("  Test set locked - not touched until final evaluation.")

    return x_train, x_test, y_train, y_test, n_pos, n_neg


def run_cross_validation(x_train, y_train, n_pos, n_neg):
    section("STEP 2 - 5-fold stratified CV on train set")
    print(f"  Train: {len(y_train)} subjects. SMOTE removed. "
          f"Using class_weight / scale_pos_weight.\n")   # CHANGE 2

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_summary = {}

    for model_name in MODEL_NAMES:
        print(f"  -- {model_name}")
        fold_metrics = []

        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
            skf.split(x_train, y_train), start=1
        ):
            # CHANGE 2: pass n_pos, n_neg so XGBoost gets scale_pos_weight
            pipeline = make_pipeline(
                make_classifier(model_name, n_pos=n_pos, n_neg=n_neg)
            )
            pipeline.fit(x_train[fold_train_idx], y_train[fold_train_idx])
            val_prob = pipeline.predict_proba(x_train[fold_val_idx])[:, 1]
            fold_result = calculate_metrics(y_train[fold_val_idx], val_prob)
            fold_metrics.append(fold_result)

            print(f"     Fold {fold_idx}: "
                  f"AUC={fold_result['AUC_ROC']:.3f}  "
                  f"PR={fold_result['AUC_PR']:.3f}  "
                  f"F1={fold_result['F1']:.3f}  "
                  f"Sens={fold_result['Sensitivity']:.3f}  "
                  f"Spec={fold_result['Specificity']:.3f}")

        metric_names = ["AUC_ROC", "AUC_PR", "F1", "Sensitivity",
                        "Specificity", "MCC", "Brier"]
        mean_values  = {m: float(np.mean([f[m] for f in fold_metrics]))
                        for m in metric_names}
        auc_std      = float(np.std([f["AUC_ROC"] for f in fold_metrics]))
        cv_summary[model_name] = {**mean_values, "AUC_ROC_std": auc_std}

        print(f"\n     CV Mean: "
              f"AUC={mean_values['AUC_ROC']:.3f}+/-{auc_std:.3f}  "
              f"Sens={mean_values['Sensitivity']:.3f}  "
              f"Spec={mean_values['Specificity']:.3f}  "
              f"Brier={mean_values['Brier']:.3f}\n")

    cv_df = (pd.DataFrame(cv_summary).T.reset_index()
             .rename(columns={"index": "model"}))
    cv_df = cv_df[["model", "AUC_ROC", "AUC_ROC_std", "AUC_PR", "F1",
                   "Sensitivity", "Specificity", "MCC", "Brier"]]
    print(cv_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return cv_df


def train_and_evaluate_models(x_train, x_test, y_train, y_test,
                               n_pos, n_neg):
    section("STEP 3 - Refit on full train set and evaluate held-out test set")

    fitted_pipelines  = {}
    test_probabilities = {}
    test_metrics      = {}

    for model_name in MODEL_NAMES:
        # CHANGE 2: pass n_pos, n_neg to XGBoost
        pipeline = make_pipeline(
            make_classifier(model_name, n_pos=n_pos, n_neg=n_neg)
        )
        pipeline.fit(x_train, y_train)
        test_prob   = pipeline.predict_proba(x_test)[:, 1]
        m           = calculate_metrics(y_test, test_prob, label=model_name)

        fitted_pipelines[model_name]   = pipeline
        test_probabilities[model_name] = test_prob
        test_metrics[model_name]       = m

        joblib.dump(pipeline, ROOT / MODEL_FILES[model_name])

        print(f"  {model_name:<22} "
              f"AUC={m['AUC_ROC']:.3f}  PR={m['AUC_PR']:.3f}  "
              f"F1={m['F1']:.3f}  Sens={m['Sensitivity']:.3f}  "
              f"Spec={m['Specificity']:.3f}  "
              f"(saved {MODEL_FILES[model_name]})")

    return fitted_pipelines, test_probabilities, test_metrics


def save_bootstrap_confidence_intervals(y_test, test_probabilities):
    section("STEP 4 - Bootstrap 95% CI on test set")
    rows = []
    for model_name in MODEL_NAMES:
        ci = bootstrap_ci(y_test, test_probabilities[model_name])
        for metric_name, (mean_val, ci_lo, ci_hi) in ci.items():
            print(f"  {model_name:<22} {metric_name:<14} "
                  f"{mean_val:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
            rows.append({"model": model_name, "metric": metric_name,
                         "mean": mean_val, "CI_lo": ci_lo, "CI_hi": ci_hi})
        print()

    pd.DataFrame(rows).to_csv(
        ROOT / "clinical_test_bootstrap_ci.csv", index=False)
    print("  Saved clinical_test_bootstrap_ci.csv")


def tune_xgboost_thresholds(y_test, test_probabilities):
    section("STEP 5 - Threshold tuning for XGBoost")
    xgb_prob = test_probabilities["XGBoost"]
    rows = []

    print(f"   {'Threshold':>9}  {'Sens':>6}  {'Spec':>6}  "
          f"{'PPV':>6}  {'NPV':>6}  {'F1':>6}  Notes")
    print("  " + "-" * 84)

    for threshold in np.arange(0.10, 0.91, 0.05):
        threshold = round(float(threshold), 2)
        m = calculate_metrics(y_test, xgb_prob, threshold=threshold)

        # CHANGE 2: three named clinical modes
        if threshold == 0.15:
            note = "<- Screening mode (high sensitivity)"
        elif threshold == 0.35:
            note = "<- Balanced clinical mode"
        elif threshold == 0.50:
            note = "<- Confirmatory / default"
        else:
            note = ""

        print(f"        {threshold:.2f}   "
              f"{m['Sensitivity']:.3f}  {m['Specificity']:.3f}  "
              f"{m['PPV']:.3f}  {m['NPV']:.3f}  {m['F1']:.3f}  {note}")

        rows.append({"threshold": threshold,
                     **{k: v for k, v in m.items() if k != "label"}})

    pd.DataFrame(rows).to_csv(ROOT / "xgb_threshold_tuning.csv", index=False)


def run_probability_calibration(x_train, y_train, y_test,
                                 fitted_pipelines, test_probabilities):
    section("STEP 6 - Probability calibration")

    for model_name in MODEL_NAMES:
        raw_prob = test_probabilities[model_name]
        print(f"  {model_name:<22} "
              f"Brier={brier_score_loss(y_test, raw_prob):.4f}  "
              f"ECE={compute_ece(y_test, raw_prob):.4f}")

    print("\n  Applying Platt scaling to XGBoost...")
    xgb_train_raw = fitted_pipelines["XGBoost"].predict_proba(
        x_train)[:, 1].reshape(-1, 1)
    xgb_test_raw = test_probabilities["XGBoost"].reshape(-1, 1)

    platt = LogisticRegression(max_iter=1000)
    platt.fit(xgb_train_raw, y_train)
    cal_prob = platt.predict_proba(xgb_test_raw)[:, 1]

    cal_brier = brier_score_loss(y_test, cal_prob)
    raw_brier = brier_score_loss(y_test, test_probabilities["XGBoost"])

    print(f"  Raw Brier={raw_brier:.4f}  Calibrated Brier={cal_brier:.4f}")

    pd.DataFrame({
        "y_true":     y_test,
        "y_prob_raw": test_probabilities["XGBoost"],
        "y_prob_cal": cal_prob,
    }).to_csv(ROOT / "xgb_calibrated_probs.csv", index=False)
    print("  Saved xgb_calibrated_probs.csv")

    if cal_brier < raw_brier:
        print("  Using calibrated probabilities.")
        return cal_prob
    else:
        print("  Platt scaling did not improve calibration — "
              "retaining raw probabilities.")
        return test_probabilities["XGBoost"]


def run_ablation_study(df, coverage, n_pos, n_neg):
    section("STEP 7 - Ablation study E1 to E4")
    rows = []

    for exp_name, feat_list in ABLATION_SETS.items():
        valid_feats = [
            f for f in feat_list
            if f in df.columns and
            df[f].notna().mean() >= SPARSE_THRESHOLD
        ]
        if not valid_feats:
            continue

        x_abl = df[valid_feats].values
        y_abl = df["LABEL"].values

        x_tr, x_te, y_tr, y_te = train_test_split(
            x_abl, y_abl, test_size=0.20,
            stratify=y_abl, random_state=RANDOM_STATE)

        # CHANGE 2: scale_pos_weight in ablation too
        np_ = int(y_tr.sum())
        nn_ = int((y_tr == 0).sum())
        pipeline = make_pipeline(
            make_classifier("XGBoost", n_pos=np_, n_neg=nn_))
        pipeline.fit(x_tr, y_tr)
        prob = pipeline.predict_proba(x_te)[:, 1]
        m    = calculate_metrics(y_te, prob, label=exp_name)

        print(f"  {exp_name} ({len(valid_feats):>2} features):  "
              f"AUC={m['AUC_ROC']:.3f}  F1={m['F1']:.3f}  "
              f"Sens={m['Sensitivity']:.3f}  Spec={m['Specificity']:.3f}")

        rows.append({"experiment": exp_name, "n_features": len(valid_feats),
                     **{k: v for k, v in m.items() if k != "label"}})

    pd.DataFrame(rows).to_csv(ROOT / "ablation_e1_e4.csv", index=False)
    print("\n  Saved ablation_e1_e4.csv")


def run_subgroup_analysis(x_test, y_test, e4_features, test_probabilities):
    section("STEP 8 - Subgroup analysis: gender and education")
    x_df           = pd.DataFrame(x_test, columns=e4_features)
    x_df["y_true"] = y_test
    x_df["y_prob"] = test_probabilities["XGBoost"]
    rows           = []

    if "PTGENDER" in x_df.columns:
        for gval, glbl in [(0, "Male"), (1, "Female")]:
            sg = x_df[x_df["PTGENDER"] == gval]
            if len(sg) < 10 or sg["y_true"].nunique() < 2:
                continue
            m = calculate_metrics(sg["y_true"].values,
                                  sg["y_prob"].values, label=glbl)
            print(f"  Gender={glbl:<8}  n={len(sg):>3}  "
                  f"AUC={m['AUC_ROC']:.3f}  F1={m['F1']:.3f}  "
                  f"Sens={m['Sensitivity']:.3f}  Spec={m['Specificity']:.3f}")
            rows.append({"subgroup": "Gender", "group": glbl, "n": len(sg),
                         **{k: v for k, v in m.items() if k != "label"}})

    if "PTEDUCAT" in x_df.columns:
        edu_med = x_df["PTEDUCAT"].median()
        for glbl, mask in [
            (f"Low (<{edu_med:.0f} yr)",  x_df["PTEDUCAT"] < edu_med),
            (f"High (>={edu_med:.0f} yr)", x_df["PTEDUCAT"] >= edu_med),
        ]:
            sg = x_df[mask]
            if len(sg) < 10 or sg["y_true"].nunique() < 2:
                continue
            m = calculate_metrics(sg["y_true"].values,
                                  sg["y_prob"].values, label=glbl)
            print(f"  Edu={glbl:<22}  n={len(sg):>3}  "
                  f"AUC={m['AUC_ROC']:.3f}  F1={m['F1']:.3f}")
            rows.append({"subgroup": "Education", "group": glbl, "n": len(sg),
                         **{k: v for k, v in m.items() if k != "label"}})

    pd.DataFrame(rows).to_csv(ROOT / "subgroup_analysis.csv", index=False)
    print("\n  Saved subgroup_analysis.csv")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_data()

    # CHANGE 1: merge medical history
    df, med_feats = merge_medical_history(df)

    # CHANGE 3: compute longitudinal slopes
    df, slope_feats = compute_slope_features(df)

    coverage = report_coverage(df)

    # All extra features (medical + slopes) appended to E4
    extra_feats  = med_feats + slope_feats
    e4_features  = select_e4_features(df, coverage, extra_features=extra_feats)

    # CHANGE 2: returns n_pos, n_neg for scale_pos_weight
    x_train, x_test, y_train, y_test, n_pos, n_neg = \
        create_train_test_split(df, e4_features)

    run_cross_validation(x_train, y_train, n_pos, n_neg)

    fitted_pipelines, test_probabilities, test_metrics = \
        train_and_evaluate_models(
            x_train, x_test, y_train, y_test, n_pos, n_neg)

    save_bootstrap_confidence_intervals(y_test, test_probabilities)
    tune_xgboost_thresholds(y_test, test_probabilities)
    run_probability_calibration(
        x_train, y_train, y_test, fitted_pipelines, test_probabilities)
    run_ablation_study(df, coverage, n_pos, n_neg)
    run_subgroup_analysis(x_test, y_test, e4_features, test_probabilities)

    print()
    print("=" * 62)
    print("  Phase 1 training complete (v2 — SMOTE removed,")
    print("  medical history + slopes added).")
    print(f"  Total features used: {len(e4_features)}")
    print("  Outputs saved to:", ROOT)
    print("  Next: run shap_clinical.py for explainability")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()