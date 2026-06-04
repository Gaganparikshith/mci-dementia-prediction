#!/usr/bin/env python3
"""
README - shap_clinical.py
=========================

This script generates SHAP explainability outputs for the Phase 1
clinical-only models.

It loads the trained XGBoost and Random Forest models from Phase 1 training,
recreates the same 80/20 train-test split used in train_clinical.py, and
generates SHAP plots and CSV files for model interpretation.

Main purpose
------------
The goal of this script is to explain which clinical features contributed most
to pMCI vs sMCI prediction.

Models explained
----------------
1. XGBoost
2. Random Forest

Main outputs
------------
SHAP bar plots
    Mean absolute SHAP importance for each model.

SHAP beeswarm plots
    Direction and magnitude of feature effects.

SHAP waterfall plots
    Individual patient-level explanations for selected cases.

SHAP force plots
    Local explanations for one high-risk pMCI case and one high-confidence
    sMCI case.

SHAP dependence plots
    Feature-level interaction/effect plots for the top XGBoost features.

CSV outputs
-----------
shap_importance_xgboost.csv
shap_importance_randomforest.csv
shap_values_xgboost.csv
shap_values_randomforest.csv
shap_comparison.csv

Run
---
python shap_clinical.py

Run this script only after train_clinical.py has completed successfully.
"""

import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")
matplotlib.use("Agg")


# Paths
BASE_DIR = Path(r"C:\Users\ASUS\Desktop\Research Resources\DementiaResearch")

META_FILE = BASE_DIR / "data" / "metadata" / "final_metadata.csv"
MODELS_DIR = BASE_DIR
PLOTS_DIR = BASE_DIR / "plots" / "shap"
RESULTS_DIR = BASE_DIR / "results"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Settings
RANDOM_STATE = 42
PLOT_KW = {
    "dpi": 150,
    "bbox_inches": "tight",
}


# E4 feature set.
# This must match the feature set used in train_clinical.py.
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


# Human-readable names for plots
FEATURE_LABELS = {
    "PTGENDER": "Gender",
    "PTEDUCAT": "Education (yrs)",
    "MMSE_BL": "MMSE",
    "FAQ_BL": "FAQ",
    "GDS_BL": "GDS",
    "MOCA_BL": "MoCA",
    "ADAS11_BL": "ADAS-Cog 11",
    "ADAS13_BL": "ADAS-Cog 13",
    "RAVLT_immediate": "RAVLT Immediate",
    "RAVLT_forgetting": "RAVLT Forgetting",
    "RAVLT_delayed": "RAVLT Delayed",
    "RAVLT_forget_rate": "RAVLT Forget Rate",
    "DigitSpan": "Digit Span",
    "TrailsB": "Trails B",
    "MMSE_FAQ_composite": "MMSE-FAQ Composite",
    "ADAS_MMSE_gap": "ADAS-MMSE Gap",
    "CDRSB_BL": "CDR Sum of Boxes",
    "CDR_GLOBAL_BL": "CDR Global",
}


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    }
)


def section(title):
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def safe_shap_values(shap_values):
    """
    Convert SHAP output into a clean 2D array.

    Different model types and SHAP versions can return values in different
    formats. This helper keeps only the positive-class SHAP values.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        return shap_array[:, :, 1]

    return shap_array


def readable_labels(feature_list):
    """Convert raw feature names into readable plot labels."""
    return [FEATURE_LABELS.get(feature, feature) for feature in feature_list]


def get_classifier_from_pipeline(pipeline):
    """
    Get the classifier step from the saved pipeline.

    Older scripts used the step name 'clf'. Some cleaned versions may use
    'classifier'. This function supports both.
    """
    if "clf" in pipeline.named_steps:
        return pipeline.named_steps["clf"]

    if "classifier" in pipeline.named_steps:
        return pipeline.named_steps["classifier"]

    raise KeyError("No classifier step found in the pipeline.")


def preprocess_for_shap(pipeline, x_data):
    """
    Apply the fitted imputer and scaler from the saved pipeline.

    SMOTE is intentionally skipped because it is only used during training.
    """
    x_imputed = pipeline.named_steps["imputer"].transform(x_data)
    x_scaled = pipeline.named_steps["scaler"].transform(x_imputed)

    return x_scaled


def load_data():
    section("Loading data and reproducing train/test split")

    df = pd.read_csv(META_FILE)

    print(f"  Loaded : {df.shape[0]} subjects x {df.shape[1]} columns")

    valid_features = [
        feature
        for feature in E4_FEATURES
        if feature in df.columns
    ]

    missing_features = [
        feature
        for feature in E4_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        print(f"  WARNING - missing features: {missing_features}")

    x_data = df[valid_features].values
    y_data = df["LABEL"].values

    _, x_test, _, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.20,
        stratify=y_data,
        random_state=RANDOM_STATE,
    )

    feature_labels = readable_labels(valid_features)

    print(f"  Test set : {len(y_test)} subjects")
    print(f"  Features : {len(valid_features)}")

    return x_test, y_test, valid_features, feature_labels


def load_models():
    section("Loading trained models")

    xgb_pipeline = None
    rf_pipeline = None

    model_files = [
        ("best_xgb.pkl", "XGBoost"),
        ("best_rf.pkl", "RandomForest"),
    ]

    for file_name, model_label in model_files:
        model_path = MODELS_DIR / file_name

        if model_path.exists():
            if model_label == "XGBoost":
                xgb_pipeline = joblib.load(model_path)
            else:
                rf_pipeline = joblib.load(model_path)

            print(f"  Loaded: {file_name}")

        else:
            print(f"  Not found: {file_name}  (run train_clinical.py first)")

    if xgb_pipeline is None and rf_pipeline is None:
        raise FileNotFoundError("No models found. Run train_clinical.py first.")

    return xgb_pipeline, rf_pipeline


def build_shap_explanation(pipeline, x_processed, feature_labels):
    classifier = get_classifier_from_pipeline(pipeline)

    explainer = shap.TreeExplainer(classifier)
    raw_shap_values = explainer.shap_values(x_processed)
    shap_values = safe_shap_values(raw_shap_values)

    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(shap_values), expected_value),
        data=x_processed,
        feature_names=feature_labels,
    )

    return explainer, shap_values, expected_value, shap_explanation


def save_shap_importance_plots(
    xgb_pipeline,
    rf_pipeline,
    x_test_processed_xgb,
    x_test_processed_rf,
    valid_features,
    feature_labels,
):
    section("PLOT 1 - SHAP bar charts")

    shap_rows = []

    model_items = [
        (xgb_pipeline, x_test_processed_xgb, "XGBoost", "#2563EB"),
        (rf_pipeline, x_test_processed_rf, "RandomForest", "#16A34A"),
    ]

    for pipeline, x_processed, model_tag, color in model_items:
        if pipeline is None or x_processed is None:
            continue

        _, shap_values, _, _ = build_shap_explanation(
            pipeline,
            x_processed,
            feature_labels,
        )

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs_shap)[::-1]

        importance_df = pd.DataFrame(
            {
                "feature": [valid_features[index] for index in order],
                "label": [feature_labels[index] for index in order],
                "mean_abs_shap": mean_abs_shap[order],
            }
        )

        importance_df.to_csv(
            RESULTS_DIR / f"shap_importance_{model_tag.lower()}.csv",
            index=False,
        )

        shap_values_df = pd.DataFrame(
            shap_values,
            columns=feature_labels,
        )

        shap_values_df.to_csv(
            RESULTS_DIR / f"shap_values_{model_tag.lower()}.csv",
            index=False,
        )

        top_n = min(15, len(valid_features))
        top_indices = order[:top_n]

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.barh(
            [feature_labels[index] for index in reversed(top_indices)],
            [mean_abs_shap[index] for index in reversed(top_indices)],
            color=color,
            alpha=0.88,
        )

        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Feature Importance - {model_tag} SHAP")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        out_path = PLOTS_DIR / f"shap_bar_{model_tag.lower()}.png"
        plt.savefig(out_path, **PLOT_KW)
        plt.close()

        print(f"  Saved: {out_path.name}")

        print(f"\n  Top 10 SHAP features ({model_tag}):")

        for rank, index in enumerate(order[:10], start=1):
            print(f"    {rank:>2}. {feature_labels[index]:<28} {mean_abs_shap[index]:.4f}")

            shap_rows.append(
                {
                    "model": model_tag,
                    "rank": rank,
                    "feature": valid_features[index],
                    "mean_abs_shap": mean_abs_shap[index],
                }
            )

    pd.DataFrame(shap_rows).to_csv(
        RESULTS_DIR / "shap_comparison.csv",
        index=False,
    )

    print("\n  Saved: shap_comparison.csv")


def save_beeswarm_plot(shap_explanation, model_name, output_name):
    plt.figure(figsize=(10, 7))

    shap.plots.beeswarm(
        shap_explanation,
        max_display=15,
        show=False,
    )

    plt.title(f"SHAP Beeswarm - {model_name} (pMCI vs sMCI)", pad=14)
    plt.tight_layout()

    out_path = PLOTS_DIR / output_name

    plt.savefig(out_path, **PLOT_KW)
    plt.close()

    print(f"  Saved: {out_path.name}")


def save_xgb_beeswarm(shap_explanation_xgb):
    section("PLOT 2 - SHAP beeswarm for XGBoost")

    if shap_explanation_xgb is not None:
        save_beeswarm_plot(
            shap_explanation_xgb,
            "XGBoost",
            "shap_xgb_beeswarm.png",
        )


def save_rf_beeswarm(shap_explanation_rf):
    section("PLOT 3 - SHAP beeswarm for RandomForest")

    if shap_explanation_rf is not None:
        save_beeswarm_plot(
            shap_explanation_rf,
            "RandomForest",
            "shap_rf_beeswarm.png",
        )


def select_waterfall_cases(y_test, y_prob_xgb):
    pmci_mask = y_test == 1
    smci_mask = y_test == 0

    pmci_probabilities = np.where(pmci_mask, y_prob_xgb, -1)
    high_risk_pmci_index = int(np.argmax(pmci_probabilities))

    smci_probabilities = np.where(smci_mask, y_prob_xgb, 2)
    high_confidence_smci_index = int(np.argmin(smci_probabilities))

    uncertain_index = int(np.argmin(np.abs(y_prob_xgb - 0.5)))

    return {
        "high_risk_pMCI": high_risk_pmci_index,
        "high_conf_sMCI": high_confidence_smci_index,
        "most_uncertain": uncertain_index,
    }


def save_waterfall_plots(shap_explanation_xgb, y_test, y_prob_xgb):
    section("PLOT 4 - SHAP waterfall plots for XGBoost")

    if shap_explanation_xgb is None:
        return None

    selected_cases = select_waterfall_cases(y_test, y_prob_xgb)

    for case_name, index in selected_cases.items():
        true_label = "pMCI" if y_test[index] == 1 else "sMCI"
        probability = y_prob_xgb[index]

        plt.figure(figsize=(10, 6))

        shap.plots.waterfall(
            shap_explanation_xgb[index],
            max_display=12,
            show=False,
        )

        plt.title(
            f"SHAP Waterfall - {case_name.replace('_', ' ')}\n"
            f"True: {true_label} | pMCI probability: {probability:.3f}",
            pad=12,
        )

        plt.tight_layout()

        out_path = PLOTS_DIR / f"shap_waterfall_{case_name}.png"

        plt.savefig(out_path, **PLOT_KW)
        plt.close()

        print(
            f"  Saved: {out_path.name} "
            f"(subject {index}, true={true_label}, prob={probability:.3f})"
        )

    return selected_cases


def save_force_plots(
    shap_values_xgb,
    expected_value_xgb,
    x_test_processed_xgb,
    feature_labels,
    y_prob_xgb,
    selected_cases,
):
    section("PLOT 5 - SHAP force plots for XGBoost")

    if selected_cases is None:
        return

    force_cases = [
        ("force_pos", selected_cases["high_risk_pMCI"], "pMCI"),
        ("force_neg", selected_cases["high_conf_sMCI"], "sMCI"),
    ]

    for case_name, index, label in force_cases:
        plt.figure(figsize=(14, 3))

        shap.force_plot(
            expected_value_xgb,
            shap_values_xgb[index],
            x_test_processed_xgb[index],
            feature_names=feature_labels,
            matplotlib=True,
            show=False,
        )

        plt.title(
            f"Force plot - {label} (prob={y_prob_xgb[index]:.3f})",
            pad=10,
        )

        plt.tight_layout()

        out_path = PLOTS_DIR / f"shap_xgb_{case_name}.png"

        plt.savefig(out_path, **PLOT_KW)
        plt.close()

        print(f"  Saved: {out_path.name}")


def save_dependence_plots(
    shap_values_xgb,
    x_test_processed_xgb,
    feature_labels,
):
    section("PLOT 6 - SHAP dependence plots for top XGBoost features")

    mean_abs_shap = np.abs(shap_values_xgb).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:6]

    for feature_index in top_indices:
        feature_name = feature_labels[feature_index]

        fig, ax = plt.subplots(figsize=(7, 5))

        shap.dependence_plot(
            feature_index,
            shap_values_xgb,
            x_test_processed_xgb,
            feature_names=feature_labels,
            ax=ax,
            show=False,
        )

        ax.set_title(f"SHAP Dependence - {feature_name} (XGBoost)")

        plt.tight_layout()

        safe_name = feature_name.replace(" ", "_").replace("/", "-")
        out_path = PLOTS_DIR / f"shap_xgb_dependence_{safe_name}.png"

        plt.savefig(out_path, **PLOT_KW)
        plt.close()

        print(f"  Saved: {out_path.name}")


def print_final_summary():
    section("shap_clinical.py complete")

    plots_saved = list(PLOTS_DIR.glob("*.png"))

    print(f"\n  Plots saved : {len(plots_saved)} files -> {PLOTS_DIR}")

    print("  CSVs saved  : shap_importance_xgboost.csv")
    print("                shap_importance_randomforest.csv")
    print("                shap_values_xgboost.csv")
    print("                shap_values_randomforest.csv")
    print("                shap_comparison.csv")

    print(
        """
  Plot inventory:
    shap_bar_xgboost.png
    shap_bar_randomforest.png
    shap_xgb_beeswarm.png
    shap_rf_beeswarm.png
    shap_waterfall_high_risk_pMCI.png
    shap_waterfall_high_conf_sMCI.png
    shap_waterfall_most_uncertain.png
    shap_xgb_force_pos.png
    shap_xgb_force_neg.png
    shap_xgb_dependence_*.png

  Next: validate_oasis2.py
"""
    )


def main():
    x_test, y_test, valid_features, feature_labels = load_data()

    xgb_pipeline, rf_pipeline = load_models()

    x_test_processed_xgb = None
    x_test_processed_rf = None

    shap_values_xgb = None
    expected_value_xgb = None
    shap_explanation_xgb = None
    shap_explanation_rf = None

    if xgb_pipeline is not None:
        x_test_processed_xgb = preprocess_for_shap(
            xgb_pipeline,
            x_test,
        )

        _, shap_values_xgb, expected_value_xgb, shap_explanation_xgb = build_shap_explanation(
            xgb_pipeline,
            x_test_processed_xgb,
            feature_labels,
        )

    if rf_pipeline is not None:
        x_test_processed_rf = preprocess_for_shap(
            rf_pipeline,
            x_test,
        )

        _, _, _, shap_explanation_rf = build_shap_explanation(
            rf_pipeline,
            x_test_processed_rf,
            feature_labels,
        )

    save_shap_importance_plots(
        xgb_pipeline,
        rf_pipeline,
        x_test_processed_xgb,
        x_test_processed_rf,
        valid_features,
        feature_labels,
    )

    save_xgb_beeswarm(shap_explanation_xgb)

    save_rf_beeswarm(shap_explanation_rf)

    if xgb_pipeline is not None:
        y_prob_xgb = xgb_pipeline.predict_proba(x_test)[:, 1]

        selected_cases = save_waterfall_plots(
            shap_explanation_xgb,
            y_test,
            y_prob_xgb,
        )

        save_force_plots(
            shap_values_xgb,
            expected_value_xgb,
            x_test_processed_xgb,
            feature_labels,
            y_prob_xgb,
            selected_cases,
        )

        save_dependence_plots(
            shap_values_xgb,
            x_test_processed_xgb,
            feature_labels,
        )

    print_final_summary()


if __name__ == "__main__":
    main()