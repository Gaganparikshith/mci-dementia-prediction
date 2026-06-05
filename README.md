# Explainable MCI-to-Dementia Conversion Prediction

## Overview

This project is about predicting whether a person with **Mild Cognitive Impairment (MCI)** may progress to dementia within a **36-month period**.

MCI is a stage where a person has memory or thinking difficulties, but they can still manage daily life independently. Some people with MCI remain stable, while others later develop Alzheimer’s disease or dementia. The aim of this project is to build an explainable machine learning system that can help estimate this conversion risk using clinical assessments and MRI-based analysis.

The main focus of the project is not only prediction accuracy, but also **explainability**. The model should not behave like a black box. It should help users understand which clinical features influenced the prediction.

---

## Project Aim

The aim of this project is to develop an explainable, non-invasive, and clinically useful machine learning framework for predicting conversion from MCI to dementia.

The project uses clinical data and MRI data from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)** dataset. External validation was also tested using the **OASIS-2** dataset.

---

## Why this project matters

Early identification of high-risk MCI patients can help clinicians:

* monitor patients more closely,
* plan follow-up assessments,
* identify patients who may need further investigation,
* support clinical trial screening,
* reduce unnecessary dependence on expensive or invasive tests.

This project shows that carefully selected clinical features can provide strong predictive information, and that MRI-based features may not always significantly improve performance over a strong clinical baseline.

---

## Dataset Used

The main dataset used in this project is **ADNI**.

The project uses two main ADNI cohorts:

| Cohort                     | Subjects | Purpose                                |
| -------------------------- | -------: | -------------------------------------- |
| Phase 1 clinical cohort    |      767 | Clinical-only XGBoost model            |
| MRI-quality-assured cohort |      701 | MRI volumetric, CNN, and fusion phases |

The Phase 1 clinical cohort contains:

| Class | Meaning                                | Count |
| ----- | -------------------------------------- | ----: |
| pMCI  | Progressive MCI, converted to dementia |   286 |
| sMCI  | Stable MCI, did not convert            |   481 |

The Phase 1 test set contains:

| Test Set       | Count |
| -------------- | ----: |
| Total subjects |   154 |
| pMCI           |    57 |
| sMCI           |    97 |

---

## Project Phases

## Phase 1: Clinical XGBoost Model

In this phase, the model uses only clinical and cognitive assessment features. MRI, PET, CSF biomarkers, and APOE4 genetic information are excluded to keep the model non-invasive and easier to use in a clinical setting.

The model uses 18 clinical features such as:

* ADAS-Cog 13
* MMSE
* FAQ
* CDR Sum of Boxes
* RAVLT memory test scores
* Digit Span
* Trails B
* MoCA
* GDS
* Education
* Gender
* Composite clinical scores

The best model was **XGBoost**, selected because it achieved the best held-out test performance and supports SHAP-based explainability.

Final Phase 1 performance:

| Metric              | Value |
| ------------------- | ----: |
| AUC-ROC             | 0.821 |
| AUC-PR              | 0.688 |
| Screening threshold |  0.15 |
| Sensitivity at 0.15 | 0.895 |
| Specificity at 0.15 | 0.649 |
| F1-score at 0.15    | 0.718 |

Platt calibration was tested, but it did not improve the Brier score. Therefore, the original XGBoost probabilities were retained.

---

## Phase 1.5: MRI Volumetric Feature Ablation

This phase checks whether adding MRI volumetric features improves the clinical model.

Twelve FreeSurfer-derived MRI volumetric features were added to the 18 clinical features, creating a 30-feature Clinical + MRI model.

Result:

| Model                           | AUC-ROC |
| ------------------------------- | ------: |
| Clinical-only model             |   0.821 |
| Clinical + MRI volumetric model |   0.818 |

The MRI volumetric features did not significantly improve the clinical model. This is an important result because it shows that strong clinical assessments already contain major predictive information.

---

## Phase 2: MRI CNN Model

This phase uses MRI images directly.

A **ResNet-18 CNN** was trained using 2.5D axial MRI triplets. The backbone was frozen and only the final layer was trained to reduce overfitting.

Final MRI CNN result:

| Metric          |          Value |
| --------------- | -------------: |
| AUC-ROC         |          0.710 |
| 95% CI          | [0.606, 0.816] |
| Best checkpoint |       Epoch 16 |
| Validation AUC  |          0.658 |

Grad-CAM was used to visualise which brain regions influenced the CNN prediction. The attention maps focused on hippocampal and entorhinal regions, which are biologically meaningful in Alzheimer’s disease.

---

## Phase 3: Multi-Modal Fusion

This phase checks whether combining clinical predictions and MRI predictions improves performance.

A total of **23 fusion configurations** were tested, including:

* early fusion,
* late fusion,
* uncertain-case MRI correction,
* MRI-only baselines,
* clinical-only baseline.

Best fusion result:

| Model                                 | AUC-ROC | 95% CI         |
| ------------------------------------- | ------: | -------------- |
| Clinical-only baseline on same cohort |   0.855 | [0.777, 0.920] |
| Best fusion model                     |   0.860 | [0.785, 0.924] |

The improvement was only **+0.005 AUC**, and the confidence intervals overlapped. Therefore, fusion did not provide a statistically significant improvement over the clinical-only model.

---

## Explainability

This project uses two explainability methods.

### SHAP

SHAP is used for the clinical XGBoost model. It explains which features influenced the model prediction.

Top SHAP features included:

1. ADAS-Cog 13
2. RAVLT Delayed
3. Digit Span
4. CDR Sum of Boxes
5. Trails B
6. FAQ

SHAP was used to generate:

* global feature importance plots,
* beeswarm plots,
* patient-level waterfall plots.

### Grad-CAM

Grad-CAM is used for the MRI CNN model. It shows which regions of the MRI image influenced the CNN prediction.

This helps check whether the CNN is focusing on meaningful brain regions instead of irrelevant image artefacts.

---

## Streamlit Web App

A simple Streamlit web application was developed.

The app allows the user to enter 18 clinical values and returns:

* pMCI conversion probability,
* risk tier,
* SHAP waterfall explanation,
* clinical disclaimer.

Risk categories:

| Risk Tier | Probability Range |
| --------- | ----------------- |
| Low       | < 0.35            |
| Moderate  | 0.35–0.60         |
| High      | ≥ 0.60            |

This app is only a **research prototype**. It is not a medical diagnostic device.

---

# How to Download and Use the Code from GitHub

## 1. Clone the repository

Open Command Prompt, PowerShell, Git Bash, or terminal and run:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

Then go inside the project folder:

```bash
cd YOUR_REPOSITORY_NAME
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub username and repository name.

Example:

```bash
git clone https://github.com/gaganparikshith04/dementia-conversion-prediction.git
cd dementia-conversion-prediction
```

---

## 2. Create a virtual environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install required packages

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

If the repository does not already have a `requirements.txt` file, create one with the main packages:

```txt
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
optuna
shap
matplotlib
seaborn
torch
torchvision
streamlit
joblib
nibabel
```

Then run:

```bash
pip install -r requirements.txt
```

---

## 4. Prepare the dataset

This project uses ADNI and OASIS-2 data. Because ADNI data is protected under a Data Use Agreement, the raw dataset is **not included** in the GitHub repository.

You need to download the required files separately from the official ADNI portal and place them in the expected folders.

Recommended folder structure:

```text
DementiaResearch/
│
├── data/
│   ├── raw/
│   ├── metadata/
│   ├── labels/
│   └── splits/
│
├── Phase_1_Clinical/
├── Phase_1_5_Clinical_MRI_Volumes/
├── Phase_2_Raw_MRI_CNN/
├── Phase_3_Fusion/
├── Website/
├── outputs/
├── plots/
├── results/
├── requirements.txt
└── README.md
```

Place clinical CSV files such as ADNI clinical tables inside:

```text
data/raw/
```

The scripts will generate processed metadata files inside:

```text
data/metadata/
```

---

## 5. Run Phase 1 Clinical Pipeline

Go to the Phase 1 folder:

```bash
cd Phase_1_Clinical
```

Run dataset building:

```bash
python build_dataset.py
```

Run clinical preprocessing:

```bash
python clinical_cleaning.py
```

Train the clinical model:

```bash
python train_clinical.py
```

Generate SHAP plots:

```bash
python shap_clinical.py
```

Run external validation on OASIS-2:

```bash
python build_oasis2_metadata.py
python validate_oasis2.py
```

---

## 6. Run Phase 1.5 Clinical + MRI Volumetric Model

Go to the Phase 1.5 folder:

```bash
cd ../Phase_1_5_Clinical_MRI_Volumes
```

Run the scripts for combining clinical features with FreeSurfer MRI volumetric features:

```bash
python train_phase15.py
```

Generate SHAP modality decomposition:

```bash
python shap_phase15.py
```

---

## 7. Run Phase 2 MRI CNN Model

Go to the MRI CNN folder:

```bash
cd ../Phase_2_Raw_MRI_CNN
```

Preprocess MRI images:

```bash
python mri_preprocess.py
```

Train the CNN model:

```bash
python train_cnn.py
```

Generate Grad-CAM heatmaps:

```bash
python gradcam_analysis.py
```

---

## 8. Run Phase 3 Fusion

Go to the fusion folder:

```bash
cd ../Phase_3_Fusion
```

Run the fusion ablation experiments:

```bash
python fusion_ablation.py
```

This script evaluates 23 fusion configurations and saves the ranked fusion results.

---

## 9. Run the Streamlit Web App

Go to the website folder:

```bash
cd ../Website
```

Run:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser.

Usually the local URL will be:

```text
http://localhost:8501
```

---

# Expected Outputs

After running the project, outputs will be saved in folders such as:

```text
outputs/
results/
plots/
```

Important outputs include:

| Output File                      | Purpose                                        |
| -------------------------------- | ---------------------------------------------- |
| `best_xgb.pkl`                   | Trained Phase 1 XGBoost model                  |
| `xgb_threshold_tuning.csv`       | Threshold-wise sensitivity/specificity results |
| `clinical_test_bootstrap_ci.csv` | Bootstrap confidence intervals                 |
| `shap_bar_xgboost.png`           | SHAP global feature importance                 |
| `shap_xgb_beeswarm.png`          | SHAP beeswarm plot                             |
| `shap_waterfall_*.png`           | Patient-level explanations                     |
| `oasis2_bootstrap_ci.csv`        | OASIS-2 validation confidence intervals        |
| `cnn2d_v3_training_history.csv`  | CNN training history                           |
| `gradcam_average_heatmap.png`    | Grad-CAM visualisation                         |
| `fusion_ablation_ranked.csv`     | Ranked fusion strategy results                 |

---

# Important Notes

## ADNI data is not included

The raw ADNI data cannot be uploaded publicly because it is governed by ADNI data access rules. Users must download the data themselves after obtaining permission from ADNI.

## This is not a medical device

This project is a research prototype. It should not be used for clinical diagnosis or treatment decisions.

## Results may vary

Results may change slightly depending on:

* hardware,
* random seed,
* library versions,
* preprocessing settings,
* dataset version.

The project uses random seed `42` wherever possible to improve reproducibility.

---

# Main Findings

The project found that a clinical-only XGBoost model using non-invasive clinical features achieved strong prediction performance.

The clinical model achieved:

```text
AUC-ROC = 0.821
AUC-PR  = 0.688
Sensitivity = 0.895 at threshold 0.15
```

Adding MRI volumetric features and MRI CNN fusion did not significantly improve performance. This suggests that clinical neuropsychological assessments carry the strongest predictive signal for this task.

---

# Final Conclusion

This project demonstrates that explainable machine learning can support early risk prediction for MCI-to-dementia conversion using non-invasive clinical features. The system combines strong clinical performance with SHAP-based explanation and MRI-based Grad-CAM validation. The main scientific finding is that a well-tuned clinical model can outperform or match more complex MRI-based and fusion-based approaches, while remaining easier to deploy in real-world clinical settings.
