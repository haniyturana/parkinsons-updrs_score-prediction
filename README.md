# Parkinson's Disease UPDRS Score Prediction using Voice Features

This repository contains a machine learning pipeline designed to predict the **Unified Parkinson’s Disease Rating Scale (UPDRS)** score of patients using linear and time-frequency based voice features. The pipeline implements and compares multiple regression techniques: **Lasso Regression**, **Ridge Regression**, **Principal Component Regression (PCR)**, **Random Forest**, and **XGBoost**.

---

## 📊 Dataset Overview
The dataset includes biometric voice recordings from **40 individuals** (20 diagnosed with Parkinson's Disease and 20 healthy control subjects). From each subject, multiple voice samples (26 recordings including sustained vowels, numbers, words, and short sentences) were captured.

### Demographic Breakdown:
* **Parkinson's Cohort:** 20 individuals (14 male, 6 female)
* **Healthy Control Cohort:** 20 individuals (10 male, 10 female)

### Attribute Information:
* **Column 1:** Subject ID (`SubjectID`)
* **Columns 2–27 (26 Speech Features):**
  * **Jitter Variants (1–5):** Local, Local (Absolute), Rap, PPQ5, DDP (Measures of pitch period variations).
  * **Shimmer Variants (6–11):** Local, Local (dB), APQ3, APQ5, APQ11, DDA (Measures of amplitude variations).
  * **Harmonicity Parameters (12–14):** AC (Autocorrelation), NTH (Noise-to-Harmonic), HTN (Harmonic-to-Noise).
  * **Pitch Statistics (15–19):** Median, Mean, Standard Deviation, Minimum, and Maximum pitch.
  * **Pulse & Period Metrics (20–23):** Number of pulses, Number of periods, Mean period, Standard deviation of period.
  * **Voice Stability Analysis (24–26):** Fraction of locally unvoiced frames, Number of voice breaks, Degree of voice breaks.
* **Column 28 (Target):** UPDRS Score (Clinically determined by an expert physician).
* **Column 29 (Class Label):** Classification metadata (`0`: Healthy, `1`: Parkinson's).

---

## 🛠️ Pipeline Architecture
To combat the data characteristics (high feature correlation and multiple vocal replicates clustered per subject), the pipeline executes the following stages:

1. **Feature Standardization:** Uses `StandardScaler` to normalize the scales of various physical sound units (Hz, dB, counts, percentages) prior to applying regression models.
2. **Group-Aware Validation:** Employs a `GroupKFold` strategy nested within `GridSearchCV` on `SubjectID`. This ensures that voice samples belonging to the same individual do not cross-contaminate both the training and test sets during validation.
3. **Hyperparameter Tuning:** Automatically evaluates penalty factors ($\alpha$) and estimator properties to identify the configurations that minimize Mean Squared Error (MSE).

---

## 📈 Experimental Results & Discussion

### Model Evaluation
The models were trained and evaluated across training partitions (to diagnose overfitting trends) and unseen test partitions. Below is the summary of the performance metrics achieved:

| Model | Train MSE | Test MSE | MSE Diff | Model Status | Test $R^2$ Score | Interpretation |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| **Lasso** | 225.58 | 256.08 | 30.50 | Slightly Overfit | `1%` | Almost no predictive power |
| **Ridge** | 212.82 | 258.00 | 45.18 | Moderately Overfit | `1%` | Almost no predictive power |
| **PCR** | 214.16 | 265.54 | 51.38 | Overfit | `-2%` | Almost no predictive power |
| **Random Forest** | 192.94 | 234.89 | 41.95 | Moderately Overfit | `9%` | Weak but slightly improved |
| **XGBoost** | 90.91 | 228.62 | 137.71 | Severely Overfit | `10%` | Weak but slightly improved |

### Key Takeaways & Discussion:
* **Severe Generalization Challenges:** Linear models with regularization (Lasso and Ridge) struggled heavily to model the dataset, explaining only roughly **1%** of the target variance ($R^2 = 0.01$). This indicates that the direct relationships between standard acoustic metrics and the complex clinical UPDRS score are largely non-linear.
* **Non-Linear Model Gains & Overfitting:** Non-linear tree-based ensembles (**Random Forest** and **XGBoost**) managed to push the variance explained up to **9% and 10%** respectively. However, XGBoost suffered from severe overfitting (with a massive MSE gap of `137.71`), meaning it memorized noise within the training subset rather than extracting generalized patterns.
* **Feature Quality Constraints:** Based on the study, the primary performance bottleneck is not the choice of algorithm, but rather the **quality and informativeness of the available voice features**. The acoustic features extracted may contain high levels of ambient/recording noise or lack the medical signal complexity needed to capture the structural variations of a clinical UPDRS score.

---

## 💡 Conclusion & Future Recommendations
* **Shift to Feature Engineering:** Future work should redirect focus away from hyperparameter tuning and instead focus on advanced **feature engineering**.
* **Alternative Representations:** Incorporating deep learning embeddings (e.g., mel-spectrogram features processed via CNNs or Wav2Vec models) might capture hidden voice markers better than simple linear temporal/pitch statistics.
* **Inclusion of Clinical Predictors:** Adding multi-modal markers, such as non-voice clinical variables or demographic details, could provide more informative predictors to stabilize the target mapping.

---

## 💻 Technical Stack
* **Language:** Python 3.x
* **Core Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `xgboost`
* **Development Environment:** Jupyter Notebook
