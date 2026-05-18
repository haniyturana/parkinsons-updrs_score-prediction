# Parkinson's Disease UPDRS Score Prediction using Voice Features

This repository contains a machine learning pipeline designed to predict the **Unified Parkinson’s Disease Rating Scale (UPDRS)** score of patients using linear and time-frequency based voice features. The pipeline implements and compares three regularization and dimensionality reduction techniques: **Ridge Regression**, **LASSO Regression**, and **Principal Component Regression (PCR)**.

---

## 📊 Dataset Overview
The dataset includes biometric voice recordings from **40 individuals** (20 diagnosed with Parkinson's Disease and 20 healthy control subjects). From each subject, multiple voice samples (26 recordings including sustained vowels, numbers, words, and short sentences) were captured.

### Demographic Breakdown:
* **Parkinson's Cohort:** 20 individuals (14 male, 6 female)
* **Healthy Control Cohort:** 20 individuals (10 male, 10 female)

### Attribute Information:
* **Column 1:** Subject ID (`SubjectID`)
* **Columns 2–27 (26 Acoustic Features):**
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
To address high feature correlation and protect against multiple vocal replicates clustered per subject, the dataset executes the following processing stages:

1. **Feature Standardization:** Uses `StandardScaler` to normalize the scales of various physical sound units (Hz, dB, counts, percentages) prior to applying cost-regularization steps.
2. **Group-Aware Validation:** Utilizes a `GroupKFold` strategy paired with a grid-search engine (`GridSearchCV`, `RidgeCV`, `LassoCV`) on `SubjectID` to guarantee that voice samples belonging to the same individual do not cross-contaminate both the training and evaluation steps during tuning.
3. **Hyperparameter Optimization:** Automatically iterates over hyperparameter spaces to locate optimal cost penalties ($\alpha$).
4. **Dimensionality Reduction (PCR):** Leverages a scikit-learn `Pipeline` pairing Principal Component Analysis (`PCA`) directly with an Ordinary Least Squares (`LinearRegression`) core.

---

## 📈 Experimental Results & Discussion

### Model Evaluation
Models were cross-validated and assessed across training partitions (to diagnose fit stability) and unseen testing partitions using Mean Squared Error (MSE) and $R^2$ statistics:

| Machine Learning Model | Optimal Hyperparameters | Test Mean Squared Error (MSE) | Test $R^2$ Score | Train $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **LASSO Regression** | $\alpha = 0.0933$ *(via LassoCV)* | `124.6934` | `0.5855` | `0.6482` |
| **Ridge Regression** | $\alpha = 10.0$ *(via RidgeCV)* | `125.1098` | `0.5841` | `0.6558` |
| **PCR (PCA + OLS)** | $n\_components = 7$ *(95% Variance)* | `131.2589` | `0.5637` | `0.6277` |

### Key Takeaways & Discussion:
* **Feature Multicollinearity:** The acoustic data structure (especially repetitive Jitter and Shimmer calculations) exhibits massive multicollinearity. Standard OLS models without penalty factors overfit significantly under these conditions.
* **L1 vs. L2 Regularization Effectiveness:** **LASSO Regression** yielded the highest performance metrics, minimizing testing error down to a **Test MSE of 124.6934** and explaining the greatest degree of unseen variance (**Test $R^2$: 0.5855**). Because LASSO deploys an L1 penalty, it successfully drops non-impactful or highly collinear coefficient weights to absolute zero, operating as an automated feature selector.
* **PCR Representation:** Principal Component Regression successfully captures variance by consolidating the 26 metrics into orthogonal components. Retaining only **7 principal components** (representing 95% of explained variance) effectively eliminates multi-variable inflation while producing competitive testing accuracy.
* **Core Limitation Summary:** Evaluation metrics across all models suggest that performance boundaries are primarily capped by the informativeness and noise levels of the voice features rather than limitations in the linear modeling approaches. Further development should emphasize feature engineering or adding complementary clinical predictors.

---

## 💻 Technical Stack
* **Language:** Python 3.x
* **Core Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `xgboost`
* **Development Environment:** Jupyter Notebook
