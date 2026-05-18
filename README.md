# Parkinson's Disease UPDRS Score Prediction using Voice Features

This repository contains a machine learning pipeline designed to predict the **Unified Parkinson’s Disease Rating Scale (UPDRS)** score of patients using linear and time-frequency based voice features. The pipeline implements and compares three regularization and dimensionality reduction techniques: **Ridge Regression**, **LASSO Regression**, and **Principal Component Regression (PCR)**.

---

## 📊 Dataset Overview
The dataset includes biometric voice recordings from **40 individuals** (20 diagnosed with Parkinson's Disease and 20 healthy control subjects). From each subject, multiple voice samples (26 recordings including sustained vowels, numbers, words, and short sentences) were captured.

### Attribute Information:
* **Column 1:** Subject ID
* **Columns 2–27 (26 Speech Features):**
  * **Jitter Variants (1–5):** Local, Local (Absolute), Rap, PPQ5, DDP (Measures of pitch period variations).
  * **Shimmer Variants (6–11):** Local, Local (dB), APQ3, APQ5, APQ11, DDA (Measures of amplitude variations).
  * **Harmonicity Parameters (12–14):** AC, NTH, HTN (Noise-to-harmonic/harmonic-to-noise ratios).
  * **Pitch Statistics (15–19):** Median, Mean, Standard Deviation, Minimum, and Maximum pitch.
  * **Pulse & Period Metrics (20–23):** Number of pulses, Number of periods, Mean period, Standard deviation of period.
  * **Voice Break Analysis (24–26):** Fraction of locally unvoiced frames, Number of voice breaks, Degree of voice breaks.
* **Column 28 (Target):** UPDRS Score (Clinically determined by an expert physician).
* **Column 29 (Class Label):** Classification metadata (`0`: Healthy, `1`: Parkinson's).

---

## 🛠️ Pipeline Architecture
To combat the data characteristics (high feature correlation and multiple vocal replicates clustered per subject), the pipeline executes the following stages:

1. **Group-Based Splitting:** Employs `GroupShuffleSplit` on `Subject ID` to ensure that voice samples belonging to the same individual do not cross-contaminate both the training and test sets.
2. **Feature Standardization:** Uses `StandardScaler` to normalize the scales of various physical sound units (Hz, dB, counts, percentages) prior to applying regularized costs.
3. **Hyperparameter Optimization:** Utilizes cross-validated grid search (`GridSearchCV`, `RidgeCV`, `LassoCV`) to determine optimal penalty factors ($\alpha$).
4. **Dimensionality Reduction (PCR):** Constructs a `Pipeline` connecting `PCA` mapping with an ordinary `LinearRegression` engine.

---

## 📈 Experimental Results & Discussion

### Model Evaluation
The models were optimized over their respective hyperparameter spaces and evaluated using Mean Squared Error (MSE) and $R^2$ statistics on the unseen test partition:

| Machine Learning Model | Optimal Hyperparameters | Test Mean Squared Error (MSE) | Test $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Ridge Regression** | $\alpha = 10.0$ *(via RidgeCV)* | `125.1098` | Train: `0.6558` / Test: `0.5841` |
| **LASSO Regression** | $\alpha = 0.0933$ *(via LassoCV)* | `124.6934` | Train: `0.6482` / Test: `0.5855` |
| **PCR (PCA + OLS)** | $n\_components = 7$ *(95% Variance)* | `131.2589` | Train: `0.6277` / Test: `0.5637` |

### Key Takeaways & Discussion:
* **Feature Multicollinearity:** The acoustic features (specifically various Jitter and Shimmer calculations) exhibit extreme levels of multicollinearity. Standard OLS models overfit significantly under these conditions.
* **Ridge vs. LASSO Effectiveness:** **LASSO Regression** provided the best performance with the lowest testing error (**Test MSE: 124.6934**) and explained the highest variance (**Test $R^2$: 0.5855**). Since LASSO performs L1 regularization, it successfully drives non-impactful collinear coefficient weights to absolute zero, serving as an automated feature selector.
* **PCR Representation:** Principal Component Regression successfully captures variance by consolidating the 26 acoustic metrics into orthogonal components. By retaining only **7 principal components** (representing 95% of explained variance), it effectively eliminates multi-variable inflation while producing competitive testing accuracy.

---

## 💻 Technical Stack
* **Language:** Python 3.x
* **Core Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Development Environment:** Jupyter Notebook

