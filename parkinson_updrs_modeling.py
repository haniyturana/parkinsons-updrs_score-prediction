"""Train group-aware models for Parkinson's UPDRS prediction.

This script replaces notebook-only experimentation with a reproducible command-line
pipeline. It keeps all samples from the same subject in the same split, prevents
scaling leakage by fitting preprocessing only on training folds, and writes a
model comparison table plus diagnostic plots.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REQUIRED_PACKAGES = {
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
}
missing_packages = [
    package_name
    for import_name, package_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(import_name) is None
]
if missing_packages:
    package_list = " ".join(missing_packages)
    sys.exit(
        "Missing required package(s): "
        f"{', '.join(missing_packages)}. "
        "Install them with `python -m pip install -r requirements.txt` "
        f"or `python -m pip install {package_list}`."
    )

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FEATURE_COLUMNS = [f"Feat_{i}" for i in range(1, 27)]
COLUMN_NAMES = ["SubjectID", *FEATURE_COLUMNS, "UPDRS", "Class"]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the comma-separated acoustic feature dataset."""
    df = pd.read_csv(path, header=None)
    expected_columns = len(COLUMN_NAMES)
    if df.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns, found {df.shape[1]} in {path}")
    df.columns = COLUMN_NAMES
    return df


def build_models(include_xgboost: bool) -> dict[str, tuple[Pipeline | RandomForestRegressor, dict[str, list]]]:
    """Create model pipelines and compact hyperparameter grids."""
    models: dict[str, tuple[Pipeline | RandomForestRegressor, dict[str, list]]] = {
        "Baseline Mean": (Pipeline([("model", DummyRegressor(strategy="mean"))]), {}),
        "Lasso": (
            Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=100_000, random_state=RANDOM_STATE))]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "Ridge": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=RANDOM_STATE))]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "PCR": (
            Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("model", LinearRegression())]),
            {"pca__n_components": list(range(1, 27))},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [200, 500],
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [3, 5, 10],
                "max_features": ["sqrt", 0.75, 1.0],
            },
        ),
    }

    if include_xgboost and importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor

        models["XGBoost"] = (
            XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror", n_jobs=1),
            {
                "n_estimators": [100, 250],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
                "reg_lambda": [1.0, 5.0],
            },
        )
    return models


def evaluate(model, x_train, y_train, x_test, y_test) -> dict[str, float]:
    """Return train/test metrics for a fitted estimator."""
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    return {
        "Train MSE": mean_squared_error(y_train, train_pred),
        "Test MSE": mean_squared_error(y_test, test_pred),
        "Test RMSE": mean_squared_error(y_test, test_pred) ** 0.5,
        "Test MAE": mean_absolute_error(y_test, test_pred),
        "Test R2": r2_score(y_test, test_pred),
    }


def plot_diagnostics(results: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    """Save compact model-comparison and prediction diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    ordered = results.sort_values("Test MSE")
    sns.barplot(data=ordered, x="Test MSE", y="Model", color="#4C72B0")
    plt.title("Group-Holdout Test MSE by Model")
    plt.tight_layout()
    plt.savefig(output_dir / "model_test_mse.png", dpi=150)
    plt.close()

    best_model = ordered.iloc[0]["Model"]
    best_predictions = predictions[predictions["Model"] == best_model]
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=best_predictions, x="Actual UPDRS", y="Predicted UPDRS", hue="Class", palette="viridis")
    limits = [best_predictions[["Actual UPDRS", "Predicted UPDRS"]].min().min(), best_predictions[["Actual UPDRS", "Predicted UPDRS"]].max().max()]
    plt.plot(limits, limits, "r--", label="Perfect fit")
    plt.title(f"Actual vs Predicted UPDRS ({best_model})")
    plt.tight_layout()
    plt.savefig(output_dir / "best_model_actual_vs_predicted.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train group-aware UPDRS regression models.")
    parser.add_argument("--data", type=Path, default=Path("cousticfeatureparkinson.txt"), help="Path to comma-separated dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for metrics and plots.")
    parser.add_argument("--include-xgboost", action="store_true", help="Include XGBoost when the package is installed.")
    args = parser.parse_args()

    df = load_dataset(args.data)
    x = df[FEATURE_COLUMNS]
    y = df["UPDRS"]
    groups = df["SubjectID"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    cv = GroupKFold(n_splits=5)
    rows = []
    prediction_rows = []
    for name, (estimator, grid) in build_models(args.include_xgboost).items():
        search = GridSearchCV(estimator, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        search.fit(x_train, y_train, groups=train_groups)
        metrics = evaluate(search.best_estimator_, x_train, y_train, x_test, y_test)
        rows.append({"Model": name, "Best Params": search.best_params_, **metrics})
        for index, actual, predicted in zip(x_test.index, y_test, search.best_estimator_.predict(x_test)):
            prediction_rows.append({
                "Model": name,
                "SubjectID": df.loc[index, "SubjectID"],
                "Class": df.loc[index, "Class"],
                "Actual UPDRS": actual,
                "Predicted UPDRS": predicted,
            })

    results = pd.DataFrame(rows).sort_values("Test MSE")
    predictions = pd.DataFrame(prediction_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "model_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    plot_diagnostics(results, predictions, args.output_dir)

    print("\nModel comparison (sorted by Test MSE):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
