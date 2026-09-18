from __future__ import annotations

import csv
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_checkpoints

FEATURE_COLUMNS = [
    "runs_so_far",
    "wickets_down",
    "current_run_rate",
    "balls_since_boundary",
    "batting_team",
]
NUMERIC_FEATURES = FEATURE_COLUMNS[:-1]
CATEGORICAL_FEATURES = ["batting_team"]
TARGET_COLUMN = "final_score"

VALIDATION_SIZE = 0.20
SEEDS = list(range(10))
# Week 4 baseline values are evaluated on the same match-level split for each seed.
BOOTSTRAP_SAMPLES = 2000


def prepare_data(checkpoints: pd.DataFrame) -> pd.DataFrame:
    """Keep complete Over-6 observations for B1."""
    data = checkpoints.loc[checkpoints["over_mark"] == 6].copy()
    required = ["match_id", *FEATURE_COLUMNS, TARGET_COLUMN]
    data = data.dropna(subset=required)
    if data.empty:
        raise ValueError("No complete Over-6 observations are available.")
    return data


def build_model() -> Pipeline:
    """Build the Week 6 Linear Regression model with team one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("team", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def current_run_rate_baseline(data: pd.DataFrame) -> np.ndarray:
    return data["current_run_rate"].to_numpy(dtype=float) * 20.0


def metrics(y_true, y_pred) -> tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(mae), float(rmse)


def bootstrap_ci(y_true, y_pred, metric: str, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for a single seed's validation metric."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    indices = rng.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))
    errors = y_true[indices] - y_pred[indices]
    if metric == "MAE":
        values = np.mean(np.abs(errors), axis=1)
    else:
        values = np.sqrt(np.mean(errors ** 2, axis=1))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def append_experiment(row: dict[str, object]) -> None:
    path = PROJECT_ROOT / "experiments.csv"
    fields = [
        "run_id", "date", "who", "what_changed",
        "main_metric", "value", "seed", "notes",
    ]
    existing = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not existing:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    data = prepare_data(load_checkpoints())
    all_rows = []
    failure_rows = []

    for seed in SEEDS:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SIZE,
            random_state=seed,
        )
        train_idx, validation_idx = next(
            splitter.split(data, groups=data["match_id"])
        )
        train = data.iloc[train_idx].copy()
        validation = data.iloc[validation_idx].copy()

        train_matches = set(train["match_id"])
        validation_matches = set(validation["match_id"])
        if train_matches.intersection(validation_matches):
            raise RuntimeError("Match leakage detected between train and validation.")

        model = build_model()
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        model_predictions = model.predict(validation[FEATURE_COLUMNS])
        baseline_predictions = current_run_rate_baseline(validation)

        model_mae, model_rmse = metrics(validation[TARGET_COLUMN], model_predictions)
        base_mae, base_rmse = metrics(validation[TARGET_COLUMN], baseline_predictions)
        model_mae_low, model_mae_high = bootstrap_ci(
            validation[TARGET_COLUMN], model_predictions, "MAE", seed + 1000
        )
        model_rmse_low, model_rmse_high = bootstrap_ci(
            validation[TARGET_COLUMN], model_predictions, "RMSE", seed + 2000
        )

        all_rows.extend([
            {
                "seed": seed,
                "model": "Linear Regression",
                "validation_rows": len(validation),
                "validation_matches": validation["match_id"].nunique(),
                "MAE": model_mae,
                "RMSE": model_rmse,
                "MAE_CI_low": model_mae_low,
                "MAE_CI_high": model_mae_high,
                "RMSE_CI_low": model_rmse_low,
                "RMSE_CI_high": model_rmse_high,
                "baseline_MAE": base_mae,
                "baseline_RMSE": base_rmse,
                "MAE_change_vs_baseline": model_mae - base_mae,
                "RMSE_change_vs_baseline": model_rmse - base_rmse,
            },
        ])

        errors = pd.DataFrame({
            "match_id": validation["match_id"].to_numpy(),
            "innings": validation["innings"].to_numpy(),
            "actual_score": validation[TARGET_COLUMN].to_numpy(),
            "predicted_score": model_predictions,
            "error": validation[TARGET_COLUMN].to_numpy() - model_predictions,
        })
        errors["absolute_error"] = errors["error"].abs()
        worst = errors.nlargest(5, "absolute_error").copy()
        worst.insert(0, "seed", seed)
        failure_rows.append(worst)

    results = pd.DataFrame(all_rows)
    failures = pd.concat(failure_rows, ignore_index=True)

    summary = pd.DataFrame([
        {
            "model": "Linear Regression",
            "seeds": len(SEEDS),
            "MAE_mean": results["MAE"].mean(),
            "MAE_std": results["MAE"].std(ddof=1),
            "MAE_min": results["MAE"].min(),
            "MAE_max": results["MAE"].max(),
            "RMSE_mean": results["RMSE"].mean(),
            "RMSE_std": results["RMSE"].std(ddof=1),
            "RMSE_min": results["RMSE"].min(),
            "RMSE_max": results["RMSE"].max(),
            "baseline_MAE_mean": results["baseline_MAE"].mean(),
            "baseline_RMSE_mean": results["baseline_RMSE"].mean(),
            "MAE_mean_change_vs_baseline": results["MAE_change_vs_baseline"].mean(),
            "RMSE_mean_change_vs_baseline": results["RMSE_change_vs_baseline"].mean(),
        }
    ])

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "linear_regression_seed_results.csv", index=False)
    summary.to_csv(results_dir / "linear_regression_summary.csv", index=False)
    failures.to_csv(results_dir / "linear_regression_failure_cases.csv", index=False)

    run_date = date.today().isoformat()
    append_experiment({
        "run_id": "W6-B1-LINEAR-REGRESSION-10-SEEDS",
        "date": run_date,
        "who": "Member 3",
        "what_changed": "Implemented Over-6 Linear Regression with match-level validation",
        "main_metric": "MAE",
        "value": f"{summary.loc[0, 'MAE_mean']:.4f}",
        "seed": "0-9",
        "notes": (
            f"10 match-level splits; MAE mean {summary.loc[0, 'MAE_mean']:.4f}, "
            f"range [{summary.loc[0, 'MAE_min']:.4f}, {summary.loc[0, 'MAE_max']:.4f}], "
            f"std {summary.loc[0, 'MAE_std']:.4f}; baseline mean "
            f"{summary.loc[0, 'baseline_MAE_mean']:.4f}."
        ),
    })
    append_experiment({
        "run_id": "W6-B1-LINEAR-REGRESSION-10-SEEDS-RMSE",
        "date": run_date,
        "who": "Member 3",
        "what_changed": "Compared Over-6 Linear Regression with the Week 4 current-run-rate baseline",
        "main_metric": "RMSE",
        "value": f"{summary.loc[0, 'RMSE_mean']:.4f}",
        "seed": "0-9",
        "notes": (
            f"10 match-level splits; RMSE mean {summary.loc[0, 'RMSE_mean']:.4f}, "
            f"range [{summary.loc[0, 'RMSE_min']:.4f}, {summary.loc[0, 'RMSE_max']:.4f}], "
            f"std {summary.loc[0, 'RMSE_std']:.4f}; baseline mean "
            f"{summary.loc[0, 'baseline_RMSE_mean']:.4f}."
        ),
    })

    print("\nB1 Week 6–7 — Linear Regression")
    print(f"Seeds: {len(SEEDS)} (0–9)")
    print(f"Match-level validation: yes")
    print(f"MAE mean: {summary.loc[0, 'MAE_mean']:.2f}")
    print(f"MAE range: {summary.loc[0, 'MAE_min']:.2f}–{summary.loc[0, 'MAE_max']:.2f}")
    print(f"MAE std: {summary.loc[0, 'MAE_std']:.2f}")
    print(f"Baseline MAE mean: {summary.loc[0, 'baseline_MAE_mean']:.2f}")
    print(f"RMSE mean: {summary.loc[0, 'RMSE_mean']:.2f}")
    print(f"RMSE range: {summary.loc[0, 'RMSE_min']:.2f}–{summary.loc[0, 'RMSE_max']:.2f}")
    print(f"RMSE std: {summary.loc[0, 'RMSE_std']:.2f}")
    print(f"Baseline RMSE mean: {summary.loc[0, 'baseline_RMSE_mean']:.2f}")
    print(f"Saved: {results_dir / 'linear_regression_seed_results.csv'}")
    print(f"Saved: {results_dir / 'linear_regression_summary.csv'}")
    print(f"Saved: {results_dir / 'linear_regression_failure_cases.csv'}")
    print(f"Updated: {PROJECT_ROOT / 'experiments.csv'}")


if __name__ == "__main__":
    main()
