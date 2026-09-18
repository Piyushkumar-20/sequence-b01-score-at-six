from __future__ import annotations

import csv
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
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
BOOTSTRAP_SAMPLES = 2000


def prepare_data(checkpoints: pd.DataFrame) -> pd.DataFrame:
    """Keep complete Over-6 observations for B1."""
    data = checkpoints.loc[checkpoints["over_mark"] == 6].copy()
    required = ["match_id", *FEATURE_COLUMNS, TARGET_COLUMN]
    data = data.dropna(subset=required)
    if data.empty:
        raise ValueError("No complete Over-6 observations are available.")
    return data


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("team", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_linear_regression() -> Pipeline:
    """Build the Week 6 Linear Regression model."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )


def build_gradient_boosting(seed: int) -> Pipeline:
    """Build the Week 6 Gradient Boosting model for one experiment seed."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=seed,
                ),
            ),
        ]
    )


def current_run_rate_baseline(data: pd.DataFrame) -> np.ndarray:
    return data["current_run_rate"].to_numpy(dtype=float) * 20.0


def metrics(y_true, y_pred) -> tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(mae), float(rmse)


def bootstrap_ci(y_true, y_pred, metric: str, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for a validation metric."""
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
    """Append one experiment row, replacing an existing identical run_id."""
    path = PROJECT_ROOT / "experiments.csv"
    fields = [
        "run_id", "date", "who", "what_changed",
        "main_metric", "value", "seed", "notes",
    ]
    rows: list[dict[str, str]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    run_id = str(row["run_id"])
    rows = [existing for existing in rows if existing.get("run_id") != run_id]
    rows.append({field: str(row.get(field, "")) for field in fields})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_model_experiment(
    data: pd.DataFrame,
    model_name: str,
    model_builder,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one model across the ten required match-level validation seeds."""
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

        model = model_builder(seed)
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

        all_rows.append(
            {
                "seed": seed,
                "model": model_name,
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
            }
        )

        errors = pd.DataFrame(
            {
                "match_id": validation["match_id"].to_numpy(),
                "innings": validation["innings"].to_numpy(),
                "actual_score": validation[TARGET_COLUMN].to_numpy(),
                "predicted_score": model_predictions,
                "error": validation[TARGET_COLUMN].to_numpy() - model_predictions,
            }
        )
        errors["absolute_error"] = errors["error"].abs()
        worst = errors.nlargest(5, "absolute_error").copy()
        worst.insert(0, "seed", seed)
        failure_rows.append(worst)

    results = pd.DataFrame(all_rows)
    failures = pd.concat(failure_rows, ignore_index=True)
    summary = pd.DataFrame(
        [
            {
                "model": model_name,
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
        ]
    )
    return results, summary, failures


def save_model_outputs(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    failures: pd.DataFrame,
    file_stem: str,
) -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / f"{file_stem}_seed_results.csv", index=False)
    summary.to_csv(results_dir / f"{file_stem}_summary.csv", index=False)
    failures.to_csv(results_dir / f"{file_stem}_failure_cases.csv", index=False)


def log_model_results(
    summary: pd.DataFrame, model_name: str, run_prefix: str, who: str
) -> None:
    row = summary.loc[0]
    run_date = date.today().isoformat()
    append_experiment(
        {
            "run_id": f"{run_prefix}-10-SEEDS",
            "date": run_date,
            "who": who,
            "what_changed": f"Implemented Over-6 {model_name} with match-level validation",
            "main_metric": "MAE",
            "value": f"{row['MAE_mean']:.4f}",
            "seed": "0-9",
            "notes": (
                f"10 match-level splits; MAE mean {row['MAE_mean']:.4f}, "
                f"range [{row['MAE_min']:.4f}, {row['MAE_max']:.4f}], "
                f"std {row['MAE_std']:.4f}; baseline mean "
                f"{row['baseline_MAE_mean']:.4f}."
            ),
        }
    )
    append_experiment(
        {
            "run_id": f"{run_prefix}-10-SEEDS-RMSE",
            "date": run_date,
            "who": who,
            "what_changed": f"Compared Over-6 {model_name} with the Week 4 current-run-rate baseline",
            "main_metric": "RMSE",
            "value": f"{row['RMSE_mean']:.4f}",
            "seed": "0-9",
            "notes": (
                f"10 match-level splits; RMSE mean {row['RMSE_mean']:.4f}, "
                f"range [{row['RMSE_min']:.4f}, {row['RMSE_max']:.4f}], "
                f"std {row['RMSE_std']:.4f}; baseline mean "
                f"{row['baseline_RMSE_mean']:.4f}."
            ),
        }
    )


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    data = prepare_data(load_checkpoints())

    linear_results, linear_summary, linear_failures = run_model_experiment(
        data,
        "Linear Regression",
        lambda _seed: build_linear_regression(),
    )
    save_model_outputs(
        linear_results,
        linear_summary,
        linear_failures,
        "linear_regression",
    )
    log_model_results(
        linear_summary, "Linear Regression", "W6-B1-LINEAR-REGRESSION", "Member 3"
    )

    gradient_results, gradient_summary, gradient_failures = run_model_experiment(
        data,
        "Gradient Boosting",
        build_gradient_boosting,
    )
    save_model_outputs(
        gradient_results,
        gradient_summary,
        gradient_failures,
        "gradient_boosting",
    )
    log_model_results(
        gradient_summary, "Gradient Boosting", "W6-B1-GRADIENT-BOOSTING", "Member 1"
    )

    print("\nB1 Week 6–7 — Model comparison")
    print(f"Seeds: {len(SEEDS)} (0–9)")
    print("Match-level validation: yes")
    for summary in (linear_summary, gradient_summary):
        row = summary.loc[0]
        print(f"\n{row['model']}")
        print(f"MAE mean: {row['MAE_mean']:.2f}")
        print(f"MAE range: {row['MAE_min']:.2f}–{row['MAE_max']:.2f}")
        print(f"MAE std: {row['MAE_std']:.2f}")
        print(f"Baseline MAE mean: {row['baseline_MAE_mean']:.2f}")
        print(f"RMSE mean: {row['RMSE_mean']:.2f}")
        print(f"RMSE range: {row['RMSE_min']:.2f}–{row['RMSE_max']:.2f}")
        print(f"RMSE std: {row['RMSE_std']:.2f}")
        print(f"Baseline RMSE mean: {row['baseline_RMSE_mean']:.2f}")

    print("\nSaved Linear Regression outputs in results/.")
    print("Saved Gradient Boosting outputs in results/.")
    print(f"Updated: {PROJECT_ROOT / 'experiments.csv'}")


if __name__ == "__main__":
    main()
