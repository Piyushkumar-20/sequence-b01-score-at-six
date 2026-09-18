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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
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



CONFORMAL_COVERAGE = 0.90
CALIBRATION_SIZE = 0.20


def conformal_quantile(abs_residuals: np.ndarray, coverage: float) -> float:
    """Return a finite-sample conservative absolute-residual quantile."""
    values = np.sort(np.asarray(abs_residuals, dtype=float))
    if values.size == 0:
        raise ValueError("No calibration residuals are available.")
    rank = int(np.ceil((values.size + 1) * coverage)) - 1
    rank = min(max(rank, 0), values.size - 1)
    return float(values[rank])


def build_final_prediction_output(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a leakage-safe B1 prediction file and uncertainty validation report.

    For each seed, GroupKFold assigns every match to exactly one outer fold.
    The outer fold is never used for fitting or calibration. The remaining
    matches are split into model-fit and calibration portions, and a 90%
    split-conformal residual quantile is used for the prediction interval.
    """
    outputs = []
    interval_rows = []

    for seed in SEEDS:
        outer_cv = GroupKFold(n_splits=10, shuffle=True, random_state=seed)

        for fold_number, (fit_cal_idx, outer_idx) in enumerate(
            outer_cv.split(data, groups=data["match_id"])
        ):
            fit_cal = data.iloc[fit_cal_idx].copy()
            outer = data.iloc[outer_idx].copy()

            inner_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=CALIBRATION_SIZE,
                random_state=seed * 100 + fold_number,
            )
            fit_idx, calibration_idx = next(
                inner_splitter.split(fit_cal, groups=fit_cal["match_id"])
            )
            fit_data = fit_cal.iloc[fit_idx].copy()
            calibration = fit_cal.iloc[calibration_idx].copy()

            model = build_gradient_boosting(seed)
            model.fit(fit_data[FEATURE_COLUMNS], fit_data[TARGET_COLUMN])

            calibration_predictions = model.predict(
                calibration[FEATURE_COLUMNS]
            )
            calibration_residuals = (
                calibration[TARGET_COLUMN].to_numpy(dtype=float)
                - calibration_predictions
            )
            interval_half_width = conformal_quantile(
                np.abs(calibration_residuals), CONFORMAL_COVERAGE
            )

            outer_predictions = model.predict(outer[FEATURE_COLUMNS])
            lower = np.maximum(
                0.0, outer_predictions - interval_half_width
            )
            upper = outer_predictions + interval_half_width

            fold_output = pd.DataFrame(
                {
                    "match": outer["match_id"].to_numpy(),
                    "innings": outer["innings"].to_numpy(),
                    "predicted_score": outer_predictions,
                    "low_estimate": lower,
                    "high_estimate": upper,
                    "actual_score": outer[TARGET_COLUMN].to_numpy(),
                    "interval_half_width": interval_half_width,
                    "seed": seed,
                    "fold": fold_number,
                }
            )
            outputs.append(fold_output)

            interval_rows.append(
                {
                    "seed": seed,
                    "fold": fold_number,
                    "coverage_target": CONFORMAL_COVERAGE,
                    "empirical_coverage": (
                        (
                            (fold_output["actual_score"] >= fold_output["low_estimate"])
                            & (
                                fold_output["actual_score"]
                                <= fold_output["high_estimate"]
                            )
                        ).mean()
                    ),
                    "interval_half_width": interval_half_width,
                    "mean_range_width": (
                        fold_output["high_estimate"]
                        - fold_output["low_estimate"]
                    ).mean(),
                    "validation_rows": len(fold_output),
                    "validation_matches": fold_output["match"].nunique(),
                }
            )

    all_predictions = pd.concat(outputs, ignore_index=True)

    # Each row appears exactly once for every seed. Average the ten seed
    # predictions/ranges to obtain one final handoff row per match/innings.
    grouped = (
        all_predictions.groupby(["match", "innings"], as_index=False)
        .agg(
            predicted_score=("predicted_score", "mean"),
            low_estimate=("low_estimate", "mean"),
            high_estimate=("high_estimate", "mean"),
            actual_score=("actual_score", "first"),
        )
    )

    if len(grouped) != len(data):
        raise RuntimeError(
            "Final prediction output does not contain exactly one row per "
            "Over-6 input match/innings."
        )

    grouped["predicted_score"] = grouped["predicted_score"].clip(lower=0)
    grouped["low_estimate"] = np.minimum(
        grouped["low_estimate"], grouped["predicted_score"]
    )
    grouped["high_estimate"] = np.maximum(
        grouped["high_estimate"], grouped["predicted_score"]
    )
    grouped["covered"] = (
        (grouped["actual_score"] >= grouped["low_estimate"])
        & (grouped["actual_score"] <= grouped["high_estimate"])
    )

    coverage_summary = pd.DataFrame(
        [
            {
                "coverage_target": CONFORMAL_COVERAGE,
                "empirical_coverage": grouped["covered"].mean(),
                "covered_rows": int(grouped["covered"].sum()),
                "total_rows": len(grouped),
                "mean_range_width": (
                    grouped["high_estimate"] - grouped["low_estimate"]
                ).mean(),
                "median_range_width": (
                    grouped["high_estimate"] - grouped["low_estimate"]
                ).median(),
                "point_MAE": mean_absolute_error(
                    grouped["actual_score"], grouped["predicted_score"]
                ),
                "point_RMSE": np.sqrt(
                    mean_squared_error(
                        grouped["actual_score"], grouped["predicted_score"]
                    )
                ),
            }
        ]
    )
    return grouped, coverage_summary

def save_final_prediction_outputs(
    predictions: pd.DataFrame, coverage_summary: pd.DataFrame
) -> None:
    """Write the B1 handoff file and uncertainty validation report."""
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    handoff = predictions[
        ["match", "innings", "predicted_score", "low_estimate", "high_estimate"]
    ].copy()
    handoff["match"] = handoff["match"].astype(str)
    handoff["innings"] = handoff["innings"].astype(int)
    handoff["predicted_score"] = handoff["predicted_score"].round(2)
    handoff["low_estimate"] = handoff["low_estimate"].round(2)
    handoff["high_estimate"] = handoff["high_estimate"].round(2)
    handoff.to_csv(results_dir / "pred_over6.csv", index=False)

    coverage_summary.to_csv(
        results_dir / "prediction_interval_coverage.csv", index=False
    )

    predictions[
        [
            "match",
            "innings",
            "predicted_score",
            "low_estimate",
            "high_estimate",
            "actual_score",
            "covered",
        ]
    ].to_csv(
        results_dir / "final_prediction_validation.csv", index=False
    )


def log_final_prediction_experiment(
    coverage_summary: pd.DataFrame,
) -> None:
    row = coverage_summary.loc[0]
    append_experiment(
        {
            "run_id": "W8-B1-FINAL-PREDICTION-CONFORMAL",
            "date": date.today().isoformat(),
            "who": "Member 2",
            "what_changed": (
                "Generate leakage-safe Over-6 predictions with 90% "
                "split-conformal prediction intervals"
            ),
            "main_metric": "coverage",
            "value": f"{row['empirical_coverage']:.4f}",
            "seed": "0-9",
            "notes": (
                f"Out-of-fold predictions across 10 match-level outer splits; "
                f"target coverage {row['coverage_target']:.2f}, empirical coverage "
                f"{row['empirical_coverage']:.4f}; mean range width "
                f"{row['mean_range_width']:.2f}; point MAE {row['point_MAE']:.4f}; "
                f"point RMSE {row['point_RMSE']:.4f}."
            ),
        }
    )


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    data = prepare_data(load_checkpoints())

    # Existing Week 6–7 model comparisons remain part of the main pipeline.
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

    final_predictions, coverage_summary = build_final_prediction_output(data)
    save_final_prediction_outputs(final_predictions, coverage_summary)
    log_final_prediction_experiment(coverage_summary)

    row = coverage_summary.loc[0]
    print("\nB1 final prediction output")
    print("Final model: Gradient Boosting")
    print(f"Rows: {len(final_predictions)}")
    print(
        f"90% interval empirical coverage: "
        f"{row['empirical_coverage']:.2%}"
    )
    print(f"Mean interval width: {row['mean_range_width']:.2f} runs")
    print(f"Point MAE (OOF): {row['point_MAE']:.2f}")
    print(f"Point RMSE (OOF): {row['point_RMSE']:.2f}")
    print("Saved: results/pred_over6.csv")
    print("Saved: results/prediction_interval_coverage.csv")
    print("Saved: results/final_prediction_validation.csv")


if __name__ == "__main__":
    main()
