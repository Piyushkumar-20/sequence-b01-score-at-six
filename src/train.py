from __future__ import annotations

import csv
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Allow imports from project root when run as: python src/train.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_checkpoints
from src.evaluate import evaluate_predictions, prediction_errors
from src.model import current_run_rate_baseline

SEED = 42
VALIDATION_SIZE = 0.20
BOOTSTRAP_SAMPLES = 2000

FEATURE_COLUMNS = [
    "runs_so_far",
    "wickets_down",
    "current_run_rate",
    "balls_since_boundary",
    "batting_team",
]
TARGET_COLUMN = "final_score"


# Reproducibility for any future stochastic work in this script.
random.seed(SEED)
np.random.seed(SEED)


def prepare_data(checkpoints: pd.DataFrame) -> pd.DataFrame:
    """Keep complete Over-6 observations for B1."""
    data = checkpoints.loc[checkpoints["over_mark"] == 6].copy()
    required_columns = ["match_id", *FEATURE_COLUMNS, TARGET_COLUMN]
    data = data.dropna(subset=required_columns)

    if data.empty:
        raise ValueError("No complete Over-6 observations are available.")

    return data


def development_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a match-level development/validation split.

    This is intentionally called validation, not test: the final test data must
    remain untouched until the final evaluation stage.
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
    )
    train_idx, validation_idx = next(
        splitter.split(data, groups=data["match_id"])
    )
    train = data.iloc[train_idx].copy()
    validation = data.iloc[validation_idx].copy()

    train_matches = set(train["match_id"])
    validation_matches = set(validation["match_id"])
    overlap = train_matches.intersection(validation_matches)
    if overlap:
        raise RuntimeError(
            f"Match leakage detected: {len(overlap)} match IDs appear in both splits."
        )

    return train, validation


def append_experiment(row: dict[str, object]) -> None:
    """Append one experiment record using the guidebook schema."""
    path = PROJECT_ROOT / "experiments.csv"
    fields = [
        "run_id",
        "date",
        "who",
        "what_changed",
        "main_metric",
        "value",
        "seed",
        "notes",
    ]

    existing = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not existing:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    checkpoints = load_checkpoints()
    data = prepare_data(checkpoints)
    train_data, validation_data = development_split(data)

    # Week 4 baseline: hold the current run rate flat for 20 overs.
    y_validation = validation_data[TARGET_COLUMN]
    baseline_predictions = current_run_rate_baseline(validation_data)

    result = evaluate_predictions(
        y_validation,
        baseline_predictions,
        "Current Run Rate Baseline",
        bootstrap_seed=SEED,
        n_bootstrap=BOOTSTRAP_SAMPLES,
    )

    # Persist the actual validation predictions so the result is auditable.
    prediction_frame = validation_data[
        ["match_id", "innings", "over_mark", TARGET_COLUMN]
    ].copy()
    prediction_frame["predicted_score"] = baseline_predictions
    prediction_frame["error"] = (
        prediction_frame[TARGET_COLUMN] - prediction_frame["predicted_score"]
    )
    prediction_frame["absolute_error"] = prediction_frame["error"].abs()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(
        results_dir / "baseline_predictions.csv",
        index=False,
    )

    metrics_frame = pd.DataFrame([result])
    metrics_frame.insert(1, "validation_rows", len(validation_data))
    metrics_frame.insert(2, "validation_matches", validation_data["match_id"].nunique())
    metrics_frame.insert(3, "validation_seed", SEED)
    metrics_frame.to_csv(
        results_dir / "baseline_metrics.csv",
        index=False,
    )

    errors = prediction_errors(y_validation, baseline_predictions)
    worst = errors.nlargest(10, "absolute_error").copy()
    worst["match_id"] = validation_data.iloc[worst.index]["match_id"].to_numpy()
    worst["innings"] = validation_data.iloc[worst.index]["innings"].to_numpy()
    worst.to_csv(results_dir / "baseline_failure_cases.csv", index=False)

    # Log the two required metrics.  The interval is included in notes because
    # experiments.csv has one value column in the guidebook's schema.
    run_date = date.today().isoformat()
    append_experiment(
        {
            "run_id": "W4-B1-BASELINE-MAE-42",
            "date": run_date,
            "who": "Member 1",
            "what_changed": "Established Over-6 current-run-rate baseline with match-level development split",
            "main_metric": "MAE",
            "value": f"{result['MAE']:.4f}",
            "seed": SEED,
            "notes": (
                f"95% bootstrap CI [{result['MAE_CI_low']:.4f}, {result['MAE_CI_high']:.4f}]; "
                f"validation rows={len(validation_data)}; validation matches={validation_data['match_id'].nunique()}."
            ),
        }
    )
    append_experiment(
        {
            "run_id": "W4-B1-BASELINE-RMSE-42",
            "date": run_date,
            "who": "Member 1",
            "what_changed": "Established Over-6 current-run-rate baseline with match-level development split",
            "main_metric": "RMSE",
            "value": f"{result['RMSE']:.4f}",
            "seed": SEED,
            "notes": (
                f"95% bootstrap CI [{result['RMSE_CI_low']:.4f}, {result['RMSE_CI_high']:.4f}]; "
                f"validation rows={len(validation_data)}; validation matches={validation_data['match_id'].nunique()}."
            ),
        }
    )

    print("\nB1 Week 4 — Current Run Rate Baseline")
    print(f"Development rows: {len(train_data)}")
    print(f"Validation rows:  {len(validation_data)}")
    print(f"Validation matches: {validation_data['match_id'].nunique()}")
    print("Final test data: NOT USED")
    print(f"MAE:  {result['MAE']:.2f} (95% bootstrap CI {result['MAE_CI_low']:.2f}–{result['MAE_CI_high']:.2f})")
    print(f"RMSE: {result['RMSE']:.2f} (95% bootstrap CI {result['RMSE_CI_low']:.2f}–{result['RMSE_CI_high']:.2f})")
    print(f"Saved: {results_dir / 'baseline_metrics.csv'}")
    print(f"Saved: {results_dir / 'baseline_predictions.csv'}")
    print(f"Saved: {results_dir / 'baseline_failure_cases.csv'}")
    print(f"Updated: {PROJECT_ROOT / 'experiments.csv'}")


if __name__ == "__main__":
    main()
