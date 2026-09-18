"""Evaluation utilities for the B1 Over-6 regression baseline.

This module intentionally contains no top-level experiment code.  It can be
imported safely by training scripts and notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate standard regression metrics."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    return {
        "MAE": float(mean_absolute_error(y_true_array, y_pred_array)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
    }


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric: str = "MAE",
    n_bootstrap: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for a metric.

    The bootstrap resamples prediction errors/pairs within the validation set.
    This gives an uncertainty interval for the reported validation metric; it is
    not a substitute for multi-seed model variability, which is handled later.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if len(y_true_array) != len(y_pred_array):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true_array) == 0:
        raise ValueError("Cannot bootstrap an empty validation set")

    rng = np.random.default_rng(seed)
    n = len(y_true_array)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    true_samples = y_true_array[indices]
    pred_samples = y_pred_array[indices]

    if metric.upper() == "MAE":
        values = np.mean(np.abs(true_samples - pred_samples), axis=1)
    elif metric.upper() == "RMSE":
        values = np.sqrt(np.mean((true_samples - pred_samples) ** 2, axis=1))
    else:
        raise ValueError("metric must be 'MAE' or 'RMSE'")

    alpha = 1.0 - confidence
    lower = float(np.quantile(values, alpha / 2))
    upper = float(np.quantile(values, 1.0 - alpha / 2))
    return lower, upper


def evaluate_predictions(
    y_true,
    y_pred,
    model_name: str,
    *,
    bootstrap_seed: int = 42,
    n_bootstrap: int = 2000,
) -> dict[str, float | str]:
    """Evaluate predictions and attach 95% bootstrap CIs to each metric."""
    metrics = calculate_metrics(y_true, y_pred)
    mae_low, mae_high = bootstrap_metric_ci(
        y_true, y_pred, "MAE", n_bootstrap, bootstrap_seed
    )
    rmse_low, rmse_high = bootstrap_metric_ci(
        y_true, y_pred, "RMSE", n_bootstrap, bootstrap_seed + 1
    )

    return {
        "model": model_name,
        "MAE": metrics["MAE"],
        "MAE_CI_low": mae_low,
        "MAE_CI_high": mae_high,
        "RMSE": metrics["RMSE"],
        "RMSE_CI_low": rmse_low,
        "RMSE_CI_high": rmse_high,
    }


def prediction_errors(y_true, y_pred) -> pd.DataFrame:
    """Return row-level signed and absolute prediction errors."""
    true_values = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)

    if len(true_values) != len(predictions):
        raise ValueError("y_true and y_pred must have the same length")

    return pd.DataFrame(
        {
            "actual_score": true_values,
            "predicted_score": predictions,
            "error": true_values - predictions,
            "absolute_error": np.abs(true_values - predictions),
        }
    )
