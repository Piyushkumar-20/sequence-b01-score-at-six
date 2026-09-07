import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    """
    Calculate standard regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "MAE": mae,
        "RMSE": rmse,
    }


def evaluate_predictions(y_true, y_pred, model_name):
    """
    Evaluate one set of predictions and return the results.
    """
    metrics = calculate_metrics(y_true, y_pred)

    print(model_name)
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")

    return {
        "model": model_name,
        **metrics,
    }


def prediction_errors(y_true, y_pred):
    """
    Calculate signed and absolute prediction errors.
    """
    errors = y_true.to_numpy() - np.asarray(y_pred)

    return {
        "errors": errors,
        "absolute_errors": np.abs(errors),
        "mean_absolute_error": np.mean(np.abs(errors)),
        "largest_error": np.max(np.abs(errors)),
    }

from src.evaluate import (
    evaluate_predictions,
    prediction_errors,
)


baseline_result = evaluate_predictions(
    y_test,
    baseline_pred,
    "Current Run Rate Baseline",
)

linear_result = evaluate_predictions(
    y_test,
    linear_pred,
    "Linear Regression",
)

boosting_result = evaluate_predictions(
    y_test,
    boosting_pred,
    "Gradient Boosting",
)


results = [
    baseline_result,
    linear_result,
    boosting_result,
]

for result in results:
    print(result)


linear_errors = prediction_errors(
    y_test,
    linear_pred,
)

print(
    "Mean absolute error:",
    linear_errors["mean_absolute_error"],
)

print(
    "Largest absolute error:",
    linear_errors["largest_error"],
)
