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
