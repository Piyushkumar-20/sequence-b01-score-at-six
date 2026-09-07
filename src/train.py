import random
import numpy as np

from pathlib import Path
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from src.data_loader import load_checkpoints
from src.model import (
    current_run_rate_baseline,
    build_linear_regression,
    build_gradient_boosting,
)


FEATURE_COLUMNS = [
    "runs_so_far",
    "wickets_down",
    "current_run_rate",
    "balls_since_boundary",
    "batting_team",
]

TARGET_COLUMN = "final_score"


def prepare_data(checkpoints):
    """
    Keep only over-6 checkpoint rows and remove
    rows with missing required values.
    """
    data = checkpoints[
        checkpoints["over_mark"] == 6
    ].copy()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    data = data.dropna(
        subset=required_columns
    )

    return data


def build_preprocessor():
    """
    Encode batting_team while keeping numeric
    features unchanged.
    """
    numeric_features = [
        "runs_so_far",
        "wickets_down",
        "current_run_rate",
        "balls_since_boundary",
    ]

    categorical_features = [
        "batting_team"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "team",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    return preprocessor


def main():

    # --------------------------------------------------
    # 1. Load checkpoints
    # --------------------------------------------------
    checkpoints = load_checkpoints()

    # --------------------------------------------------
    # 2. Prepare over-6 data
    # --------------------------------------------------
    data = prepare_data(checkpoints)

    # --------------------------------------------------
    # 3. Features and target
    # --------------------------------------------------
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    # --------------------------------------------------
    # 4. Development train/test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
    )

    # --------------------------------------------------
    # 5. Current Run Rate baseline
    # --------------------------------------------------
    baseline_pred = current_run_rate_baseline(
        X_test
    )

    # --------------------------------------------------
    # 6. Preprocessing
    # --------------------------------------------------
    preprocessor = build_preprocessor()

    # --------------------------------------------------
    # 7. Linear Regression
    # --------------------------------------------------
    linear_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                build_linear_regression(),
            ),
        ]
    )

    linear_model.fit(
        X_train,
        y_train,
    )

    linear_pred = linear_model.predict(
        X_test
    )

    # --------------------------------------------------
    # 8. Gradient Boosting
    # --------------------------------------------------
    boosting_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                build_gradient_boosting(),
            ),
        ]
    )

    boosting_model.fit(
        X_train,
        y_train,
    )

    boosting_pred = boosting_model.predict(
        X_test
    )

    # --------------------------------------------------
    # 9. Display predictions
    # --------------------------------------------------
    print("\nCurrent Run Rate Baseline:")
    print(baseline_pred[:5])

    print("\nLinear Regression:")
    print(linear_pred[:5])

    print("\nGradient Boosting:")
    print(boosting_pred[:5])

    return (
        y_test,
        baseline_pred,
        linear_pred,
        boosting_pred,
    )


if __name__ == "__main__":
    main()