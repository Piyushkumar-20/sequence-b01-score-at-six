import random
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def current_run_rate_baseline(df):
    """
    Baseline prediction:
    current run rate × 20 overs.
    """
    return df["current_run_rate"].to_numpy() * 20


def build_linear_regression():
    """
    Create the Linear Regression model.
    """
    return LinearRegression()


def build_gradient_boosting():
    """
    Create the Gradient Boosting regression model.
    """
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=SEED,
    )