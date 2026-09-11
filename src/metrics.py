"""Scoring rules.

WAPE answers "how much energy did we get wrong, relative to how much there
was". Pinball loss scores a quantile: it is the only one of these that can tell
a useful interval from a decorative one, and coverage says whether the interval
keeps the promise printed on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def wape(y: pd.Series, yhat: pd.Series) -> float:
    """Total absolute error divided by total actual.

    Not the mean of per-hour percentage errors: that statistic is dominated by
    the smallest denominators, which in load are the quiet night hours nobody
    is paid to get right.
    """
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def mae(y: pd.Series, yhat: pd.Series) -> float:
    return float(np.abs(y - yhat).mean())


def rmse(y: pd.Series, yhat: pd.Series) -> float:
    return float(np.sqrt(((y - yhat) ** 2).mean()))


def pinball_loss(y: pd.Series, yhat: pd.Series, alpha: float) -> float:
    """Asymmetric loss for the alpha-quantile.

    Being below the truth costs alpha per unit; being above costs 1 - alpha.
    A P90 forecast is therefore punished nine times harder for being too low
    than too high, which is what makes it a P90 rather than a mean.
    """
    error = y - yhat
    return float(np.maximum(alpha * error, (alpha - 1) * error).mean())


def mean_pinball(y: pd.Series, predictions: pd.DataFrame) -> float:
    losses = [pinball_loss(y, predictions[f"q{q}"], q) for q in config.QUANTILES]
    return float(np.mean(losses))


def empirical_coverage(y: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Fraction of observations inside the interval.

    Compared against NOMINAL_COVERAGE. Promising 80% and delivering 55% means
    the interval is decoration.
    """
    return float(((y >= lower) & (y <= upper)).mean())


def evaluate_predictions(y: pd.Series, predictions: pd.DataFrame) -> dict[str, float]:
    low, mid, high = config.QUANTILES[0], config.QUANTILES[1], config.QUANTILES[2]
    point = predictions[f"q{mid}"]
    return {
        "wape": wape(y, point),
        "mae": mae(y, point),
        "rmse": rmse(y, point),
        "pinball": mean_pinball(y, predictions),
        "coverage": empirical_coverage(y, predictions[f"q{low}"], predictions[f"q{high}"]),
    }
