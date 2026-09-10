"""The feature matrix, built separately for each horizon.

Leakage in a forecasting project almost never looks like a bug. It looks like a
very good score. The single rule enforced here is that a feature for a target at
time t may only use observations at or before the forecast origin for that
target, and that origin is fixed by the horizon.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.calendar_features import calendar_frame

# Hour-by-hour lags for the shortest horizons (1-12), the same hour on the
# previous day (24), and the same hour on the four previous same weekdays
# (168, 336, 504, 672).
LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 168, 336, 504, 672)
SAME_HOUR_DOW_LAGS = (168, 336, 504, 672)
SAME_HOUR_7D_LAGS = (24, 48, 72, 96, 120, 144, 168)


def _usable(lags: tuple[int, ...], horizon: int) -> tuple[int, ...]:
    """The single definition of the L >= h rule.

    The forecast is issued at 00:00 of day D and the last observation is 23:00
    of day D-1, so the target for horizon h is 00:00 + (h-1) hours and a lag of
    L hours reaches back to a timestamp that is observed exactly when L >= h.
    Every lag list in this module is filtered through this one function, so the
    rule is written exactly once.
    """
    return tuple(lag for lag in lags if lag >= horizon)


def _check_no_leakage(lags: tuple[int, ...], horizon: int) -> None:
    """Raise ValueError if any lag in `lags` would not yet be observed at `horizon`.

    This re-verifies the rule against the lags actually used to build the
    feature matrix, independently of whether they were produced by `_usable`,
    so that a future lag list added without the `>= horizon` filter cannot
    quietly leak. A real check rather than `assert`, so `python -O` cannot
    strip it.
    """
    for lag in lags:
        if lag < horizon:
            raise ValueError(f"lag {lag} would not be observed at horizon {horizon}")


def available_lags(horizon: int) -> tuple[int, ...]:
    """Lags whose value is already observed when the forecast is issued."""
    if horizon not in config.HORIZONS:
        raise ValueError(f"horizon must be one of {config.HORIZONS}, got {horizon}")
    return _usable(LAGS, horizon)


def build_features(series: pd.Series, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for one horizon, indexed by the target timestamp."""
    usable = available_lags(horizon)
    dow_lags = _usable(SAME_HOUR_DOW_LAGS, horizon)
    week_lags = _usable(SAME_HOUR_7D_LAGS, horizon)
    _check_no_leakage((*usable, *dow_lags, *week_lags), horizon)

    columns: dict[str, pd.Series] = {}
    for lag in usable:
        columns[f"lag_{lag}"] = series.shift(lag)

    dow_frame = pd.concat([series.shift(lag) for lag in dow_lags], axis=1)
    columns["mean_same_hour_dow_4w"] = dow_frame.mean(axis=1)
    columns["std_same_hour_dow_4w"] = dow_frame.std(axis=1)

    columns["mean_same_hour_7d"] = pd.concat([series.shift(lag) for lag in week_lags], axis=1).mean(axis=1)

    X = pd.DataFrame(columns, index=series.index)

    # Calendar position is known arbitrarily far in advance, so it carries no
    # leakage risk. Raw integers only: trees split on thresholds, and a sine
    # transform would only blur the cut.
    X["hour"] = series.index.hour.astype("float64")
    X["dayofweek"] = series.index.dayofweek.astype("float64")
    X["month"] = series.index.month.astype("float64")
    X["dayofyear"] = series.index.dayofyear.astype("float64")
    X = X.join(calendar_frame(series.index))

    y = series.rename("load")
    complete = X.notna().all(axis=1)
    return X.loc[complete], y.loc[complete]
