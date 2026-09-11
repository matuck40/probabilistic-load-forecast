"""Seasonal naive: the ruler the other models are measured against.

Tomorrow at 14:00 will look like today at 14:00, or like last Tuesday at 14:00.
On electricity load this is a strong forecast, not a straw man: the weekly
autocorrelation of the target series is 0.890. A repository without this
baseline has no scale on which its other numbers mean anything.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.models.base import Forecaster


class SeasonalNaive(Forecaster):
    def __init__(self, lag: int) -> None:
        if lag < 24:
            raise ValueError("a lag below 24 hours is not observed at a midnight origin")
        self.lag = lag
        self.name = f"naive_{lag}h"
        self.residual_quantiles: tuple[float, float] | None = None

    def fit(self, train: pd.Series) -> None:
        """Fit nothing but the interval, from residuals on the training window."""
        residuals = (train - train.shift(self.lag)).dropna()
        low, high = config.QUANTILES[0], config.QUANTILES[2]
        self.residual_quantiles = (
            float(residuals.quantile(low)),
            float(residuals.quantile(high)),
        )

    def predict(self, series: pd.Series, test_index: pd.DatetimeIndex) -> pd.DataFrame:
        if self.residual_quantiles is None:
            raise RuntimeError("call fit before predict")
        point = series.shift(self.lag).reindex(test_index)
        low_offset, high_offset = self.residual_quantiles
        return self._frame(test_index, point + low_offset, point, point + high_offset)
