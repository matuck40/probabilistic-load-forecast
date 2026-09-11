"""The contract every forecaster satisfies.

One interface means the evaluation harness does not care which model it is
scoring, and adding a fourth model later costs one file instead of a rewrite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src import config

QUANTILE_COLUMNS = [f"q{q}" for q in config.QUANTILES]


class Forecaster(ABC):
    """Fit on a training window, then produce three quantiles per target hour."""

    name: str

    @abstractmethod
    def fit(self, train: pd.Series) -> None:
        """Learn from the fold's training window. Never sees the test period."""

    @abstractmethod
    def predict(self, series: pd.Series, test_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return a frame indexed by test_index with columns QUANTILE_COLUMNS."""

    @staticmethod
    def _frame(index: pd.DatetimeIndex, low, mid, high) -> pd.DataFrame:
        frame = pd.DataFrame(dict(zip(QUANTILE_COLUMNS, [low, mid, high], strict=True)), index=index)
        # Quantiles estimated independently can cross, which would make coverage
        # meaningless. Sorting each row repairs that. np.sort does it in one
        # vectorised pass; a row-wise apply here costs minutes over six folds.
        return pd.DataFrame(np.sort(frame.to_numpy(), axis=1), index=index, columns=QUANTILE_COLUMNS)
