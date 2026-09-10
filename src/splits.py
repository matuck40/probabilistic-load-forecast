"""Walk-forward folds with a rolling origin and an expanding training window.

Random splitting would let the model train on next July to predict last March.
Every number this repository reports comes from a model that only ever saw the
past of the period it was scored on.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass(frozen=True)
class Fold:
    number: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train(self, series: pd.Series) -> pd.Series:
        """Everything up to and including train_end. Nothing after it, ever."""
        return series.loc[: self.train_end]

    def test_index(self, series: pd.Series) -> pd.DatetimeIndex:
        return series.loc[self.test_start : self.test_end].index


def folds() -> list[Fold]:
    return [
        Fold(
            number=i,
            train_end=pd.Timestamp(train_end),
            test_start=pd.Timestamp(test_start),
            test_end=pd.Timestamp(test_end),
        )
        for i, (train_end, test_start, test_end) in enumerate(config.FOLDS, start=1)
    ]
