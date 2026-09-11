"""Where the model breaks, by segment.

An aggregate WAPE tells you the model is imperfect. It does not tell you that
the imperfection lives entirely in the evening peak, or on the working day
between a holiday and a weekend. Fixing what you found, and reporting how much
it moved, is the part of this work that is hard to fake.
"""

from __future__ import annotations

import pandas as pd

from src.calendar_features import calendar_frame
from src.metrics import wape

SEGMENTS = ("hour", "dayofweek", "month", "holiday")


def _buckets(index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    frame["hour"] = index.hour
    frame["dayofweek"] = index.dayofweek
    frame["month"] = index.month
    frame["holiday"] = calendar_frame(index)["is_holiday"].astype(int).to_numpy()
    return frame


def error_breakdown(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    overall = wape(actual, predicted)
    buckets = _buckets(actual.index)
    rows = []
    for segment in SEGMENTS:
        for bucket, group in buckets.groupby(segment):
            sliced = group.index
            segment_wape = wape(actual.loc[sliced], predicted.loc[sliced])
            rows.append(
                {
                    "segment": segment,
                    "bucket": bucket,
                    "n": len(sliced),
                    "wape": segment_wape,
                    "ratio_to_overall": segment_wape / overall,
                }
            )
    return pd.DataFrame(rows)


def worst_segments(breakdown: pd.DataFrame, minimum_n: int = 24) -> pd.DataFrame:
    """Rank segments by how much worse they are than the model overall.

    minimum_n guards against a bucket of three hours looking catastrophic by
    chance and sending the next hour of work in the wrong direction.
    """
    eligible = breakdown[breakdown["n"] >= minimum_n]
    return eligible.sort_values("ratio_to_overall", ascending=False).reset_index(drop=True)
