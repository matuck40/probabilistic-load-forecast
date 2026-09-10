import numpy as np
import pandas as pd

from src.error_analysis import error_breakdown, worst_segments


def _pair(n=720, seed=5):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=n, freq="h")
    actual = pd.Series(30000 + rng.normal(0, 200, n), index=index)
    predicted = actual.copy()
    # Inject a known defect: hour 19 is forecast 10% low.
    predicted[predicted.index.hour == 19] *= 0.9
    return actual, predicted


def test_breakdown_covers_every_segment():
    actual, predicted = _pair()
    breakdown = error_breakdown(actual, predicted)
    assert set(breakdown["segment"]) == {"hour", "dayofweek", "month", "holiday"}
    assert (breakdown["n"] > 0).all()


def test_worst_segment_finds_the_injected_defect():
    actual, predicted = _pair()
    worst = worst_segments(error_breakdown(actual, predicted))
    top = worst.iloc[0]
    assert top["segment"] == "hour"
    assert int(top["bucket"]) == 19
    assert top["ratio_to_overall"] > 2


def test_ratio_is_relative_to_the_overall_wape():
    actual, predicted = _pair()
    breakdown = error_breakdown(actual, predicted)
    hours = breakdown[breakdown.segment == "hour"]
    assert hours["ratio_to_overall"].min() < 1 < hours["ratio_to_overall"].max()
