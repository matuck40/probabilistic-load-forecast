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
    """A breakdown that silently dropped a bucket must be caught here.

    One full week (168 hours, starting at midnight) guarantees every
    hour-of-day (0-23) and every day-of-week (0-6) bucket appears at least
    once, and 2026-01-01 (New Year, the only Brazilian holiday in this
    window) guarantees both holiday buckets (0 and 1) appear too. Checking
    the bucket set alone would not catch a bucket silently merged into
    another one without losing rows, so each segment's row counts are also
    checked to sum back to the length of the input.
    """
    rng = np.random.default_rng(1)
    index = pd.date_range("2026-01-01", periods=168, freq="h")
    actual = pd.Series(30000 + rng.normal(0, 200, len(index)), index=index)
    predicted = actual * 0.98

    breakdown = error_breakdown(actual, predicted)
    assert set(breakdown["segment"]) == {"hour", "dayofweek", "month", "holiday"}
    assert (breakdown["n"] > 0).all()

    hours = breakdown[breakdown.segment == "hour"]
    assert set(hours["bucket"]) == set(range(24))
    assert hours["n"].sum() == len(actual)

    weekdays = breakdown[breakdown.segment == "dayofweek"]
    assert set(weekdays["bucket"]) == set(range(7))
    assert weekdays["n"].sum() == len(actual)

    holidays = breakdown[breakdown.segment == "holiday"]
    assert set(holidays["bucket"]) == {0, 1}
    assert holidays["n"].sum() == len(actual)


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


def test_ratio_denominator_uses_pooled_wape_not_the_mean_of_bucket_wapes():
    """ratio_to_overall must divide by the pooled WAPE over every row, not
    the unweighted mean of the per-bucket WAPEs.

    With equal bucket sizes and near-identical magnitudes (as in `_pair`
    above) the two denominators coincide almost exactly, so a wrong
    implementation using the unweighted mean would still pass every other
    test in this file. This uses two hour buckets with deliberately unequal
    sizes (5 vs 45) and unequal error magnitudes (20% vs 1%) so the pooled
    WAPE and the unweighted mean of the two bucket WAPEs diverge
    substantially, then checks the small bucket's ratio against the pooled
    definition computed by hand from the same numbers used to build it.
    """
    n_small, n_large = 5, 45
    actual_value = 1000.0
    error_small, error_large = 200.0, 10.0  # 20% and 1% of actual_value

    small_hours = pd.date_range("2026-02-01", periods=n_small, freq="D")  # hour 0
    large_hours = pd.date_range("2026-03-01", periods=n_large, freq="D") + pd.Timedelta(hours=12)  # hour 12
    index = small_hours.append(large_hours)

    actual = pd.Series(actual_value, index=index)
    predicted = actual.copy()
    predicted.loc[small_hours] = actual_value - error_small
    predicted.loc[large_hours] = actual_value - error_large

    breakdown = error_breakdown(actual, predicted)
    hours = breakdown[breakdown.segment == "hour"].copy()
    hours["bucket"] = hours["bucket"].astype(int)
    hours = hours.set_index("bucket")

    wape_small = error_small / actual_value
    wape_large = error_large / actual_value
    pooled_overall = (error_small * n_small + error_large * n_large) / (actual_value * (n_small + n_large))
    unweighted_mean = (wape_small + wape_large) / 2
    assert abs(pooled_overall - unweighted_mean) > 0.03, (
        "the two candidate denominators must differ substantially or this test cannot discriminate them"
    )

    assert hours.loc[0, "n"] == n_small
    assert hours.loc[12, "n"] == n_large
    assert np.isclose(hours.loc[0, "wape"], wape_small)
    assert np.isclose(hours.loc[0, "ratio_to_overall"], wape_small / pooled_overall)
