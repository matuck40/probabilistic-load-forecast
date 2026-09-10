import numpy as np
import pandas as pd
import pytest

from src.features import available_lags, build_features


def _series(n=1500, start="2021-01-01 00:00", seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq="h")
    hour = index.hour.to_numpy()
    values = 30000 + 4000 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 200, n)
    return pd.Series(values, index=index, name="load", dtype="float64")


def test_available_lags_respects_the_rule():
    for horizon in range(1, 25):
        for lag in available_lags(horizon):
            assert lag >= horizon, f"lag {lag} is not known at horizon {horizon}"


def test_features_and_target_are_aligned_and_finite():
    series = _series()
    X, y = build_features(series, horizon=1)
    assert X.index.equals(y.index)
    assert len(X) > 0
    assert np.isfinite(X.to_numpy()).all()
    assert (y == series.reindex(y.index)).all()


def test_no_leakage_future_values_cannot_change_present_features():
    """The test the whole repository exists to pass.

    Build features on a series. Then corrupt every observation strictly after a
    cut timestamp and rebuild. Every feature row whose target is at or before
    the cut must be byte-identical. If a feature peeked forward, it changes.
    """
    series = _series()
    cut = series.index[1000]

    corrupted = series.copy()
    corrupted.loc[corrupted.index > cut] = 999999.0

    for horizon in (1, 12, 24):
        original_X, _ = build_features(series, horizon=horizon)
        corrupted_X, _ = build_features(corrupted, horizon=horizon)

        rows = original_X.index[original_X.index <= cut]
        assert len(rows) > 100, "the test needs a meaningful number of rows"
        pd.testing.assert_frame_equal(
            original_X.loc[rows],
            corrupted_X.loc[rows],
            check_exact=True,
            obj=f"features at horizon {horizon} changed when the future changed",
        )


def test_no_leakage_is_actually_detectable():
    """Prove the test above can fail.

    A deliberately leaking feature must be caught by the same comparison, so the
    passing test is evidence rather than a tautology.
    """
    series = _series()
    cut = series.index[1000]
    corrupted = series.copy()
    corrupted.loc[corrupted.index > cut] = 999999.0

    leaking_original = series.shift(-1).rename("peek")
    leaking_corrupted = corrupted.shift(-1).rename("peek")
    rows = series.index[(series.index <= cut) & (series.index >= series.index[10])]

    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(
            leaking_original.loc[rows], leaking_corrupted.loc[rows], check_exact=True
        )


def test_expected_columns_exist_and_no_trigonometric_encoding():
    X, _ = build_features(_series(), horizon=6)
    for column in [
        "lag_24",
        "lag_168",
        "lag_336",
        "mean_same_hour_dow_4w",
        "std_same_hour_dow_4w",
        "mean_same_hour_7d",
        "hour",
        "dayofweek",
        "month",
        "dayofyear",
        "is_holiday",
        "is_holiday_eve",
        "is_day_after_holiday",
        "is_bridge_day",
    ]:
        assert column in X.columns, f"missing feature {column}"
    assert not [c for c in X.columns if "sin" in c or "cos" in c], "no trigonometric encoding"
