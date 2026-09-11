import numpy as np
import pandas as pd
import pytest

from src import features
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


def test_available_lags_excludes_lags_shorter_than_the_horizon(monkeypatch):
    """The filter itself must be exercised, independent of whatever LAGS is.

    With the real LAGS (min 1) the filter already does real work -- see
    test_feature_matrices_differ_across_distinct_lag_signatures -- but this
    test isolates the `>= horizon` rule itself from the current LAGS value by
    monkeypatching it to a small fixed tuple and checking exactly which lags
    are kept and dropped at two different horizons.
    """
    monkeypatch.setattr(features, "LAGS", (6, 24, 168))
    assert available_lags(12) == (24, 168)
    assert available_lags(1) == (6, 24, 168)


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


def test_no_leakage_check_catches_a_forward_reading_column_in_a_real_frame():
    """Certify the exact comparison above, applied to a frame of the real shape.

    `test_no_leakage_is_actually_detectable` only proves `assert_series_equal`
    can fail on a hand-built series; it never calls `build_features` and never
    uses `assert_frame_equal`. This splices one deliberately forward-reading
    column into real `build_features` output and re-runs the same
    `assert_frame_equal(..., check_exact=True)` the real leakage test performs.
    """
    series = _series()
    cut = series.index[1000]
    corrupted = series.copy()
    corrupted.loc[corrupted.index > cut] = 999999.0

    horizon = 12
    original_X, _ = build_features(series, horizon=horizon)
    corrupted_X, _ = build_features(corrupted, horizon=horizon)

    original_X = original_X.copy()
    corrupted_X = corrupted_X.copy()
    original_X["peek"] = series.shift(-1).reindex(original_X.index)
    corrupted_X["peek"] = corrupted.shift(-1).reindex(corrupted_X.index)

    rows = original_X.index[original_X.index <= cut]
    assert len(rows) > 100, "the test needs a meaningful number of rows"
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            original_X.loc[rows],
            corrupted_X.loc[rows],
            check_exact=True,
        )


def test_feature_matrices_differ_across_distinct_lag_signatures():
    """Horizons 1, 2 and 12 each drop a different set of short lags, so their
    feature matrices must be pairwise distinct, and all three must differ from
    horizon 13, which is short enough to drop every lag under 24.
    """
    series = _series()
    frames = {h: build_features(series, horizon=h)[0] for h in (1, 2, 12, 13)}
    for h_a, h_b in [(1, 2), (1, 12), (2, 12), (1, 13), (2, 13), (12, 13)]:
        assert list(frames[h_a].columns) != list(frames[h_b].columns), (
            f"horizons {h_a} and {h_b} should not share a lag signature"
        )


def test_feature_matrices_are_identical_for_horizons_13_through_24():
    """Every lag under 24 is filtered out for horizons 13 and above, and the
    only lags left (24, 168, 336, 504, 672) are all still >= 24, so horizons 13
    through 24 see exactly the same information at the forecast origin.
    """
    series = _series()
    reference, _ = build_features(series, horizon=13)
    for horizon in range(14, 25):
        X, _ = build_features(series, horizon=horizon)
        pd.testing.assert_frame_equal(reference, X, check_exact=True)


def test_available_lags_extended_range():
    assert 1 in available_lags(1)
    assert all(lag >= 24 for lag in available_lags(13))


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
