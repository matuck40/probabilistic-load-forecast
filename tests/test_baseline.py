import numpy as np
import pandas as pd

from src.models.baseline import SeasonalNaive


def _series(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n, freq="h")
    hour = index.hour.to_numpy()
    return pd.Series(
        30000 + 5000 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 300, n),
        index=index,
        name="load",
    )


def test_point_forecast_is_the_value_one_lag_earlier():
    series = _series()
    train = series.iloc[:1500]
    test_index = series.index[1500:1548]
    model = SeasonalNaive(lag=24)
    model.fit(train)
    predictions = model.predict(series, test_index)
    expected = series.shift(24).reindex(test_index)
    pd.testing.assert_series_equal(
        predictions["q0.5"].rename("load"), expected.rename("load"), check_exact=False
    )


def test_returns_the_three_quantile_columns_ordered():
    series = _series()
    test_index = series.index[1500:1548]
    model = SeasonalNaive(lag=168)
    model.fit(series.iloc[:1500])
    predictions = model.predict(series, test_index)
    assert list(predictions.columns) == ["q0.1", "q0.5", "q0.9"]
    assert (predictions["q0.1"] <= predictions["q0.5"]).all()
    assert (predictions["q0.5"] <= predictions["q0.9"]).all()
    expected = series.shift(168).reindex(test_index)
    pd.testing.assert_series_equal(
        predictions["q0.5"].rename("load"), expected.rename("load"), check_exact=False
    )


def test_interval_width_comes_from_training_residuals_only():
    series = _series()
    model = SeasonalNaive(lag=24)
    model.fit(series.iloc[:1500])
    narrow = model.residual_quantiles

    noisy = series.copy()
    noisy.iloc[1500:] = noisy.iloc[1500:] + 50000  # corrupt the future only
    other = SeasonalNaive(lag=24)
    other.fit(noisy.iloc[:1500])
    assert other.residual_quantiles == narrow, "the future must not widen the interval"


def test_predict_interval_offsets_come_only_from_training():
    """fit() being pure is not enough: predict() is called with the full series
    every time (LightGBM and Chronos both follow this contract), so the guarantee
    that only the training window shapes the interval has to hold at predict's
    boundary too, not just at fit's.
    """
    series = _series()
    train = series.iloc[:1500]
    test_index = series.index[1500:1548]

    model = SeasonalNaive(lag=24)
    model.fit(train)
    clean_predictions = model.predict(series, test_index)
    clean_low_offset = clean_predictions["q0.5"] - clean_predictions["q0.1"]
    clean_high_offset = clean_predictions["q0.9"] - clean_predictions["q0.5"]

    corrupted = series.copy()
    corrupted.loc[test_index] = corrupted.loc[test_index] + 50000  # corrupt the test period only

    other = SeasonalNaive(lag=24)
    other.fit(train)  # same clean training window
    corrupted_predictions = other.predict(corrupted, test_index)
    corrupted_low_offset = corrupted_predictions["q0.5"] - corrupted_predictions["q0.1"]
    corrupted_high_offset = corrupted_predictions["q0.9"] - corrupted_predictions["q0.5"]

    # The point forecast may legitimately change: it reads lagged actuals, and some
    # of those lags fall inside the corrupted test period. The two offsets must not,
    # because they come from residuals frozen at fit() time on the training window.
    pd.testing.assert_series_equal(
        clean_low_offset.rename("offset"), corrupted_low_offset.rename("offset"), check_exact=False
    )
    pd.testing.assert_series_equal(
        clean_high_offset.rename("offset"), corrupted_high_offset.rename("offset"), check_exact=False
    )


def test_predict_does_not_read_beyond_the_test_window():
    """Nothing at or before the test window may depend on what comes after it."""
    series = _series()
    train = series.iloc[:1500]
    test_index = series.index[1500:1548]

    model = SeasonalNaive(lag=24)
    model.fit(train)
    clean_predictions = model.predict(series, test_index)

    corrupted = series.copy()
    corrupted.iloc[1548:] = corrupted.iloc[1548:] + 50000  # corrupt strictly after the test window

    other = SeasonalNaive(lag=24)
    other.fit(train)
    corrupted_predictions = other.predict(corrupted, test_index)

    pd.testing.assert_frame_equal(clean_predictions, corrupted_predictions)


def test_name_identifies_the_lag():
    assert SeasonalNaive(lag=24).name == "naive_24h"
    assert SeasonalNaive(lag=168).name == "naive_168h"
