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
    model = SeasonalNaive(lag=168)
    model.fit(series.iloc[:1500])
    predictions = model.predict(series, series.index[1500:1548])
    assert list(predictions.columns) == ["q0.1", "q0.5", "q0.9"]
    assert (predictions["q0.1"] <= predictions["q0.5"]).all()
    assert (predictions["q0.5"] <= predictions["q0.9"]).all()


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


def test_name_identifies_the_lag():
    assert SeasonalNaive(lag=24).name == "naive_24h"
    assert SeasonalNaive(lag=168).name == "naive_168h"
