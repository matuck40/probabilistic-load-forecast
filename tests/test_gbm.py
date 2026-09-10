import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.gbm import LightGBMForecaster


def _series(n=6000, seed=2):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n, freq="h")
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    values = (
        30000
        + 5000 * np.sin(2 * np.pi * hour / 24)
        - 2000 * (dow >= 5)
        + rng.normal(0, 300, n)
    )
    return pd.Series(values, index=index, name="load")


@pytest.fixture(scope="module")
def fitted():
    series = _series()
    train = series.iloc[:5000]
    model = LightGBMForecaster(objective="l1", params={"n_estimators": 40, "num_leaves": 7})
    model.fit(train)
    return series, model


def test_fits_one_model_per_horizon_and_quantile(fitted):
    _, model = fitted
    for horizon in config.HORIZONS:
        assert (horizon, "point") in model.models
        for alpha in config.QUANTILES:
            assert (horizon, f"q{alpha}") in model.models
    assert len(model.models) == 24 * (1 + len(config.QUANTILES))


def test_predictions_cover_the_test_index_with_ordered_quantiles(fitted):
    series, model = fitted
    test_index = series.index[5000:5048]
    predictions = model.predict(series, test_index)
    assert predictions.index.equals(test_index)
    assert list(predictions.columns) == ["q0.1", "q0.5", "q0.9"]
    assert predictions.notna().all().all()
    assert (predictions["q0.1"] <= predictions["q0.9"]).all()


def test_beats_a_constant_forecast(fitted):
    series, model = fitted
    test_index = series.index[5000:5048]
    predictions = model.predict(series, test_index)
    actual = series.reindex(test_index)
    model_error = np.abs(actual - predictions["q0.5"]).mean()
    constant_error = np.abs(actual - series.iloc[:5000].mean()).mean()
    assert model_error < constant_error


def test_is_deterministic_under_a_fixed_seed():
    series = _series()
    train = series.iloc[:5000]
    test_index = series.index[5000:5024]
    first = LightGBMForecaster(objective="l1", params={"n_estimators": 30, "num_leaves": 7})
    second = LightGBMForecaster(objective="l1", params={"n_estimators": 30, "num_leaves": 7})
    first.fit(train)
    second.fit(train)
    pd.testing.assert_frame_equal(first.predict(series, test_index), second.predict(series, test_index))
