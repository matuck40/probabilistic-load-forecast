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


def test_fits_one_model_per_distinct_lag_signature_and_quantile(fitted):
    """Horizons whose available lags coincide are the same learning problem.

    With LAGS extended to include 1-12, horizons 1-12 each drop a different
    set of short lags (12 distinct signatures) and horizons 13-24 all reduce
    to the same signature (24, 168, 336, 504, 672), since every lag under 24
    is filtered out and the rest are already >= 24. That is 13 signatures, not
    24 horizons, so the fit count should reflect the smaller number.
    """
    _, model = fitted
    signatures = set(model.signature_of.values())
    assert len(signatures) == 13
    for horizon in config.HORIZONS:
        assert model.model_for(horizon, "point") is not None
        for alpha in config.QUANTILES:
            assert model.model_for(horizon, f"q{alpha}") is not None
    assert len(model.models) == 13 * (1 + len(config.QUANTILES))


def test_horizons_sharing_a_signature_share_the_same_fitted_model(fitted):
    _, model = fitted
    assert model.model_for(13, "point") is model.model_for(24, "point")
    assert model.model_for(1, "point") is not model.model_for(2, "point")


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


class _ConstantModel:
    """A stand-in fitted model that always predicts one fixed value.

    Used only to pin down routing: since it ignores its input entirely, the
    only way its value can show up at a given hour is through predict's own
    hour -> horizon -> signature -> model lookup, so an off-by-one there
    changes which constant comes out.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype="float64")


def test_predict_routes_each_hour_to_its_own_horizon_model():
    """Pin the hour -> horizon -> signature -> model chain end to end.

    Horizons that share a signature share a model object (see
    test_horizons_sharing_a_signature_share_the_same_fitted_model), so a
    stand-in keyed by horizon would silently collapse for horizons 13-24.
    Keying each stand-in by signature instead, with a value equal to the
    smallest horizon in that signature's group, gives every hour an
    unambiguous expected value: an off-by-one in predict's `hour - 1`
    recovery would pull a different signature's constant, which this test
    would catch and the old horizon-keyed dict could not, because with the
    pre-fix LAGS every horizon shared one signature and the shift was
    unobservable.

    Uses its own freshly fit model rather than the shared `fitted` fixture,
    since the models dict is mutated below and the fixture is reused by
    other tests in this module.
    """
    series = _series()
    train = series.iloc[:5000]
    model = LightGBMForecaster(objective="l1", params={"n_estimators": 10, "num_leaves": 7})
    model.fit(train)

    # _frame sorts each row's three quantile values ascending (so quantiles
    # cannot cross), which would otherwise scramble a lone tiny "point"
    # constant to the low end of the row. Give q0.1/point/q0.9 constants that
    # are already in ascending order for the same horizon, so the sort is a
    # no-op and "q0.5" is exactly the point stand-in's value.
    representative: dict[tuple[int, ...], int] = {}
    for horizon in config.HORIZONS:
        representative.setdefault(model.signature_of[horizon], horizon)
    for signature, horizon in representative.items():
        model.models[(signature, "q0.1")] = _ConstantModel(float(horizon) - 0.1)
        model.models[(signature, "point")] = _ConstantModel(float(horizon))
        model.models[(signature, "q0.9")] = _ConstantModel(float(horizon) + 0.1)

    test_index = series.index[5000:5048]
    predictions = model.predict(series, test_index)

    for horizon in config.HORIZONS:
        hours = test_index[test_index.hour == horizon - 1]
        if len(hours) == 0:
            continue
        expected = float(representative[model.signature_of[horizon]])
        assert (predictions.loc[hours, "q0.5"] == expected).all(), (
            f"horizon {horizon} (hour {horizon - 1}) did not route to its own model"
        )
