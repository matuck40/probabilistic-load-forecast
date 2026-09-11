import numpy as np
import pandas as pd
import pytest

from src import config, features
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
    """Horizons whose full signature coincides are the same learning problem.

    With LAGS extended to include 1-12, horizons 1-12 each drop a different
    set of short lags (12 distinct signatures) and horizons 13-24 all reduce
    to the same signature, since every lag under 24 is filtered out of all
    three lag lists and the rest are already >= 24. That is 13 signatures,
    not 24 horizons, so the fit count should reflect the smaller number.

    The alpha=0.5 quantile model is not fit at all: predict always reports
    the point model's output as the median (its final overwrite of "q0.5"
    discards whatever a quantile model there would have produced), so a
    quantile model at alpha 0.5 would only ever be thrown away.
    """
    _, model = fitted
    signatures = set(model.signature_of.values())
    assert len(signatures) == 13
    for horizon in config.HORIZONS:
        assert model.model_for(horizon, "point") is not None
        for alpha in config.QUANTILES:
            if alpha == 0.5:
                assert (model.signature_of[horizon], "q0.5") not in model.models
                continue
            assert model.model_for(horizon, f"q{alpha}") is not None
    fitted_targets_per_signature = 1 + sum(1 for alpha in config.QUANTILES if alpha != 0.5)
    assert len(model.models) == 13 * fitted_targets_per_signature


def test_horizons_sharing_a_signature_share_the_same_fitted_model(fitted):
    _, model = fitted
    assert model.model_for(13, "point") is model.model_for(24, "point")
    assert model.model_for(1, "point") is not model.model_for(2, "point")


def test_signature_reflects_same_hour_7d_lags_not_just_the_main_lags(monkeypatch):
    """A signature built from the main LAGS alone would miss this case.

    `mean_same_hour_7d` keeps the same column name no matter which lags feed
    its average, so two horizons whose SAME_HOUR_7D_LAGS differ could still
    look identical to a signature that only tracks the main lags. Monkeypatch
    a lag into the middle of the 13-23 gap -- where horizons 13 and 24
    currently share a signature -- and check that the fix actually splits
    them: horizon 13 still sees the new lag (20 >= 13) but horizon 24 does
    not (20 < 24).
    """
    monkeypatch.setattr(features, "SAME_HOUR_7D_LAGS", (20, 24, 48, 72, 96, 120, 144, 168))
    model = LightGBMForecaster(objective="l1")
    assert model.signature_of[13] != model.signature_of[24]


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


def test_predict_raises_on_a_column_mismatch_with_the_fitted_model():
    """Prove predict's column guard actually fires on a genuine mismatch.

    The guard exists for a future features.py change that alters
    build_features's *column names* for a horizon without signature_for
    noticing. The most direct real-world version of that would seem to be
    monkeypatching features.SAME_HOUR_7D_LAGS between fit and predict, but
    that specifically does NOT trip the guard -- confirmed by trying it
    before writing this test: SAME_HOUR_7D_LAGS only feeds the single
    "mean_same_hour_7d" aggregate column, whose *name* is the same no matter
    which lags feed it (this is exactly fix round 2's Finding 1, and now the
    module docstring's maintenance note). Changing that lag list changes the
    column's values, not build_features's `tuple(X.columns)`, so the guard --
    which compares column names, not values -- has nothing to catch. That
    is the guard's documented limitation, not a bug to fix here.

    So this test stands in for the case the guard *can* catch instead: a
    future build_features change that does alter the column names for one
    signature (a reordered/renamed/added column not driven by any of the
    three lag tuples) without signature_for or feature_columns noticing.
    Simulated directly here by corrupting `feature_columns` for horizon 1's
    signature after a real fit, since reproducing an actual such
    features.py change is out of scope for this round.
    """
    series = _series()
    train = series.iloc[:5000]
    model = LightGBMForecaster(objective="l1", params={"n_estimators": 10, "num_leaves": 7})
    model.fit(train)

    horizon = 1
    signature = model.signature_of[horizon]
    trained_columns = model.feature_columns[signature]
    model.feature_columns[signature] = (*trained_columns[:-1], "a_column_predict_will_never_build")

    test_index = series.index[5000:5024]
    with pytest.raises(AssertionError, match=f"horizon {horizon}"):
        model.predict(series, test_index)
