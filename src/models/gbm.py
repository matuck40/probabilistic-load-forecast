"""LightGBM with one model per distinct lag signature, not per horizon.

Horizons whose main lags, same-hour-dow lags, same-hour-7d lags, and ramp lags
all agree see exactly the same information at the forecast origin and share
one fitted model; see `LightGBMForecaster.signature_for` and `.fit`.

Maintenance note: `signature_for` is built from those four lag tuples alone,
so a future column added to `build_features` that is not driven by one of
them must be reflected in `signature_for` too -- `predict`'s column-name
guard cannot catch that omission on its own, because an aggregate column's
name does not change just because the lags feeding it did.

The loss function matters more than the hyperparameters here. Training in MW
under squared error spends the model's capacity on the high-load hours; the
metric that is reported is WAPE, so absolute error is the closer match. Both are
run and both are reported.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from src import config, features
from src.features import available_lags, build_features
from src.models.base import Forecaster

DEFAULT_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
    "verbose": -1,
    # Without this, LightGBM's multithreaded histogram build can reorder
    # floating-point sums and give slightly different trees between runs, which
    # would make the fixed seed a lie.
    "deterministic": True,
    "force_col_wise": True,
}

# (main lags, same-hour-dow lags, same-hour-7d lags, ramp lags), each already
# filtered by the `L >= h` rule for one horizon. All four matter: build_features
# folds the last three into aggregate columns ("mean_same_hour_dow_4w",
# "std_same_hour_dow_4w", "mean_same_hour_7d", "ramp_168") whose names never
# change no matter which lags feed them, so a signature built from the main
# lags alone could merge two horizons that are secretly fed different
# information.
Signature = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]


def _filter_lags(lags: tuple[int, ...], horizon: int) -> tuple[int, ...]:
    """The same `L >= h` rule as `features._usable`, duplicated here.

    `SAME_HOUR_DOW_LAGS`, `SAME_HOUR_7D_LAGS`, and `RAMP_LAGS` have no public
    accessor of their own (unlike `LAGS`, via `available_lags`), and this file
    may not add one this round. The rule is one line, so duplicating it is
    cheap; reading `features.SAME_HOUR_DOW_LAGS`/`features.SAME_HOUR_7D_LAGS`/
    `features.RAMP_LAGS` as module attributes (rather than importing them by
    name) means a monkeypatched value is picked up here exactly as
    `build_features` itself would see it.
    """
    return tuple(lag for lag in lags if lag >= horizon)


class LightGBMForecaster(Forecaster):
    def __init__(self, objective: str = "l1", params: dict | None = None, quantile: bool = True) -> None:
        self.objective = objective
        self.quantile = quantile
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.name = f"lightgbm_{objective}"
        # Horizons whose full signature coincides are, at the forecast origin,
        # the same learning problem: same features, same target values. Fitting
        # them separately would just retrain an identical model. This maps each
        # horizon to that signature so the grouping is inspectable and reused
        # by both fit and predict.
        self.signature_of: dict[int, Signature] = {h: self.signature_for(h) for h in config.HORIZONS}
        # signature -> the column names (in order) of the X actually used to
        # fit that signature's models, populated by fit(). predict checks a
        # prediction-time X against this before using the routed model, so a
        # signature that quietly stopped describing the true feature set
        # fails loudly instead of scoring silently.
        self.feature_columns: dict[Signature, tuple[str, ...]] = {}
        self.models: dict[tuple[Signature, str], lgb.LGBMRegressor] = {}

    def _make(self, objective: str, alpha: float | None = None) -> lgb.LGBMRegressor:
        kwargs = dict(self.params)
        kwargs["objective"] = objective
        kwargs["random_state"] = config.SEED
        kwargs["n_jobs"] = -1
        if alpha is not None:
            kwargs["alpha"] = alpha
        return lgb.LGBMRegressor(**kwargs)

    def signature_for(self, horizon: int) -> Signature:
        """The full lag signature `build_features` would use for `horizon`.

        Combines all four lag lists `build_features` filters -- `LAGS` (via
        the public `available_lags`), `SAME_HOUR_DOW_LAGS`, `SAME_HOUR_7D_LAGS`,
        and `RAMP_LAGS` -- not just the main lags, so two horizons only share a
        model when every column of `build_features`'s output would actually be
        built from the same underlying lags.
        """
        return (
            available_lags(horizon),
            _filter_lags(features.SAME_HOUR_DOW_LAGS, horizon),
            _filter_lags(features.SAME_HOUR_7D_LAGS, horizon),
            _filter_lags(features.RAMP_LAGS, horizon),
        )

    def model_for(self, horizon: int, target: str) -> lgb.LGBMRegressor:
        """The fitted model responsible for `horizon`, found via its signature."""
        return self.models[(self.signature_of[horizon], target)]

    def fit(self, train: pd.Series) -> None:
        """One model per distinct lag signature, and per quantile except 0.5.

        Horizons are visited in order and a signature is only fit the first
        time it is seen, so horizons that share a signature share the exact
        same fitted model object. The alpha=0.5 quantile model is never fit:
        predict always reports the point model's output as the median (see
        predict's docstring), so a quantile model there would only ever be
        thrown away.
        """
        self.models.clear()
        self.feature_columns.clear()
        fitted_signatures: set[Signature] = set()
        for horizon in config.HORIZONS:
            signature = self.signature_of[horizon]
            if signature in fitted_signatures:
                continue
            fitted_signatures.add(signature)
            X, y = build_features(train, horizon)
            self.feature_columns[signature] = tuple(X.columns)
            self.models[(signature, "point")] = self._make(self.objective).fit(X, y)
            if self.quantile:
                for alpha in config.QUANTILES:
                    if alpha == 0.5:
                        continue
                    key = (signature, f"q{alpha}")
                    self.models[key] = self._make("quantile", alpha=alpha).fit(X, y)

    def predict(self, series: pd.Series, test_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Predict every target hour, routing each to the model for its horizon.

        Horizon is recoverable from the target hour because the origin is always
        midnight: hour k of a day is horizon k + 1. Horizons sharing a signature
        are routed to the same fitted model, but each horizon's own features
        (built at its own horizon) are what gets passed in. Before predicting,
        the columns just built are checked against the columns the routed
        model was actually trained on, so a signature that quietly stopped
        describing the true feature set fails loudly here instead of scoring
        silently.
        """
        if not self.models:
            raise RuntimeError("call fit before predict")

        columns = {f"q{q}": pd.Series(index=test_index, dtype="float64") for q in config.QUANTILES}
        point = pd.Series(index=test_index, dtype="float64")

        for horizon in config.HORIZONS:
            targets = test_index[test_index.hour == horizon - 1]
            if len(targets) == 0:
                continue
            X, _ = build_features(series, horizon)
            X = X.reindex(targets).dropna()
            if X.empty:
                continue

            signature = self.signature_of[horizon]
            expected_columns = self.feature_columns[signature]
            actual_columns = tuple(X.columns)
            if actual_columns != expected_columns:
                raise AssertionError(
                    f"horizon {horizon}: predict-time columns {actual_columns} do "
                    f"not match the columns the routed model was trained on "
                    f"{expected_columns}"
                )

            point.loc[X.index] = self.model_for(horizon, "point").predict(X)
            if self.quantile:
                for alpha in config.QUANTILES:
                    if alpha == 0.5:
                        continue
                    columns[f"q{alpha}"].loc[X.index] = self.model_for(horizon, f"q{alpha}").predict(X)

        if not self.quantile:
            for alpha in config.QUANTILES:
                columns[f"q{alpha}"] = point
        # The point model is what predict reports as the median; there is no
        # quantile model at alpha 0.5 to compare it to (fit skips it -- see
        # fit's docstring). _frame sorts each row's three values ascending to
        # keep quantiles from crossing, so a point estimate that falls outside
        # [q0.1, q0.9] does not stay in the middle column after that sort.
        columns["q0.5"] = point
        low, mid, high = (columns[f"q{q}"] for q in config.QUANTILES)
        return self._frame(test_index, low, mid, high)
