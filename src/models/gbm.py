"""LightGBM with one model per horizon.

The loss function matters more than the hyperparameters here. Training in MW
under squared error spends the model's capacity on the high-load hours; the
metric that is reported is WAPE, so absolute error is the closer match. Both are
run and both are reported.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from src import config
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


class LightGBMForecaster(Forecaster):
    def __init__(self, objective: str = "l1", params: dict | None = None, quantile: bool = True) -> None:
        self.objective = objective
        self.quantile = quantile
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.name = f"lightgbm_{objective}"
        # Horizons whose available_lags() coincide are, at the forecast origin,
        # the same learning problem: same features, same target values. Fitting
        # them separately would just retrain an identical model. This maps each
        # horizon to that signature so the grouping is inspectable and reused
        # by both fit and predict.
        self.signature_of: dict[int, tuple[int, ...]] = {h: available_lags(h) for h in config.HORIZONS}
        self.models: dict[tuple[tuple[int, ...], str], lgb.LGBMRegressor] = {}

    def _make(self, objective: str, alpha: float | None = None) -> lgb.LGBMRegressor:
        kwargs = dict(self.params)
        kwargs["objective"] = objective
        kwargs["random_state"] = config.SEED
        kwargs["n_jobs"] = -1
        if alpha is not None:
            kwargs["alpha"] = alpha
        return lgb.LGBMRegressor(**kwargs)

    def model_for(self, horizon: int, target: str) -> lgb.LGBMRegressor:
        """The fitted model responsible for `horizon`, found via its signature."""
        return self.models[(self.signature_of[horizon], target)]

    def fit(self, train: pd.Series) -> None:
        """One model per distinct lag signature, and per quantile.

        Horizons are visited in order and a signature is only fit the first
        time it is seen, so horizons that share a signature share the exact
        same fitted model object.
        """
        self.models.clear()
        fitted_signatures: set[tuple[int, ...]] = set()
        for horizon in config.HORIZONS:
            signature = self.signature_of[horizon]
            if signature in fitted_signatures:
                continue
            fitted_signatures.add(signature)
            X, y = build_features(train, horizon)
            self.models[(signature, "point")] = self._make(self.objective).fit(X, y)
            if self.quantile:
                for alpha in config.QUANTILES:
                    key = (signature, f"q{alpha}")
                    self.models[key] = self._make("quantile", alpha=alpha).fit(X, y)

    def predict(self, series: pd.Series, test_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Predict every target hour, routing each to the model for its horizon.

        Horizon is recoverable from the target hour because the origin is always
        midnight: hour k of a day is horizon k + 1. Horizons sharing a signature
        are routed to the same fitted model, but each horizon's own features
        (built at its own horizon) are what gets passed in.
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
            point.loc[X.index] = self.model_for(horizon, "point").predict(X)
            if self.quantile:
                for alpha in config.QUANTILES:
                    columns[f"q{alpha}"].loc[X.index] = self.model_for(horizon, f"q{alpha}").predict(X)

        if not self.quantile:
            for alpha in config.QUANTILES:
                columns[f"q{alpha}"] = point
        # The point model is the reported median; the quantile model at 0.5 is
        # kept only for the interval, so the two never disagree in the table.
        columns["q0.5"] = point
        low, mid, high = (columns[f"q{q}"] for q in config.QUANTILES)
        return self._frame(test_index, low, mid, high)
