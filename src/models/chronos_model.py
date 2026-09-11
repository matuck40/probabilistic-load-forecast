"""A zero-shot foundation model, for the experiment the repository is about.

Chronos was pretrained on a large corpus of time series and has never seen this
one. Nothing here fine-tunes it. The question is whether a model with no
knowledge of Brazilian load can compete with a tree trained on five years of it.
If it cannot, that is the result and it gets written down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.models.base import Forecaster

# One forecast per configured horizon, not a second hardcoded 24: HORIZONS is
# already the one place that number lives (see config.py's own warning about
# constants that appear twice eventually disagreeing with themselves).
PREDICTION_LENGTH = len(config.HORIZONS)


class ChronosForecaster(Forecaster):
    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-small",
        context_length: int = 512,
        num_samples: int = 20,
    ) -> None:
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples
        self.name = "chronos_zeroshot"
        self.fitted_on_target_data = False
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            # Imported here, not at module level: torch and LightGBM cannot share a
            # process on this machine (two independent OpenMP runtimes segfault), so
            # merely importing this module must not pull torch in.
            import torch
            from chronos import BaseChronosPipeline

            torch.manual_seed(config.SEED)
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map="cpu", torch_dtype=torch.float32
            )
        return self._pipeline

    def fit(self, train: pd.Series) -> None:
        """Load the weights. Deliberately does not learn from `train`."""
        self._load()
        self.fitted_on_target_data = False

    def predict(self, series: pd.Series, test_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Forecast day by day, each from history strictly before its origin.

        Each day's targets must themselves be a contiguous run of hours
        starting at that day's 00:00: the model's forecast step k is labelled
        origin + k hours, so a test_index that starts a day at, say, 06:00
        would otherwise be silently mislabelled (still conservative, since it
        would only read history before the true origin, but wrong).
        """
        days = sorted({stamp.normalize() for stamp in test_index})
        day_targets = []
        for day in days:
            targets = test_index[(test_index >= day) & (test_index < day + pd.Timedelta(days=1))]
            expected = pd.date_range(day, periods=len(targets), freq="h")
            mismatched = targets.to_numpy() != expected.to_numpy()
            if mismatched.any():
                offending = targets[int(np.argmax(mismatched))]
                raise ValueError(
                    f"{offending} does not fit a contiguous run of hours starting at "
                    f"{day}'s 00:00 origin; refusing to guess a label for it"
                )
            day_targets.append((day, targets))

        import torch  # lazy: see the comment in _load
        from chronos import ForecastType

        pipeline = self._load()
        # num_samples only means something for a sampling pipeline (ForecastType.SAMPLES,
        # e.g. the default chronos-t5-small). The designated faster substitute,
        # amazon/chronos-bolt-small, predicts quantiles directly (ForecastType.QUANTILES)
        # and its predict() has no such parameter -- passing it would raise a TypeError.
        samples_based = pipeline.forecast_type == ForecastType.SAMPLES
        predict_kwargs = {"num_samples": self.num_samples} if samples_based else {}
        pieces = []

        for origin, targets in day_targets:
            history = series.loc[series.index < origin].to_numpy()[-self.context_length :]

            torch.manual_seed(config.SEED)
            # chronos-forecasting 2.3.2's BaseChronosPipeline.predict_quantiles takes the
            # context tensor as the positional/keyword argument `inputs`, not `context`
            # (see chronos/base.py and chronos/chronos.py).
            quantiles, _ = pipeline.predict_quantiles(
                inputs=torch.tensor(history, dtype=torch.float32).unsqueeze(0),
                prediction_length=PREDICTION_LENGTH,
                quantile_levels=list(config.QUANTILES),
                **predict_kwargs,
            )
            values = np.asarray(quantiles[0])[: len(targets)]
            pieces.append(pd.DataFrame(values, index=targets, columns=[f"q{q}" for q in config.QUANTILES]))

        frame = pd.concat(pieces).reindex(test_index)
        low, mid, high = (frame[f"q{q}"] for q in config.QUANTILES)
        return self._frame(test_index, low, mid, high)
