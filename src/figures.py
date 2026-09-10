"""Exactly two figures, both generated, never pasted.

One shows whether the interval is honest on a real week. The other shows where
each model breaks across the day. A third figure would be decoration.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.dataset import load_series  # noqa: E402
from src.error_analysis import error_breakdown  # noqa: E402
from src.models.baseline import SeasonalNaive  # noqa: E402
from src.models.gbm import LightGBMForecaster  # noqa: E402
from src.splits import folds  # noqa: E402

WEEK_START = pd.Timestamp("2026-06-01 00:00")
WEEK_END = pd.Timestamp("2026-06-07 23:00")


def _fitted_predictions(series: pd.Series):
    """Refit the two headline models on the last fold and predict its test window."""
    fold = folds()[-1]
    index = fold.test_index(series)
    gbm = LightGBMForecaster(objective="l1")
    gbm.fit(fold.train(series))
    naive = SeasonalNaive(lag=168)
    naive.fit(fold.train(series))
    return index, gbm.predict(series, index), naive.predict(series, index)


def figure_week(series, index, gbm_predictions) -> None:
    window = index[(index >= WEEK_START) & (index <= WEEK_END)]
    actual = series.reindex(window)
    predictions = gbm_predictions.reindex(window)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(window, predictions["q0.1"], predictions["q0.9"], alpha=0.25, label="P10-P90")
    ax.plot(window, predictions["q0.5"], linewidth=1.6, label="LightGBM P50")
    ax.plot(window, actual, linewidth=1.2, linestyle="--", label="observed")
    ax.set_ylabel("load (MWmed)")
    ax.set_title(
        f"Subsystem {config.SUBSYSTEM}, day-ahead forecast, {WEEK_START.date()} to {WEEK_END.date()}"
    )
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "week.png", dpi=150)
    plt.close(fig)


def figure_error_by_hour(series, index, gbm_predictions, naive_predictions) -> None:
    actual = series.reindex(index)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for label, predictions in (("LightGBM", gbm_predictions), ("Seasonal naive 168h", naive_predictions)):
        breakdown = error_breakdown(actual, predictions["q0.5"])
        hours = breakdown[breakdown.segment == "hour"].sort_values("bucket")
        ax.plot(hours["bucket"], 100 * hours["wape"], marker="o", markersize=3, label=label)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("WAPE (%)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Where each model breaks across the day")
    ax.legend()
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "error_by_hour.png", dpi=150)
    plt.close(fig)


def render_all() -> None:
    series = load_series()
    index, gbm_predictions, naive_predictions = _fitted_predictions(series)
    figure_week(series, index, gbm_predictions)
    figure_error_by_hour(series, index, gbm_predictions, naive_predictions)
    print(f"wrote {config.FIGURES_DIR / 'week.png'} and {config.FIGURES_DIR / 'error_by_hour.png'}")
