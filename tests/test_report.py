import pandas as pd
import pytest

from src.figures import figure_error_by_hour, figure_week
from src.report import metrics_table, winner

# "lightgbm_l1" sorts before "naive_168h" alphabetically, but is given the
# HIGHER wape here on purpose: a table or winner that sorted by model name
# instead of by wape would pick lightgbm_l1 first, and these tests would
# catch that. (See the round-1 fix note: the original fixture had both
# orders agree, so a name sort could hide behind these tests undetected.)
METRICS = {
    "naive_168h": {
        "mean": {"wape": 0.0388, "mae": 1600.0, "rmse": 2200.0, "pinball": 700.0, "coverage": 0.77},
        "std": {"wape": 0.003, "mae": 70.0, "rmse": 100.0, "pinball": 30.0, "coverage": 0.04},
    },
    "lightgbm_l1": {
        "mean": {"wape": 0.0512, "mae": 2100.0, "rmse": 2900.0, "pinball": 900.0, "coverage": 0.81},
        "std": {"wape": 0.004, "mae": 90.0, "rmse": 120.0, "pinball": 40.0, "coverage": 0.03},
    },
}


def test_winner_is_the_lowest_wape():
    name, value = winner(METRICS)
    assert name == "naive_168h"
    assert value == 0.0388


def test_table_has_one_row_per_model_and_shows_dispersion():
    table = metrics_table(METRICS)
    lines = [line for line in table.splitlines() if line.startswith("|")]
    assert len(lines) == 4, "header, separator, two models"
    assert "naive_168h" in table
    assert "±" in table, "the standard deviation must be visible"


def test_table_is_sorted_by_wape():
    table = metrics_table(METRICS)
    assert table.index("naive_168h") < table.index("lightgbm_l1")


def _synthetic_series(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(1000.0, index=index, name="load")


def _synthetic_predictions(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"q0.1": 900.0, "q0.5": 1000.0, "q0.9": 1100.0}, index=index)


def test_figure_week_raises_when_the_window_is_not_in_the_index():
    # Entirely in 2025, so it can never contain figures.WEEK_START/WEEK_END (2026-06).
    index = pd.date_range("2025-01-01", periods=48, freq="h", name="din_instante")
    series = _synthetic_series(index)
    predictions = _synthetic_predictions(index)
    with pytest.raises(ValueError, match="does not overlap"):
        figure_week(series, index, predictions)


def test_figure_error_by_hour_raises_on_an_empty_index():
    index = pd.DatetimeIndex([], name="din_instante")
    series = pd.Series([], index=index, dtype="float64", name="load")
    predictions = pd.DataFrame({"q0.1": [], "q0.5": [], "q0.9": []}, index=index)
    with pytest.raises(ValueError, match="empty"):
        figure_error_by_hour(series, index, predictions, predictions)
