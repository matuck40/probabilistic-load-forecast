import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    empirical_coverage,
    evaluate_predictions,
    mae,
    pinball_loss,
    rmse,
    wape,
)


def test_wape_is_total_error_over_total_actual():
    y = pd.Series([100.0, 200.0, 300.0])
    yhat = pd.Series([110.0, 180.0, 300.0])
    # |10| + |20| + |0| = 30 ; total actual = 600 ; 30/600 = 0.05
    assert wape(y, yhat) == pytest.approx(0.05)


def test_wape_is_not_the_mean_of_percentage_errors():
    y = pd.Series([1.0, 1000.0])
    yhat = pd.Series([2.0, 1000.0])
    # mean of per-point percentages would be 50%; WAPE is 1/1001
    assert wape(y, yhat) == pytest.approx(1 / 1001)


def test_mae_and_rmse():
    y = pd.Series([0.0, 0.0, 0.0])
    yhat = pd.Series([3.0, 4.0, 0.0])
    assert mae(y, yhat) == pytest.approx(7 / 3)
    assert rmse(y, yhat) == pytest.approx(np.sqrt(25 / 3))


def test_pinball_penalises_the_two_sides_asymmetrically():
    y = pd.Series([10.0])
    # For alpha=0.9 an under-forecast is penalised nine times as hard.
    under = pinball_loss(y, pd.Series([8.0]), alpha=0.9)
    over = pinball_loss(y, pd.Series([12.0]), alpha=0.9)
    assert under == pytest.approx(0.9 * 2)
    assert over == pytest.approx(0.1 * 2)
    assert under > over


def test_pinball_at_the_median_is_half_the_absolute_error():
    y = pd.Series([10.0, 20.0])
    yhat = pd.Series([12.0, 16.0])
    assert pinball_loss(y, yhat, alpha=0.5) == pytest.approx(0.5 * mae(y, yhat))


def test_empirical_coverage_counts_what_falls_inside():
    y = pd.Series([1.0, 5.0, 9.0, 11.0])
    lower = pd.Series([0.0, 0.0, 0.0, 0.0])
    upper = pd.Series([10.0, 10.0, 10.0, 10.0])
    assert empirical_coverage(y, lower, upper) == pytest.approx(0.75)


def test_evaluate_predictions_returns_every_metric():
    index = pd.date_range("2026-01-01", periods=48, freq="h")
    y = pd.Series(np.linspace(30000, 40000, 48), index=index)
    predictions = pd.DataFrame(
        {"q0.1": y - 1000, "q0.5": y + 50, "q0.9": y + 1000}, index=index
    )
    result = evaluate_predictions(y, predictions)
    assert set(result) == {"wape", "mae", "rmse", "pinball", "coverage"}
    assert result["coverage"] == pytest.approx(1.0)
    assert result["wape"] > 0
