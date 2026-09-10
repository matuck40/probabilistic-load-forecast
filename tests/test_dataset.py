import pandas as pd
import pytest

from src import config
from src.dataset import load_series, validate_series


def _hourly(values, start="2021-01-01 00:00"):
    index = pd.date_range(start, periods=len(values), freq="h")
    return pd.Series(values, index=index, name="load", dtype="float64")


def test_validate_accepts_a_clean_series():
    validate_series(_hourly([10.0, 11.0, 12.0, 13.0]))


def test_validate_rejects_a_gap():
    clean = _hourly([10.0, 11.0, 12.0, 13.0])
    with_gap = clean.drop(clean.index[2])
    with pytest.raises(ValueError, match="missing"):
        validate_series(with_gap)


def test_validate_rejects_duplicates():
    clean = _hourly([10.0, 11.0, 12.0])
    duplicated = pd.concat([clean, clean.iloc[[1]]]).sort_index()
    with pytest.raises(ValueError, match="duplicate"):
        validate_series(duplicated)


def test_validate_rejects_nulls():
    with pytest.raises(ValueError, match="null"):
        validate_series(_hourly([10.0, float("nan"), 12.0]))


def test_validate_rejects_non_positive():
    # A zero here is the daylight-saving placeholder, not a reading.
    with pytest.raises(ValueError, match="non-positive"):
        validate_series(_hourly([10.0, 0.0, 12.0]))


@pytest.mark.integration
def test_load_series_returns_the_expected_window():
    series = load_series()
    assert series.index[0] == config.PERIOD_START
    assert series.index[-1] == config.PERIOD_END
    assert len(series) == 49848
    assert pd.infer_freq(series.index) == "h"
    assert series.notna().all()
    assert (series > 0).all()
