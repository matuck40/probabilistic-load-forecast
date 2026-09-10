import datetime as dt

import pandas as pd

from src.calendar_features import calendar_frame, holiday_dates


def test_includes_public_and_selected_optional_holidays():
    dates = holiday_dates([2026])
    assert dt.date(2026, 12, 25) in dates, "Christmas"
    assert dt.date(2026, 2, 16) in dates, "Carnaval, legally optional but observed"
    assert dt.date(2026, 6, 4) in dates, "Corpus Christi"
    assert dt.date(2026, 12, 24) in dates, "Christmas Eve"
    assert dt.date(2026, 12, 31) in dates, "New Year's Eve"


def test_excludes_optional_days_that_are_not_observed():
    dates = holiday_dates([2026])
    assert dt.date(2026, 2, 18) not in dates, "Ash Wednesday"
    assert dt.date(2026, 10, 28) not in dates, "Public Servant Day"


def test_flags_on_christmas_2026():
    # 2026-12-25 is a Friday. 12-24 is its eve and is itself a holiday.
    index = pd.date_range("2026-12-23", "2026-12-26 23:00", freq="h")
    frame = calendar_frame(index)
    assert frame.loc["2026-12-25 10:00", "is_holiday"] == 1.0
    assert frame.loc["2026-12-23 10:00", "is_holiday_eve"] == 1.0
    assert frame.loc["2026-12-26 10:00", "is_day_after_holiday"] == 1.0


def test_bridge_day_is_a_workday_between_a_holiday_and_a_weekend():
    # 2026-04-21 (Tiradentes) is a Tuesday, so Monday 2026-04-20 is a bridge day
    # sitting between the weekend and the holiday.
    index = pd.date_range("2026-04-18", "2026-04-22 23:00", freq="h")
    frame = calendar_frame(index)
    assert frame.loc["2026-04-20 10:00", "is_bridge_day"] == 1.0
    assert frame.loc["2026-04-22 10:00", "is_bridge_day"] == 0.0


def test_columns_are_float_and_aligned():
    index = pd.date_range("2026-01-01", periods=48, freq="h")
    frame = calendar_frame(index)
    assert list(frame.columns) == [
        "is_holiday",
        "is_holiday_eve",
        "is_day_after_holiday",
        "is_bridge_day",
    ]
    assert frame.index.equals(index)
    assert all(frame.dtypes == "float64")
