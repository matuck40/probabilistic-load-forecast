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


def test_flags_do_not_depend_on_the_window():
    # 2025-12-25 is Christmas, so 2025-12-24 is its eve no matter where the
    # window passed to calendar_frame happens to end.
    ends_at_the_eve = pd.date_range("2025-12-20", "2025-12-24 23:00", freq="h")
    ends_later = pd.date_range("2025-12-20", "2025-12-31 23:00", freq="h")
    assert calendar_frame(ends_at_the_eve).loc["2025-12-24 10:00", "is_holiday_eve"] == 1.0
    assert calendar_frame(ends_later).loc["2025-12-24 10:00", "is_holiday_eve"] == 1.0

    # 2025-12-25 is a holiday, so 2025-12-26 is the day after no matter where
    # the window passed to calendar_frame happens to start.
    starts_the_day_after = pd.date_range("2025-12-26", "2025-12-31 23:00", freq="h")
    starts_earlier = pd.date_range("2025-12-20", "2025-12-31 23:00", freq="h")
    assert calendar_frame(starts_the_day_after).loc["2025-12-26 10:00", "is_day_after_holiday"] == 1.0
    assert calendar_frame(starts_earlier).loc["2025-12-26 10:00", "is_day_after_holiday"] == 1.0


def test_first_day_of_the_study_period_follows_a_holiday():
    # 2020-12-31 (Vespera de Ano-Novo) is in the holiday set even though the
    # window below starts on 2021-01-01 and never includes it.
    index = pd.date_range("2021-01-01", "2021-01-02 23:00", freq="h")
    frame = calendar_frame(index)
    assert frame.loc["2021-01-01 10:00", "is_day_after_holiday"] == 1.0


def test_ordinary_midweek_day_has_no_eve_or_day_after_flag():
    # 2026-01-14 is a Wednesday with no holiday on either side of it.
    index = pd.date_range("2026-01-12", "2026-01-16 23:00", freq="h")
    frame = calendar_frame(index)
    assert frame.loc["2026-01-14 10:00", "is_holiday_eve"] == 0.0
    assert frame.loc["2026-01-14 10:00", "is_day_after_holiday"] == 0.0


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
