"""Calendar effects, which are known in advance and therefore legal features.

The `holidays` package ships Brazilian national holidays in its PUBLIC category
and a wider set under OPTIONAL. Four OPTIONAL days behave like holidays for
electricity load and are included; two do not and are excluded. The selection is
made on domain grounds, not fitted.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import holidays
import pandas as pd
from holidays.constants import OPTIONAL, PUBLIC

# Legally "ponto facultativo" but observed widely enough to move load.
OBSERVED_OPTIONAL = frozenset({"Carnaval", "Corpus Christi", "Véspera de Natal", "Véspera de Ano-Novo"})


def holiday_dates(years: Iterable[int]) -> set[dt.date]:
    """National public holidays plus the four observed optional days."""
    years = list(years)
    calendar = holidays.country_holidays("BR", years=years, categories=(PUBLIC, OPTIONAL))
    public = holidays.country_holidays("BR", years=years, categories=(PUBLIC,))
    selected = set()
    for date, name in calendar.items():
        if date in public or name in OBSERVED_OPTIONAL:
            selected.add(date)
    return selected


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Four day-level flags, broadcast onto an hourly index."""
    # Widened so that the padding day on each side of `days` (below) always
    # falls inside a year the holidays library was actually asked about.
    years = range(index.min().year - 1, index.max().year + 2)
    dates = holiday_dates(years)

    # index.normalize() is vectorised; a Python set comprehension over the
    # 49,848 timestamps would run 24 times per fit and dominate the runtime.
    days = pd.DatetimeIndex(index.normalize().unique()).sort_values()

    # A flag must be a function of the date alone, never of where the window
    # happens to start or end. Compute on a domain padded by one day on each
    # side, so shift() always has the real neighbouring day to look at, and
    # only select back down to `days` at the end. fill_value=False now only
    # ever lands on the padding rows, which are discarded below.
    padded = pd.date_range(days.min() - pd.Timedelta(days=1), days.max() + pd.Timedelta(days=1), freq="D")
    day_frame = pd.DataFrame(index=padded)
    is_holiday = pd.Series([d.date() in dates for d in padded], index=padded)
    weekend = pd.Series(padded.dayofweek >= 5, index=padded)

    day_frame["is_holiday"] = is_holiday
    day_frame["is_holiday_eve"] = is_holiday.shift(-1, freq="D").reindex(padded, fill_value=False)
    day_frame["is_day_after_holiday"] = is_holiday.shift(1, freq="D").reindex(padded, fill_value=False)

    # A bridge day is an ordinary working day with a holiday on one side and a
    # weekend on the other. These days behave like holidays and are where the
    # largest unexplained errors live.
    off = is_holiday | weekend
    # shift(-1, freq="D") moves each day's value one day earlier, so at day D the
    # result holds off[D+1]. Naming these by what they hold, not by the shift.
    next_is_off = off.shift(-1, freq="D").reindex(padded, fill_value=False)
    previous_is_off = off.shift(1, freq="D").reindex(padded, fill_value=False)
    day_frame["is_bridge_day"] = (~off) & next_is_off & previous_is_off

    day_frame = day_frame.loc[days]

    frame = day_frame.astype("float64").reindex(index.normalize())
    frame.index = index
    return frame
