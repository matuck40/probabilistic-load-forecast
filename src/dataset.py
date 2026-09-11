"""Turn the raw ONS files into one validated hourly series.

Everything downstream assumes a regular hourly index. That assumption is
checked here, once, loudly, rather than trusted in twelve places.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


def validate_series(series: pd.Series) -> None:
    """Raise ValueError unless the series is usable as an hourly time series.

    A lag feature shifts by POSITION. If one hour is missing, a shift of 24
    rows silently points 25 hours back and every lag built on it is wrong, with
    no error anywhere. That is why this is an exception and not a warning.
    """
    index = series.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("index must be a DatetimeIndex")
    if index.has_duplicates:
        raise ValueError(f"duplicate timestamps: {index[index.duplicated()].tolist()[:5]}")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted")
    grid = pd.date_range(index[0], index[-1], freq="h")
    missing = grid.difference(index)
    if len(missing):
        raise ValueError(f"{len(missing)} missing hours, first: {missing[0]}")
    if series.isna().any():
        raise ValueError(f"{int(series.isna().sum())} null values")
    if (series <= 0).any():
        raise ValueError(f"{int((series <= 0).sum())} non-positive values, which are placeholders")


def load_series(raw_dir: Path | None = None) -> pd.Series:
    """Load subsystem SE as a validated hourly series over the study period."""
    raw_dir = raw_dir or config.RAW_DIR
    paths = sorted(raw_dir.glob("CURVA_CARGA_*.csv"))
    if not paths:
        raise SystemExit(f"no raw files in {raw_dir}. Run data/download.py first.")

    frames = [pd.read_csv(p, sep=";", parse_dates=[config.TIME_COL]) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    # Key on the code, never the name: dictionary v1.2 changed nom_subsistema.
    df = df[df[config.ID_COL] == config.SUBSYSTEM]

    series = (
        df.set_index(config.TIME_COL)[config.VALUE_COL]
        .sort_index()
        .loc[config.PERIOD_START : config.PERIOD_END]
        .astype("float64")
        .rename("load")
    )
    validate_series(series)
    return series
