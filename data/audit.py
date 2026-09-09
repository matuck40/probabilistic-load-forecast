"""Audit the raw ONS files before any modelling depends on them.

Two questions decide whether an hourly index can be trusted:

1. Is the index regular?  A silently missing hour makes a lag of 24 rows point
   somewhere other than 24 hours back, and every lag feature built on it is then
   quietly wrong.
2. Are missing values actually visible?  The published data dictionary states
   that the load column does not allow nulls and does allow zeros. In years that
   had daylight saving time this is not true: the hour skipped by the clock
   change is stored as a null for three subsystems and as a literal ``0.0`` for
   the fourth. A zero that means "this hour did not exist" is indistinguishable
   from a valid reading unless it is checked for.

Both are reasons this project uses 2021 onward, after Brazil abolished daylight
saving time. Run ``python data/audit.py`` to reproduce the numbers.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
TIME_COL = "din_instante"
VALUE_COL = "val_cargaenergiahomwmed"
ID_COL = "id_subsistema"

# The physically plausible floor for any Brazilian subsystem, in MWmed. Anything
# at or below this is a placeholder, not a reading.
IMPLAUSIBLE_AT_OR_BELOW = 0.0


def load_raw(pattern: str = "CURVA_CARGA_*.csv") -> pd.DataFrame:
    paths = sorted(glob.glob(str(RAW_DIR / pattern)))
    if not paths:
        raise SystemExit(f"no files matching {pattern} in {RAW_DIR}. Run data/download.py first.")
    # The files are semicolon-separated; the decimal mark is a point.
    frames = [pd.read_csv(p, sep=";", parse_dates=[TIME_COL]) for p in paths]
    return pd.concat(frames, ignore_index=True)


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """One row per subsystem describing the integrity of its hourly index."""
    end = df[TIME_COL].max()
    rows = []
    for subsystem, group in df.groupby(ID_COL):
        stamps = group[TIME_COL]
        grid = pd.date_range(stamps.min(), end, freq="h")
        present = pd.DatetimeIndex(stamps.unique())
        values = group[VALUE_COL]
        rows.append({
            "subsystem": subsystem,
            "start": stamps.min(),
            "end": stamps.max(),
            "rows": len(group),
            "hours_expected": len(grid),
            "missing_hours": len(grid.difference(present)),
            "off_grid_stamps": len(present.difference(grid)),
            "duplicate_stamps": int(stamps.duplicated().sum()),
            "null_values": int(values.isna().sum()),
            "implausible_values": int((values <= IMPLAUSIBLE_AT_OR_BELOW).sum()),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the raw ONS load files.")
    parser.add_argument("--pattern", default="CURVA_CARGA_*.csv", help="glob inside data/raw/")
    args = parser.parse_args(argv)

    df = load_raw(args.pattern)
    print(f"rows: {len(df):,}    period: {df[TIME_COL].min()} -> {df[TIME_COL].max()}")

    # The dictionary changed the contents of nom_subsistema in v1.2 (2026-04-06),
    # so the stable key is the code, never the name.
    names = df.groupby(ID_COL)["nom_subsistema"].unique()
    print("\nsubsystem code -> names seen in the files:")
    for code, seen in names.items():
        print(f"  {code:3} {list(seen)}")

    report = audit(df)
    print("\nindex integrity:")
    print(report.to_string(index=False))

    problems = report[
        (report.missing_hours > 0)
        | (report.duplicate_stamps > 0)
        | (report.null_values > 0)
        | (report.implausible_values > 0)
        | (report.off_grid_stamps > 0)
    ]
    if not problems.empty:
        print("\nFAIL: the index is not clean for", ", ".join(problems.subsystem), file=sys.stderr)
        return 1
    print("\nOK: every subsystem has a gap-free, duplicate-free hourly index with no null or implausible values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
