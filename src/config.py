"""Every value that must not drift between modules or runs.

A constant defined here appears once. A number that appears in two files is a
number that will eventually disagree with itself.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Column names confirmed against the ONS data dictionary before any parser was
# written. nom_subsistema is deliberately absent: its contents changed in
# dictionary v1.2 (2026-04-06), so the code keys on the stable code column.
TIME_COL = "din_instante"
VALUE_COL = "val_cargaenergiahomwmed"
ID_COL = "id_subsistema"

SUBSYSTEM = "SE"
PERIOD_START = pd.Timestamp("2021-01-01 00:00")
PERIOD_END = pd.Timestamp("2026-09-08 23:00")

# The forecast is issued at 00:00 of day D. The last observation available is
# 23:00 of day D-1, so horizon h targets 00:00 + (h-1) hours of day D.
HORIZONS = tuple(range(1, 25))

QUANTILES = (0.1, 0.5, 0.9)
NOMINAL_COVERAGE = QUANTILES[2] - QUANTILES[0]

SEED = 42

# (train_end, test_start, test_end). The training window expands; the test
# window rolls forward one quarter at a time. 2026Q3 is excluded because ONS
# revises recently published data.
FOLDS = (
    ("2024-12-31 23:00", "2025-01-01 00:00", "2025-03-31 23:00"),
    ("2025-03-31 23:00", "2025-04-01 00:00", "2025-06-30 23:00"),
    ("2025-06-30 23:00", "2025-07-01 00:00", "2025-09-30 23:00"),
    ("2025-09-30 23:00", "2025-10-01 00:00", "2025-12-31 23:00"),
    ("2025-12-31 23:00", "2026-01-01 00:00", "2026-03-31 23:00"),
    ("2026-03-31 23:00", "2026-04-01 00:00", "2026-06-30 23:00"),
)
