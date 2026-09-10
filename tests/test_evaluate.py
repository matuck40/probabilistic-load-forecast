import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import aggregate, run_model, write_run_metadata
from src.models.baseline import SeasonalNaive
from src.splits import Fold


def _series(n=4000, seed=4):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n, freq="h")
    hour = index.hour.to_numpy()
    return pd.Series(
        30000 + 5000 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 300, n),
        index=index,
        name="load",
    )


def _folds(series):
    return [
        Fold(1, series.index[2000], series.index[2001], series.index[2500]),
        Fold(2, series.index[2500], series.index[2501], series.index[3000]),
    ]


def test_aggregate_returns_mean_and_std_per_metric():
    per_fold = [
        {"fold": 1, "wape": 0.10, "mae": 100.0, "rmse": 120.0, "pinball": 50.0, "coverage": 0.80},
        {"fold": 2, "wape": 0.20, "mae": 200.0, "rmse": 240.0, "pinball": 60.0, "coverage": 0.70},
    ]
    mean, std = aggregate(per_fold)
    # Approximate: a float64 mean of 0.10 and 0.20 is not bitwise equal to the
    # literal 0.15 (it lands one ULP away), so this must not depend on whether
    # aggregate happens to round its output.
    assert mean["wape"] == pytest.approx(0.15)
    assert std["wape"] > 0
    assert "fold" not in mean


def test_aggregate_rounds_to_ten_decimals_for_stable_diffs():
    """metrics.json is committed and meant to be diffed and reproduced.

    The last bit of a float64 sum can differ across CPUs and library builds,
    so aggregate() rounds its output to keep that artifact stable. This pins
    the rounding itself, separately from the mean/std values it produces.
    """
    per_fold = [
        {"fold": 1, "wape": 1 / 3, "mae": 100.0, "rmse": 120.0, "pinball": 50.0, "coverage": 0.80},
        {"fold": 2, "wape": 2 / 3, "mae": 200.0, "rmse": 240.0, "pinball": 60.0, "coverage": 0.70},
        {"fold": 3, "wape": 1 / 7, "mae": 300.0, "rmse": 360.0, "pinball": 70.0, "coverage": 0.60},
    ]
    mean, std = aggregate(per_fold)
    for value in {**mean, **std}.values():
        rounded = round(value, 10)
        assert value == rounded, f"{value!r} carries more than 10 decimal places"


def test_run_model_scores_every_fold():
    series = _series()
    result = run_model(SeasonalNaive(lag=24), series, _folds(series))
    assert len(result["per_fold"]) == 2
    assert set(result["mean"]) == {"wape", "mae", "rmse", "pinball", "coverage"}
    assert 0 < result["mean"]["wape"] < 1


def test_run_metadata_records_provenance(tmp_path):
    path = tmp_path / "run.json"
    write_run_metadata(path, _series(), ["naive_24h"])
    payload = json.loads(path.read_text())
    assert payload["seed"] == 42
    assert "pandas" in payload["library_versions"]
    assert len(payload["folds"]) == 6
    assert payload["models"] == ["naive_24h"]
