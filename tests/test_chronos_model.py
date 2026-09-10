import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.chronos_model import ChronosForecaster


def _series(n=1200, seed=3):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n, freq="h")
    hour = index.hour.to_numpy()
    return pd.Series(
        30000 + 5000 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 300, n),
        index=index,
        name="load",
    )


@pytest.fixture(scope="module")
def model():
    # Deferred to fixture time, not module import time: `chronos` pulls torch in
    # transitively (chronos/base.py does `import torch`), and torch must never load
    # merely because this test module was collected -- see
    # test_importing_the_module_does_not_load_torch below.
    pytest.importorskip("chronos")
    forecaster = ChronosForecaster(context_length=256, num_samples=8)
    forecaster.fit(_series().iloc[:1000])
    return forecaster


def test_importing_the_module_does_not_load_torch():
    """Torch and LightGBM segfault when both are loaded in one process (two
    independent OpenMP runtimes). The default (non-slow) suite runs LightGBM's
    tests too, so importing this module -- which every test file does at
    collection time, before any -m marker filtering -- must never pull torch in.
    Runs in a real subprocess so it isn't polluted by whatever this process,
    or an earlier test, has already imported.
    """
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from src.models.chronos_model import ChronosForecaster\n"
            "print('torch' in sys.modules)\n",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", result.stderr


@pytest.mark.slow
def test_returns_ordered_quantiles_for_one_day(model):
    series = _series()
    test_index = series.index[1008:1032]  # a full day starting at 00:00
    assert test_index[0].hour == 0
    predictions = model.predict(series, test_index)
    assert predictions.index.equals(test_index)
    assert list(predictions.columns) == ["q0.1", "q0.5", "q0.9"]
    assert predictions.notna().all().all()
    assert (predictions["q0.1"] <= predictions["q0.9"]).all()


@pytest.mark.slow
def test_does_not_read_past_the_forecast_origin(model):
    series = _series()
    test_index = series.index[1008:1032]
    baseline = model.predict(series, test_index)

    corrupted = series.copy()
    corrupted.loc[corrupted.index >= test_index[0]] = 999999.0
    after = model.predict(corrupted, test_index)

    pd.testing.assert_frame_equal(baseline, after, check_exact=False, rtol=1e-6)


@pytest.mark.slow
def test_fit_does_not_train(model):
    assert model.name == "chronos_zeroshot"
    assert model.fitted_on_target_data is False
