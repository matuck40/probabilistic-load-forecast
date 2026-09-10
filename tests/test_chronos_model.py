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
def test_returns_ordered_quantiles_across_three_consecutive_days(model):
    """Three days, not one: a day-by-day loop is claimed but never exercised by
    a single-day test, which an implementation that batched every day against
    one shared context would also pass."""
    series = _series()
    test_index = series.index[1008:1080]  # three full days starting at 00:00
    assert test_index[0].hour == 0
    assert len(test_index) == 72
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
def test_prediction_changes_when_the_context_is_perturbed(model):
    """The mirror of test_does_not_read_past_the_forecast_origin, and the other
    half of its safety argument: that test alone would pass trivially for a
    model that ignores `series` entirely and always returns a constant. This
    perturbs history strictly BEFORE the origin, inside the fixture's
    context_length=256-hour window, and requires the median to move. See
    test_prediction_changes_when_the_context_is_perturbed and
    test_does_not_read_past_the_forecast_origin together being run against a
    constant-returning predict() in the fix-round report: the mirror fails and
    the corruption test still passes, which is what proves the pair
    discriminates.
    """
    series = _series()
    test_index = series.index[1008:1032]
    baseline = model.predict(series, test_index)

    perturbed = series.copy()
    context_start = test_index[0] - pd.Timedelta(hours=256)
    window = (perturbed.index >= context_start) & (perturbed.index < test_index[0])
    # Large enough that no reasonable model could ignore it: the series lives
    # around 30000 +/- 5000 with noise of std 300, so +50000 dwarfs both.
    perturbed.loc[window] = perturbed.loc[window] + 50000.0
    after = model.predict(perturbed, test_index)

    assert not np.allclose(baseline["q0.5"].to_numpy(), after["q0.5"].to_numpy())


@pytest.mark.slow
def test_raises_when_a_days_targets_do_not_start_at_midnight(model):
    """A test_index whose day starts at, say, 06:00 would otherwise have its
    predictions silently mislabelled: forecast step k is always origin + k
    hours, and origin is always that day's 00:00, so the first 6 predicted
    hours would be dropped and the rest shifted -- conservative, not leaking,
    but wrong, and nothing previously caught it."""
    series = _series()
    misaligned = series.index[1014:1032]  # a single day's 06:00 through 23:00
    assert misaligned[0].hour == 6
    with pytest.raises(ValueError, match=str(misaligned[0])):
        model.predict(series, misaligned)


@pytest.mark.slow
def test_predict_is_deterministic_given_the_same_inputs(model):
    """torch.manual_seed is reset per day inside predict(); nothing pinned
    that directly until now. A lost reseed would otherwise only ever surface
    as a confusing failure in the corruption test."""
    series = _series()
    test_index = series.index[1008:1032]
    first = model.predict(series, test_index)
    second = model.predict(series, test_index)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.slow
def test_fit_does_not_train(model):
    assert model.name == "chronos_zeroshot"
    assert model.fitted_on_target_data is False
