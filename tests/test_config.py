import pandas as pd

from src import config


def test_locked_constants():
    assert config.SUBSYSTEM == "SE"
    assert config.SEED == 42
    assert config.QUANTILES == (0.1, 0.5, 0.9)
    assert config.HORIZONS == tuple(range(1, 25))
    assert config.PERIOD_START == pd.Timestamp("2021-01-01 00:00")
    assert config.PERIOD_END == pd.Timestamp("2026-09-08 23:00")


def test_six_folds_are_quarterly_and_ordered():
    assert len(config.FOLDS) == 6
    previous_end = None
    for train_end, test_start, test_end in config.FOLDS:
        train_end = pd.Timestamp(train_end)
        test_start = pd.Timestamp(test_start)
        test_end = pd.Timestamp(test_end)
        assert train_end < test_start, "train must end before test starts"
        assert test_start < test_end
        if previous_end is not None:
            assert test_start > previous_end, "folds must move forward in time"
        previous_end = test_end
    last_test_end = pd.Timestamp(config.FOLDS[-1][2])
    assert last_test_end == pd.Timestamp("2026-06-30 23:00"), "2026Q3 is never evaluated"
