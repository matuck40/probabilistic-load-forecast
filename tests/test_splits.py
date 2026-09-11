import pandas as pd

from src.splits import folds


def _series():
    index = pd.date_range("2021-01-01", "2026-09-08 23:00", freq="h")
    return pd.Series(1.0, index=index, name="load")


def test_six_folds_move_forward_without_overlap():
    all_folds = folds()
    assert [f.number for f in all_folds] == [1, 2, 3, 4, 5, 6]
    for earlier, later in zip(all_folds, all_folds[1:], strict=False):
        assert earlier.test_end < later.test_start


def test_training_window_expands_and_never_touches_the_test_period():
    series = _series()
    previous_length = 0
    for fold in folds():
        train = fold.train(series)
        assert train.index[-1] == fold.train_end
        assert train.index[-1] < fold.test_start
        assert len(train) > previous_length, "the training window must expand"
        previous_length = len(train)


def test_test_index_starts_at_midnight_and_is_whole_days():
    series = _series()
    for fold in folds():
        index = fold.test_index(series)
        assert index[0].hour == 0
        assert index[-1].hour == 23
        assert len(index) % 24 == 0
