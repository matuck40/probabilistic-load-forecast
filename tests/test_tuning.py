import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.tuning import SEARCH_ORDER, temporal_split


def _xy(n=1000):
    index = pd.date_range("2021-01-01", periods=n, freq="h")
    X = pd.DataFrame({"a": np.arange(n, dtype="float64")}, index=index)
    y = pd.Series(np.arange(n, dtype="float64"), index=index)
    return X, y


def test_split_is_temporal_not_random():
    X, y = _xy()
    X_fit, y_fit, X_val, y_val = temporal_split(X, y, validation_fraction=0.2)
    assert len(X_val) == 200
    assert X_fit.index.max() < X_val.index.min(), "validation must be strictly later"
    assert X_fit.index.equals(y_fit.index)
    assert X_val.index.equals(y_val.index)


def test_split_rejects_a_degenerate_fraction():
    X, y = _xy()
    with pytest.raises(ValueError):
        temporal_split(X, y, validation_fraction=0.0)


def test_search_order_matches_the_spec():
    names = [name for name, _ in SEARCH_ORDER]
    assert names == [
        "learning_rate",
        "num_leaves",
        "min_data_in_leaf",
        "feature_fraction",
        "lambda_l2",
    ]
    for _, candidates in SEARCH_ORDER:
        assert len(candidates) >= 2, "a parameter with one candidate is not being searched"


def test_tuning_json_records_that_the_tuned_params_were_not_adopted():
    """The committed artifact must say, on its own, why it must not be used.

    The search window spans the test periods of every fold before the last,
    so wiring these parameters into the default model and then reporting
    walk-forward metrics would be leakage. That is a property of how the
    committed results/tuning.json was produced, not just a comment in the
    code, so it is asserted here: a future edit that quietly drops the
    warning breaks this test rather than only a reader's judgement.
    """
    data = json.loads((config.RESULTS_DIR / "tuning.json").read_text())
    assert data["adopted"] is False
    assert isinstance(data["not_adopted_reason"], str)
    assert data["not_adopted_reason"].strip() != ""
