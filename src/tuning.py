"""Tune five parameters, in a fixed order, and record what it bought.

The order is the order of expected return, and it is fixed in advance so that
the search cannot be quietly redirected after seeing a result. Each parameter is
chosen with the others held at their current best, which is a coordinate search:
cheaper than a grid and honest about being greedy.
"""

from __future__ import annotations

import json
import time

import pandas as pd

from src import config
from src.features import build_features
from src.metrics import wape
from src.models.gbm import DEFAULT_PARAMS, LightGBMForecaster

SEARCH_ORDER: tuple[tuple[str, tuple], ...] = (
    ("learning_rate", (0.02, 0.05, 0.1)),
    ("num_leaves", (15, 31, 63)),
    ("min_data_in_leaf", (20, 40, 80)),
    ("feature_fraction", (0.6, 0.8, 1.0)),
    ("lambda_l2", (0.0, 1.0, 10.0)),
)

VALIDATION_HORIZON = 12  # one representative horizon, to keep the search affordable
VALIDATION_FRACTION = 0.15  # the holdout is the tail of the search's own training window


def temporal_split(X: pd.DataFrame, y: pd.Series, validation_fraction: float = 0.15):
    """Split by position: the validation set is the tail, never a random sample."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    cut = len(X) - int(len(X) * validation_fraction)
    if cut <= 0 or cut >= len(X):
        raise ValueError("validation_fraction leaves an empty side")
    return X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]


def _score(params: dict, X, y, X_val, y_val) -> float:
    model = LightGBMForecaster(objective="l1", params=params, quantile=False)
    estimator = model._make("l1")
    estimator.fit(X, y)
    return wape(y_val, pd.Series(estimator.predict(X_val), index=y_val.index))


def tune(series: pd.Series, train_end: pd.Timestamp) -> dict:
    """Coordinate search over the five parameters, scored on a temporal holdout.

    The result is a measurement, not a recommendation. `train_end` is whatever
    the caller passes in; this function does not know, and does not need to
    know, whether that window overlaps a walk-forward fold's test period. The
    caller that does know that (the `__main__` block below, which passes the
    *last* fold's train_end) is the one that stamps the result with whether
    the chosen parameters may be adopted and why -- see `not_adopted_reason`
    in the written JSON. The provenance fields returned here (`train_end`,
    `horizon`, `validation_fraction`, `n_fits`, `seconds`) exist so a later
    reader can audit what was measured without reading this file.
    """
    started = time.perf_counter()
    train = series.loc[:train_end]
    X, y = build_features(train, VALIDATION_HORIZON)
    X_fit, y_fit, X_val, y_val = temporal_split(X, y, validation_fraction=VALIDATION_FRACTION)

    best = dict(DEFAULT_PARAMS)
    best["n_estimators"] = 1200
    baseline = _score(best, X_fit, y_fit, X_val, y_val)
    history = [{"stage": "defaults", "params": dict(best), "wape": baseline}]
    n_fits = 1

    for name, candidates in SEARCH_ORDER:
        scores = {}
        for value in candidates:
            trial = {**best, name: value}
            scores[value] = _score(trial, X_fit, y_fit, X_val, y_val)
            n_fits += 1
        chosen = min(scores, key=scores.get)
        best[name] = chosen
        history.append({"stage": name, "chosen": chosen, "scores": {str(k): v for k, v in scores.items()}})

    final = _score(best, X_fit, y_fit, X_val, y_val)
    n_fits += 1
    gain = (baseline - final) / baseline
    return {
        "baseline_wape": baseline,
        "tuned_wape": final,
        "relative_gain": gain,
        "params": best,
        "history": history,
        "train_end": str(train_end),
        "horizon": VALIDATION_HORIZON,
        "validation_fraction": VALIDATION_FRACTION,
        "n_fits": n_fits,
        "seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    from src.dataset import load_series
    from src.splits import folds

    fold_list = folds()
    train_end = fold_list[-1].train_end
    spanned = [f.number for f in fold_list[:-1]]

    result = tune(load_series(), train_end)

    # The search window is everything up to the last fold's train_end, which
    # is *after* the test periods of every earlier fold. Adopting these
    # parameters to score those folds would select hyperparameters on data
    # the model is graded against -- exactly the leakage this project exists
    # to avoid. The gain is a legitimate, honestly-measured number; the
    # values are not wired into DEFAULT_PARAMS or any model path.
    result["adopted"] = False
    result["not_adopted_reason"] = (
        f"The search window ends at the last fold's train_end ({train_end}) and "
        f"therefore spans the test periods of fold(s) {spanned[0]} to {spanned[-1]}, "
        "so using these parameters to score those folds would select "
        "hyperparameters on data the model is graded against."
    )
    result["comparability_note"] = (
        "This WAPE is not comparable to the walk-forward numbers in results/metrics.json: "
        "it scores a single horizon, point forecast only, on a holdout carved from inside "
        "one training window, not a full walk-forward fold."
    )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "tuning.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"defaults WAPE {result['baseline_wape']:.4f} -> tuned {result['tuned_wape']:.4f}")
    print(f"relative gain {100 * result['relative_gain']:.2f}%")
