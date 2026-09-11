"""Run every model over every fold and record the result and its provenance.

A metric without the conditions that produced it is a rumour. run.json exists so
that a number in the README can be traced to a library version, a seed, a fold
boundary and a file hash.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

from src import config
from src.metrics import evaluate_predictions
from src.models.base import Forecaster
from src.splits import Fold, folds

METRIC_KEYS = ("wape", "mae", "rmse", "pinball", "coverage")
TRACKED_LIBRARIES = ("pandas", "numpy", "lightgbm", "scikit-learn", "holidays", "torch")


def aggregate(per_fold: list[dict]) -> tuple[dict, dict]:
    """Mean and standard deviation across folds.

    The standard deviation is reported because a mean alone hides a model that
    wins in January and loses in July.
    """
    frame = pd.DataFrame(per_fold)[list(METRIC_KEYS)]
    # Rounded to 10 decimal places: metrics.json is a committed artifact meant
    # to be diffed and reproduced, and the last bit of a float64 sum is not
    # portable across CPUs and library builds. 10 decimals is far more
    # precision than any of these metrics carries, so rounding to it removes
    # that noise from the artifact without losing anything meaningful.
    return frame.mean().round(10).to_dict(), frame.std(ddof=0).round(10).to_dict()


def run_model(model: Forecaster, series: pd.Series, fold_list: list[Fold] | None = None) -> dict:
    fold_list = fold_list if fold_list is not None else folds()
    per_fold = []
    for fold in fold_list:
        model.fit(fold.train(series))
        test_index = fold.test_index(series)
        # Truncated to the fold's own test_end, not the full series: no lag any
        # model builds ever reaches forward, so nothing past this point is
        # legitimately needed. This is what makes the boundary a guarantee the
        # harness enforces rather than a convention that depends on every
        # model's own tests, including ones written less carefully in future.
        predictions = model.predict(series.loc[: fold.test_end], test_index)
        scores = evaluate_predictions(series.reindex(test_index), predictions)
        per_fold.append(
            {
                "fold": fold.number,
                "train_end": str(fold.train_end),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                **scores,
            }
        )
    mean, std = aggregate(per_fold)
    return {"per_fold": per_fold, "mean": mean, "std": std}


def _library_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def write_run_metadata(path: Path, series: pd.Series, models: list[str]) -> None:
    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = {
        "seed": config.SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": _library_versions(),
        "subsystem": config.SUBSYSTEM,
        "period": [str(series.index[0]), str(series.index[-1])],
        "observations": int(len(series)),
        "quantiles": list(config.QUANTILES),
        "folds": [
            {
                "fold": f.number,
                "train_end": str(f.train_end),
                "test_start": str(f.test_start),
                "test_end": str(f.test_end),
            }
            for f in folds()
        ],
        "models": models,
        "input_sha256": {name: entry["sha256"] for name, entry in manifest["files"].items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_metrics(path: Path, results: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
