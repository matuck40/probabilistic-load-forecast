"""Turn results/metrics.json into the markdown the README shows.

Every number in the README comes through this function. A number typed by hand
into a README is a number that will still be there after the run that
contradicts it.
"""

from __future__ import annotations

import json
from pathlib import Path

from src import config

COLUMNS = ("wape", "mae", "rmse", "pinball", "coverage")
HEADERS = {"wape": "WAPE", "mae": "MAE", "rmse": "RMSE", "pinball": "Pinball", "coverage": "P10-P90 coverage"}


def load_metrics(path: Path | None = None) -> dict:
    path = path or config.RESULTS_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def winner(metrics: dict) -> tuple[str, float]:
    name = min(metrics, key=lambda k: metrics[k]["mean"]["wape"])
    return name, metrics[name]["mean"]["wape"]


def _cell(mean: float, std: float, metric: str) -> str:
    if metric in ("wape", "coverage"):
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:,.0f} ± {std:,.0f}"


def metrics_table(metrics: dict) -> str:
    order = sorted(metrics, key=lambda k: metrics[k]["mean"]["wape"])
    header = "| Model | " + " | ".join(HEADERS[c] for c in COLUMNS) + " |"
    separator = "|---" * (len(COLUMNS) + 1) + "|"
    rows = [
        "| "
        + name
        + " | "
        + " | ".join(_cell(metrics[name]["mean"][c], metrics[name]["std"][c], c) for c in COLUMNS)
        + " |"
        for name in order
    ]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    data = load_metrics()
    print(metrics_table(data))
    name, value = winner(data)
    print(f"\nLowest WAPE: {name} at {value:.4f}")
