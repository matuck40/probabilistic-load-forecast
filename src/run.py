"""Command line entry point. One switch per model, plus --all and --figures."""

from __future__ import annotations

import argparse
import sys
import time

from src import config
from src.dataset import load_series
from src.evaluate import run_model, write_metrics, write_run_metadata
from src.models.baseline import SeasonalNaive
from src.models.gbm import LightGBMForecaster

MODEL_NAMES = ("naive24", "naive168", "lightgbm_l1", "lightgbm_l2", "chronos")


def build_model(name: str):
    if name == "naive24":
        return SeasonalNaive(lag=24)
    if name == "naive168":
        return SeasonalNaive(lag=168)
    if name == "lightgbm_l1":
        return LightGBMForecaster(objective="l1")
    if name == "lightgbm_l2":
        return LightGBMForecaster(objective="l2")
    if name == "chronos":
        from src.models.chronos_model import ChronosForecaster

        return ChronosForecaster()
    raise ValueError(f"unknown model {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the load forecasting experiments.")
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], help="which model to score")
    parser.add_argument("--all", action="store_true", help="score every model")
    parser.add_argument("--figures", action="store_true", help="render the figures from saved artifacts")
    args = parser.parse_args(argv)

    if args.figures:
        from src.figures import render_all

        render_all()
        return 0

    if args.all or args.model == "all":
        # Each model runs in its own process. torch, which Chronos needs, and
        # LightGBM each load their own OpenMP runtime, and the two segfault or
        # deadlock when they share a process. Running them in sequence here
        # would import both. The per-model runs merge into results/metrics.json,
        # so the outcome is identical and the crash is structurally impossible.
        import subprocess

        for name in MODEL_NAMES:
            print(f"--- {name} ---", flush=True)
            completed = subprocess.run([sys.executable, "-m", "src.run", "--model", name], check=False)
            if completed.returncode != 0:
                print(f"{name} exited with {completed.returncode}", file=sys.stderr)
                return completed.returncode
        return 0

    if args.model is None:
        parser.error("pass --model, --all or --figures")
    names = [args.model]

    series = load_series()
    metrics_path = config.RESULTS_DIR / "metrics.json"
    results = {}
    if metrics_path.exists():
        import json

        results = json.loads(metrics_path.read_text(encoding="utf-8"))

    for name in names:
        model = build_model(name)
        started = time.perf_counter()
        results[model.name] = run_model(model, series)
        results[model.name]["seconds"] = round(time.perf_counter() - started, 1)
        print(f"{model.name}: WAPE {results[model.name]['mean']['wape']:.4f} "
              f"(sd {results[model.name]['std']['wape']:.4f}) in {results[model.name]['seconds']}s")

    write_metrics(metrics_path, results)
    write_run_metadata(config.RESULTS_DIR / "run.json", series, sorted(results))
    print(f"\nwrote {metrics_path} and {config.RESULTS_DIR / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
