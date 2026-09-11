PY := .venv/bin/python

.PHONY: setup data test lint figures all clean

# The virtualenv every other target depends on. Nothing else creates it,
# so this is the first command on a clean machine.
setup:
	python3.12 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt


data:
	$(PY) data/download.py
	$(PY) data/audit.py

# torch and LightGBM each load their own OpenMP runtime and crash when they
# share a process. pytest imports every test module during collection, BEFORE
# marker filtering, so a marker alone is not enough: the slow group must also
# be scoped by path, or collecting tests/test_gbm.py pulls LightGBM in.
test:
	$(PY) -m pytest -q -m "not slow"
	$(PY) -m pytest -q tests/test_chronos_model.py -m "slow"

lint:
	.venv/bin/ruff check .

figures:
	$(PY) -m src.run --figures

all: data lint test
	$(PY) -m src.run --model naive24
	$(PY) -m src.run --model naive168
	$(PY) -m src.run --model lightgbm_l1
	$(PY) -m src.run --model lightgbm_l2
	$(PY) -m src.run --model chronos
	$(PY) -m src.run --figures

# Caches only. The virtualenv costs an install to rebuild and data/raw costs
# a full download from ONS, so neither is removed here.
clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
