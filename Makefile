PY := .venv/bin/python

.PHONY: data test lint figures all clean

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
