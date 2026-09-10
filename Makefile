PY := .venv/bin/python

.PHONY: data test lint figures all clean

data:
	$(PY) data/download.py
	$(PY) data/audit.py

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check .

figures:
	$(PY) -m src.run --figures

all: data lint test
	$(PY) -m src.run --all
	$(PY) -m src.run --figures
