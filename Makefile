# Convenience targets. Run from the repository root.
PYTHON ?= python

.PHONY: install install-dev test test-fast lint smoke

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

## Full test suite (unit + end-to-end CLI tests on CPU, a few minutes)
test:
	$(PYTHON) -m pytest

## Unit tests only (seconds)
test-fast:
	$(PYTHON) -m pytest -m "not integration"

lint:
	$(PYTHON) -m ruff check src scripts tests

## One-batch sanity run of the pretraining script on a tiny HuggingFace dataset
smoke:
	$(PYTHON) scripts/train.py --config configs/eurosat/pretrain.yaml --fast_dev_run --num_workers 0 --precision 32
