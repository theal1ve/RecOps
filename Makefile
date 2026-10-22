.DEFAULT_GOAL := help
PY ?= python3.14
VENV := .venv
BIN := $(VENV)/bin
PROFILE ?= full
WORKDIR ?= run
CLI := $(BIN)/python -m recops.cli --profile $(PROFILE) --workdir $(WORKDIR)

.PHONY: help setup lock lint format test smoke data train eval bench report serve up down clean

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

setup:  ## Create venv, install deps (uses requirements.lock if present), pre-commit
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	@if [ -f requirements.lock ]; then \
		$(BIN)/pip install -r requirements.lock && $(BIN)/pip install --no-deps -e .; \
	else \
		$(BIN)/pip install -e ".[dev,ml,tracking,serve]"; \
	fi
	$(BIN)/pre-commit install

lock:  ## Pin all dependency versions into requirements.lock (needs network)
	$(BIN)/uv pip compile pyproject.toml --all-extras --python-version 3.14 -o requirements.lock

lint:  ## ruff check
	$(BIN)/ruff check .

format:  ## ruff autofix + format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:  ## Unit tests (fast)
	$(BIN)/pytest -m "not slow"

smoke:  ## End-to-end on synthetic data (needs ml+tracking extras)
	$(BIN)/pytest -m slow -x

data:  ## Download Yambda and build processed tables
	$(CLI) download && $(CLI) prepare && $(CLI) validate

train:  ## Rolling simulation: train -> evaluate -> promote/reject, with monitoring
	$(CLI) simulate

eval:  ## Final one-shot TEST evaluation + ablations
	$(CLI) final

bench:  ## API latency measurement
	$(CLI) bench

report:  ## Build artifacts/report.md from run outputs
	$(CLI) report

serve:  ## Run the FastAPI service (uses the registry's production model)
	RECOPS_PROFILE=$(PROFILE) RECOPS_WORKDIR=$(WORKDIR) $(CLI) serve

up:  ## docker compose up (MLflow; API with --profile api)
	docker compose up --build

down:  ## docker compose down
	docker compose down

clean:  ## Remove caches (keeps data and runs)
	rm -rf .pytest_cache .ruff_cache build *.egg-info
