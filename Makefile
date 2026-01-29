.DEFAULT_GOAL := help
PYTHON ?= python3
IMAGE  ?= surrogate-factory
TAG    ?= latest

.PHONY: help install lint format test build docker docker-run clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## Run ruff linter and mypy type-checker
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/surrogate_factory --ignore-missing-imports

format:  ## Auto-format code with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:  ## Run the test suite
	pytest tests/ -v --tb=short

build:  ## Build the Python distribution
	$(PYTHON) -m build

docker:  ## Build the Docker image
	docker build -t $(IMAGE):$(TAG) .

docker-run:  ## Run the API container locally
	docker compose up --build

clean:  ## Remove build artefacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
