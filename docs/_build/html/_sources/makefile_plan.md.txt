# Makefile Plan for trade-agent

## Purpose

Create a Makefile to standardize common development tasks including installation, testing, and running the application.

## File Location

`Makefile` (in the root directory)

## Required Commands

### 1. Installation Commands

- `install`: Install the package in development mode
- `install-deps`: Install only dependencies
- `install-dev`: Install with development dependencies

### 2. Development Commands

- `lint`: Run code quality checks
- `format`: Format code with black and isort
- `test`: Run unit tests
- `test-cov`: Run tests with coverage report

### 3. Run Commands

- `run`: Run the main application
- `smoke-test`: Run the smoke test
- `serve`: Start the API server (if applicable)

### 4. Clean Commands

- `clean`: Remove build artifacts
- `clean-pyc`: Remove Python cache files
- `clean-all`: Remove all generated files

### 5. Documentation Commands

- `docs`: Build documentation
- `docs-serve`: Serve documentation locally

## Implementation Plan

```makefile
# Makefile for trade-agent
.PHONY: install install-deps install-dev lint format test test-cov run smoke-test serve clean clean-pyc clean-all docs docs-serve

# Variables
PYTHON := python3
PIP := pip
PACKAGE := trade-agent

# Installation
install:
	$(PIP) install -e .

install-deps:
	$(PIP) install -e . --no-deps

install-dev:
	$(PIP) install -e .[dev]

# Code Quality
lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests
	$(PYTHON) -m isort --check-only src tests

format:
	$(PYTHON) -m black src tests
	$(PYTHON) -m isort src tests
	$(PYTHON) -m ruff check --fix src tests

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Running
run:
	$(PYTHON) src/main.py

smoke-test:
	$(PYTHON) src/main.py --smoke-test

serve:
	$(PYTHON) src/main.py serve

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

clean-pyc:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

clean-all: clean clean-pyc
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml

# Documentation
docs:
	$(MAKE) -C docs html

docs-serve:
	$(MAKE) -C docs serve

# Help
help:
	@echo "Available commands:"
	@echo "  install       - Install package in development mode"
	@echo "  install-deps  - Install only dependencies"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  lint          - Run code quality checks"
	@echo "  format        - Format code"
	@echo "  test          - Run unit tests"
	@echo "  test-cov      - Run tests with coverage"
	@echo "  run           - Run the main application"
	@echo "  smoke-test    - Run smoke test"
	@echo "  serve         - Start API server"
	@echo "  clean         - Remove build artifacts"
	@echo "  clean-pyc     - Remove Python cache files"
	@echo "  clean-all     - Remove all generated files"
	@echo "  docs          - Build documentation"
	@echo "  docs-serve    - Serve documentation locally"
	@echo "  help          - Show this help message"

.DEFAULT_GOAL := help
```

## Usage Examples

1. Install the package:

   ```bash
   make install
   ```

2. Run smoke test:

   ```bash
   make smoke-test
   ```

3. Format code:

   ```bash
   make format
   ```

4. Run tests:

   ```bash
   make test
   ```

5. Clean cache files:
   ```bash
   make clean-pyc
   ```

## Expected Behavior

- All commands should work without errors after dependencies are installed
- The smoke-test command should output "OK" when all dependencies are properly installed
- The format command should automatically format Python files according to project standards
- The clean commands should properly remove specified files and directories
