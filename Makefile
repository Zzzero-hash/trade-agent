# Makefile for trade-agent project

# Default target
.PHONY: help
help:  ## Display this help message
	@echo "Available tasks:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

.PHONY: test
test: ## Run all tests and linters
ruff check src tests
black --check src tests
mypy src tests
pytest

.PHONY: ci
ci: test ## Run all CI checks
$(MAKE) docs

# Backtesting task
.PHONY: backtest
backtest: data/large_sample_data.parquet  ## Run the backtesting process
	python scripts/run_backtest.py

# Data dependency
data/large_sample_data.parquet:
	@echo "Data file $@ not found. Please ensure data is available."
	@exit 1

# Documentation build
.PHONY: docs
docs:  ## Build Sphinx documentation into docs/_build/html (warnings as errors)
	@sphinx-build -W -b html docs/source docs/_build/html
	@echo "Docs built at docs/_build/html/index.html"

.PHONY: docs-versions
docs-versions:  ## Build multi-version docs into docs/_build/multiversion
	@sphinx-multiversion docs/source docs/_build/multiversion
	@echo "Multi-version docs built at docs/_build/multiversion/index.html"

.PHONY: docs-live
docs-live:  ## Auto-reload Sphinx docs (requires sphinx-autobuild)
	@command -v sphinx-autobuild >/dev/null 2>&1 || { echo 'Install sphinx-autobuild to use live mode'; exit 1; }
	@sphinx-autobuild docs/source docs/_build/html --open-browser

.PHONY: docs-clean
docs-clean:  ## Remove built documentation
	rm -rf docs/_build

# Hydra-based supervised learning training
.PHONY: train-hydra
train-hydra:  ## Run a Hydra-configured SL training (override MODEL=ridge DATA=... TARGET=...)
	@MODEL?=ridge
	@DATA?=data/sample_data.parquet
	@TARGET?=close
	python scripts/train_sl_hydra.py model=$${MODEL} train.data_path=$${DATA} train.target=$${TARGET}
