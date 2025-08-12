# Makefile for trade-agent project

# Default target
.PHONY: help
help:  ## Display this help message
	@echo "Available tasks:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# Backtesting task
.PHONY: backtest
backtest: data/large_sample_data.parquet  ## Run the backtesting process
	python scripts/run_backtest.py

# Data dependency
data/large_sample_data.parquet:
	@echo "Data file $@ not found. Please ensure data is available."
	@exit 1
