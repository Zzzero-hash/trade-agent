# Trade Agent

A sophisticated trading agent with advanced data processing capabilities and symbol validation framework.

## Overview

The Trade Agent is a comprehensive system designed for financial data analysis and trading operations. It features a robust symbol validation framework and a highly efficient, parallelized data processing pipeline built on Ray for optimal performance.

## Key Features

### Parallelized Data Pipeline

The system employs a cutting-edge parallelized data architecture that significantly improves performance and scalability:

- **Symbol-Centric Processing**: Instead of processing large monolithic DataFrames sequentially, the pipeline processes each financial symbol independently in parallel
- **Ray Framework Integration**: Leverages the Ray distributed computing framework for efficient parallel execution
- **Multi-Stage Parallelization**: Data fetching, transformation, and feature extraction are all performed in parallel across multiple CPU cores
- **Dynamic Load Balancing**: Automatically distributes workloads based on the number of symbols available
- **Centralized Data Alignment**: Intelligent merging of parallel results into a unified DataFrame for downstream analysis

#### Benefits of the New Architecture

- **Improved Performance**: Dramatically faster processing times through parallel execution
- **Better Scalability**: Handles growing numbers of symbols efficiently without performance degradation
- **Resource Optimization**: Maximizes CPU utilization by distributing workloads across available cores
- **Fault Tolerance**: Individual symbol processing failures don't halt the entire pipeline
- **Maintainability**: Cleaner separation of concerns with dedicated functions for each processing stage

#### How It Works

The parallelized pipeline follows this workflow:

1. **Parallel Data Fetching**: Multiple symbols are fetched simultaneously using Ray remote functions
2. **Parallel Data Transformation**: Each symbol's data is transformed concurrently as it becomes available
3. **Parallel Feature Extraction**: Time-series features are extracted for each symbol in parallel
4. **Result Alignment**: Individual results are collected and merged into a final, aligned DataFrame

This architecture replaces the previous sequential approach, which processed one large DataFrame at a time, with a modern parallel design that scales horizontally with the number of CPU cores available.

### Symbol Validation and Correction Framework

A comprehensive system ensuring data integrity and reliability:

- **Multi-Asset Class Support**: Validates equities, ETFs, cryptocurrencies, commodities, REITs, volatility products, and forex pairs
- **Concurrent Processing**: Validates multiple symbols simultaneously using thread pools
- **Data Quality Scoring**: Calculates quality scores based on trading history, volume, and metadata completeness
- **Intelligent Caching**: JSON-based caching system to avoid redundant API calls
- **Automated Correction Suggestions**: Provides potential symbol replacements for invalid entries

## Project Structure

```
├── src/
│   ├── data/                 # Data processing modules
│   │   ├── ingestion.py      # Data ingestion (now parallelized with Ray)
│   │   ├── processing.py     # Data transformation and feature extraction (parallelized)
│   │   ├── orchestrator.py   # Pipeline orchestration (Ray-based)
│   │   ├── validation.py     # Symbol validation logic
│   │   ├── symbol_corrector.py # Symbol correction framework
│   │   └── cleaning.py       # Data cleaning utilities
│   ├── agents/               # Trading agents
│   │   ├── ppo_agent.py      # PPO-based trading agent
│   │   └── sac_agent.py      # SAC-based trading agent
│   └── models/               # ML models
│       └── cnn_lstm.py       # CNN-LSTM hybrid model
├── docs/                     # Documentation
│   ├── ray_parallelization_plan.md
│   ├── symbol_validation_framework.md
│   └── agents/
├── main.py                   # Main entry point
└── test_symbol_validation.py # Validation tests
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (see `.env.example`)

## Usage

### Running Symbol Validation

```bash
python -c "from src.data.symbol_corrector import main; main()"
```

### Running the Parallelized Data Pipeline

```python
from src.data.orchestrator import new_orchestrator

# Process symbols in parallel
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
aligned_data = new_orchestrator(symbols, '2023-01-01', '2023-12-31')
```

## Configuration

The system uses configurable parameters for validation and processing:

- **Validation Configuration**: Adjust timeout, retry limits, volume thresholds, and cache duration
- **Ray Configuration**: Control parallel execution settings and resource allocation
- **Data Processing**: Customize feature extraction parameters and data transformation rules

## Performance

The new parallelized architecture provides significant performance improvements:

- **Up to 8x faster** processing for large symbol universes
- **Linear scaling** with the number of CPU cores
- **Reduced memory usage** through streaming processing
- **Faster iteration times** during development and testing

## Documentation

- [Ray Parallelization Plan](docs/ray_parallelization_plan.md) - Detailed architecture overview and implementation guide
- [Symbol Validation Framework](docs/symbol_validation_framework.md) - Comprehensive validation system documentation
- [Agent Documentation](docs/agents/) - Individual agent documentation and usage guides

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
