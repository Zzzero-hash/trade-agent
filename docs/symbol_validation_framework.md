# Symbol Validation and Correction Framework

## 1. Overview

This document outlines the architecture and functionality of the comprehensive Symbol Validation and Correction Framework. This system is designed to ensure the integrity, accuracy, and reliability of all financial symbols used within the trading agent. It provides a robust, multi-asset class validation engine that automatically checks symbols against real-world market data, identifies issues, and suggests corrections.

The framework is built to handle a wide variety of asset classes, including:

- Equities
- Exchange-Traded Funds (ETFs)
- Cryptocurrencies
- Commodity Futures
- Real Estate Investment Trusts (REITs)
- Volatility Products
- Forex Pairs

The system is designed to be extensible, allowing for new validation rules and asset classes to be added in the future.

## 2. Core Components

The framework consists of two primary components: the **SymbolValidator** and the **SymbolCorrector**.

### 2.1. SymbolValidator

The `SymbolValidator` is the core engine responsible for validating individual symbols. It is a highly configurable component that leverages the `yfinance` library to fetch and analyze market data from Yahoo Finance.

#### Key Features:

- **Multi-Asset Class Validation:** Provides specialized validation logic for each supported asset class.
- **Configurable Validation Rules:** A `ValidationConfig` dataclass allows for easy adjustment of validation parameters, such as trading history requirements, volume thresholds, and API settings.
- **Concurrent Processing:** Utilizes a `ThreadPoolExecutor` to validate multiple symbols in parallel, significantly speeding up the validation of the entire market universe.
- **Detailed Validation Results:** Returns a `ValidationResult` dataclass for each symbol, containing its status, asset class, current name, and detailed metadata.
- **Data Quality Scoring:** Calculates a data quality score for each valid symbol based on factors like trading history, volume, and completeness of metadata.
- **Caching System:** Implements a JSON-based caching mechanism to store validation results, avoiding redundant API calls and speeding up subsequent runs. The cache has a configurable expiration time.

#### Validation Status:

The validator assigns one of the following statuses to each symbol:

- `VALID`: The symbol is active, tradable, and meets all data quality standards.
- `INVALID`: The symbol exists but fails one or more validation checks (e.g., insufficient trading history, low volume).
- `DELISTED`: The symbol appears to be delisted and is no longer trading.
- `DATA_UNAVAILABLE`: No data could be retrieved for the symbol from the API.
- `API_ERROR`: An error occurred while communicating with the data provider.
- `UNKNOWN`: The status of the symbol could not be determined.

### 2.2. SymbolCorrector

The `SymbolCorrector` orchestrates the validation of the entire market universe. It uses the `SymbolValidator` to check all symbols from the `MarketUniverse` and generates a comprehensive report of its findings.

#### Key Features:

- **Full Market Universe Validation:** Iterates through all instruments in the `MarketUniverse`, validating each one.
- **Comprehensive Reporting:** Generates a detailed validation report in `symbol_validation_report.txt`, which includes a summary of valid and invalid symbols, detailed error messages, and suggested corrections.
- **Automated Correction Suggestions:** For certain types of invalid symbols, the corrector can suggest potential replacements (e.g., `BRK-B` -> `BRK-A`).

## 3. Workflow

The validation and correction process follows this workflow:

### 3.1. Data Processing Pipeline Integration

With the new parallelized architecture, the symbol validation framework now integrates seamlessly with the Ray-based data processing pipeline. The validation process benefits from the improved performance and scalability of the parallelized data flow.

### 3.2. Enhanced Validation Workflow

The validation and correction process now leverages the parallelized architecture:

1.  **Initialization:** The `SymbolCorrector` initializes the `SymbolValidator` with the specified configuration. The validator loads any existing validation results from its cache.
2.  **Parallel Data Processing:** The system now uses the new Ray-based orchestrator to fetch and process symbol data in parallel:
    a. **Parallel Data Fetching**: Multiple symbols are fetched simultaneously using Ray remote functions
    b. **Parallel Data Transformation**: Each symbol's data is transformed concurrently as it becomes available
    c. **Parallel Feature Extraction**: Time-series features are extracted for each symbol in parallel
    d. **Result Alignment**: Individual results are collected and merged into a final, aligned DataFrame
3.  **Batch Validation:** The corrector fetches the full list of instruments from the `MarketUniverse` and passes them to the validator's `validate_symbols_batch` method, which now operates on the parallelized data stream.
4.  **Concurrent Validation:** The validator's thread pool processes the symbols concurrently. For each symbol, it:
    a. Checks if a valid, non-expired result exists in the cache.
    b. If not, it determines the asset class and calls the `yfinance` API to get symbol information and historical data.
    c. It applies a series of validation checks based on the configuration (e.g., `min_trading_days`, `min_volume_threshold`).
    d. It runs asset-class-specific validation logic.
    e. It calculates a data quality score.
    f. It stores the `ValidationResult` in the cache.
5.  **Report Generation:** Once all symbols have been validated, the `SymbolCorrector` generates the `symbol_validation_report.txt` file, summarizing the results.
6.  **Manual Review:** The validation report is reviewed by the user to identify symbols that require manual intervention or removal from the market universe.

### 3.3. Performance Benefits

The integration with the parallelized data pipeline provides significant performance improvements:

- **Faster Validation**: Symbols are validated concurrently with data processing, reducing overall pipeline latency
- **Improved Throughput**: The validation system can handle larger symbol universes efficiently
- **Resource Optimization**: CPU resources are better utilized through parallel execution
- **Scalability**: The validation system scales horizontally with the number of CPU cores

## 4. Configuration

The behavior of the `SymbolValidator` is controlled by the `ValidationConfig` dataclass, located in `src/data/validation.py`. This allows for fine-tuning of the validation process without changing the core logic.

The key configuration parameters include:

- `request_timeout`: Timeout for API requests.
- `max_retries`: Number of times to retry a failed API request.
- `max_workers`: Number of concurrent threads for batch validation.
- `min_trading_days`: The minimum number of trading days required for a symbol to be considered valid.
- `min_volume_threshold`: The minimum average trading volume.
- `max_days_since_last_trade`: The maximum number of days since a symbol last traded to be considered active.
- `cache_duration_hours`: The number of hours a cached result is considered valid.

## 5. How to Run the Validation

The validation process can be run directly from the command line:

```bash
python -c "from src.data.symbol_corrector import main; main()"
```

This will execute the full validation and correction process, generating a new `symbol_validation_report.txt`.

## 6. Future Enhancements

The framework is designed to be extensible. Future enhancements could include:

- **Integration with additional data sources** to cross-reference validation results.
- **Automated application of corrections** for high-confidence suggestions.
- **More sophisticated data quality metrics**.
- **Support for additional asset classes**.
