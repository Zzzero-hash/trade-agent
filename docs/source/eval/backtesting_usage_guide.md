# Backtesting Module Usage Guide

## 1. Overview of the Backtesting Module

The backtesting module is a comprehensive framework for evaluating trading strategies within the trade-agent system. It provides tools for simulating trading strategies on historical data, calculating performance metrics, conducting stress tests, and generating detailed reports.

The module consists of three main components:

1. **BacktestEngine**: The core engine that runs backtests on trading strategies, handling transaction costs, slippage, and performance calculations.
2. **StressTester**: A component that performs stress testing on strategies under various market conditions.
3. **ReportGenerator**: A utility for generating comprehensive reports in HTML and CSV formats.

Key features of the backtesting module include:

- Support for long/short trading strategies
- Transaction cost and slippage modeling
- Comprehensive performance metrics calculation
- Stress testing capabilities
- Multiple report formats (HTML, CSV, PNG)
- Equity curve visualization

## 2. How to Run the Backtest Script

The backtesting module can be run using the demonstration script `scripts/run_backtest.py` or directly using the `src/eval/backtest.py` module.

### Using the Demonstration Script

The simplest way to run a backtest is to use the provided demonstration script:

```bash
python scripts/run_backtest.py
```

This script will:

1. Load sample data from `data/large_sample_data.parquet`
2. Create a simple moving average crossover strategy
3. Run the backtest using the BacktestEngine
4. Generate all reports (HTML/CSV) to the `/reports/` directory
5. Print a summary of key metrics to the console

### Using the Module Directly

For more control, you can run the backtest module directly with command-line arguments:

```bash
python src/eval/backtest.py --data data/sample_data.csv --output-dir reports
```

Available command-line options:

- `--data`: Path to the data file (required)
- `--output-dir`: Output directory for reports (default: "reports")
- `--transaction-cost`: Transaction cost per trade (default: 0.001)
- `--slippage`: Slippage per trade (default: 0.0005)
- `--initial-capital`: Initial capital for backtest (default: 100000.0)
- `--no-stress-test`: Skip stress testing

## 3. Explanation of All Output Files

The backtesting module generates several output files in the specified output directory (default: "reports"):

### HTML Reports

- `backtest_report.html`: A comprehensive HTML report containing performance metrics and stress test results in a user-friendly format.

### CSV Reports

- `trades_backtest_results.csv`: Detailed information about individual trades executed during the backtest.
- `metrics_backtest_results.csv`: All calculated performance metrics in CSV format.
- `equity_backtest_results.csv`: The equity curve data showing gross and net equity over time.

### PNG Plots

- `equity_curve.png`: A visualization of the strategy's equity curve, showing both gross and net equity over time.

## 4. Description of All Performance Metrics

The backtesting module calculates a comprehensive set of performance metrics:

### Return Metrics

- **CAGR (Compound Annual Growth Rate)**: The annualized rate of return of the strategy.
- **Volatility**: Annualized standard deviation of returns.

### Risk-Adjusted Return Metrics

- **Sharpe Ratio**: Risk-adjusted return calculated as (return - risk_free_rate) / volatility.
- **Calmar Ratio**: Risk-adjusted return calculated as CAGR / |Max Drawdown|.

### Drawdown Metrics

- **Max Drawdown**: The maximum peak-to-trough decline in the equity curve.

### Trading Metrics

- **Turnover**: Annualized measure of the strategy's trading activity.
- **Hit Ratio**: The percentage of winning trades.
- **Profit Factor**: Ratio of gross profits to gross losses.

### Statistical Metrics

- **Skewness**: Measure of the asymmetry of the return distribution.
- **Kurtosis**: Measure of the "tailedness" of the return distribution.
- **PnL Autocorrelation**: First-order autocorrelation of returns, indicating time-series dependence.

### Benchmark Metrics (when benchmark data is provided)

- **Information Ratio**: Risk-adjusted return relative to a benchmark.
- **Tracking Error**: Standard deviation of active returns relative to a benchmark.

## 5. How to Interpret Stress Test Results

The stress tester evaluates strategy performance under adverse market conditions:

### ±50% Costs

- **High Costs**: Tests strategy performance with 1.5x transaction costs and slippage.
- **Low Costs**: Tests strategy performance with 0.5x transaction costs and slippage.

These tests help evaluate the robustness of a strategy to changes in trading costs.

### Fill Delay +1 Bar

This test simulates execution delays by shifting trading signals by one additional bar. This evaluates how the strategy performs when there are delays between signal generation and execution.

When interpreting stress test results, look for:

- Strategies that maintain positive performance under stress
- Strategies with minimal performance degradation under stress
- Strategies with consistent risk metrics across stress scenarios

## 6. How to Customize the Backtesting Script for Different Strategies

To customize the backtesting script for your own strategies, you need to modify the signal generation function in `scripts/run_backtest.py`.

### Creating Custom Signals

1. Modify the `create_ma_crossover_signals` function or create a new function that generates your strategy's signals:

```python
def create_custom_strategy_signals(prices: pd.Series, **kwargs) -> pd.Series:
    """
    Create signals based on a custom strategy.

    Args:
        prices: Price series
        **kwargs: Strategy parameters

    Returns:
        pandas Series with trading signals (-1 to 1)
    """
    # Implement your strategy logic here
    signals = pd.Series(0, index=prices.index)
    # Your strategy implementation
    return signals
```

2. Update the `main` function to use your custom signal generation function:

```python
# Replace this line:
signals = create_ma_crossover_signals(prices)

# With your custom function:
signals = create_custom_strategy_signals(prices, param1=value1, param2=value2)
```

### Using Your Own Data

To use your own data:

1. Prepare your data in either CSV or Parquet format with at least a 'close' column for prices.
2. Update the data path in the script:

```python
# In load_sample_data function:
data_path = 'path/to/your/data.csv'  # or .parquet
```

### Customizing Backtest Parameters

You can modify the backtest parameters in the `main` function:

```python
engine = BacktestEngine(
    transaction_cost=0.001,    # Adjust transaction costs
    slippage=0.0005,           # Adjust slippage
    initial_capital=100000.0   # Adjust initial capital
)
```

## 7. Example Usage with Sample Code

Here's a complete example of how to use the backtesting module:

```python
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.backtest import BacktestEngine, StressTester, ReportGenerator

# Load your data
df = pd.read_parquet('data/sample_data.parquet')
prices = df['close']

# Create your strategy signals (example: mean reversion)
def create_mean_reversion_signals(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """Create signals based on a mean reversion strategy."""
    z_score = (prices - prices.rolling(lookback).mean()) / prices.rolling(lookback).std()
    signals = pd.Series(0, index=prices.index)
    signals[z_score < -1] = 1   # Buy when oversold
    signals[z_score > 1] = -1   # Sell when overbought
    return signals.fillna(0)

# Generate signals
signals = create_mean_reversion_signals(prices)

# Initialize backtest engine
engine = BacktestEngine(
    transaction_cost=0.001,
    slippage=0.0005,
    initial_capital=100000.0
)

# Run backtest
results = engine.run_backtest(signals, prices)

# Run stress tests
stress_tester = StressTester(engine)
stress_results = stress_tester.run_stress_tests(signals, prices)

# Generate reports
report_generator = ReportGenerator("reports")
report_generator.generate_csv_report(results, "mean_reversion_results.csv")
report_generator.generate_html_report(results, stress_results, "mean_reversion_report.html")
report_generator.plot_equity_curve(results, "mean_reversion_equity.png")

# Print key metrics
metrics = results['metrics']
print(f"CAGR: {metrics.get('cagr', 0):.4f}")
print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
```

This example demonstrates how to:

1. Load data
2. Create a custom strategy
3. Run a backtest
4. Perform stress tests
5. Generate reports
6. Access performance metrics
