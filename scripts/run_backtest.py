#!/usr/bin/env python3
"""
Script to demonstrate how to use the backtesting module.

This script:
1. Loads sample data from data/sample_data.parquet
2. Creates a simple moving average crossover strategy
3. Runs the backtest using src/eval/backtest.py
4. Generates all reports (HTML/CSV) to the /reports/ directory
5. Prints a summary of key metrics to the console
"""

import os
import sys

import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.backtest import BacktestEngine, ReportGenerator, StressTester


def load_sample_data(
    data_path: str = 'data/large_sample_data.parquet'
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load sample data for backtesting.

    Args:
        data_path: Path to the sample data file

    Returns:
        Tuple of (DataFrame, prices Series)
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} data points")

    # Use closing prices for the strategy
    prices = df['close']

    return df, prices


def create_ma_crossover_signals(
    prices: pd.Series,
    short_window: int = 10,
    long_window: int = 30
) -> pd.Series:
    """
    Create signals based on a simple moving average crossover strategy.

    Args:
        prices: Price series
        short_window: Short moving average window
        long_window: Long moving average window

    Returns:
        pandas Series with trading signals (-1 to 1)
    """
    # Calculate moving averages
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()

    # Create signals
    signals = pd.Series(0, index=prices.index)
    signals[short_ma > long_ma] = 1   # Buy signal
    signals[short_ma < long_ma] = -1  # Sell signal

    # Forward fill any NaN values
    signals = signals.fillna(0)

    print(f"Generated signals using {short_window}/{long_window} MA")
    print(f"Signal distribution: {signals.value_counts().to_dict()}")

    return signals


def main():
    """Main function to run the backtest demonstration."""
    try:
        # 1. Load sample data
        _, prices = load_sample_data()

        # 2. Create a simple example strategy (moving average crossover)
        signals = create_ma_crossover_signals(prices)

        # 3. Run the backtest using src/eval/backtest.py
        print("\nInitializing backtest engine...")
        engine = BacktestEngine(
            transaction_cost=0.001,  # 0.1% transaction cost
            slippage=0.0005,         # 0.05% slippage
            initial_capital=100000.0
        )

        print("Running backtest...")
        results = engine.run_backtest(signals, prices)
        print("Backtest completed successfully")

        # 4. Run stress tests
        print("Running stress tests...")
        stress_tester = StressTester(engine)
        stress_results = stress_tester.run_stress_tests(signals, prices)
        print("Stress tests completed")

        # 5. Generate all reports (HTML/CSV) to the /reports/ directory
        print("Generating reports...")
        report_generator = ReportGenerator("reports")

        # Generate CSV reports
        csv_path = report_generator.generate_csv_report(
            results, "backtest_results.csv"
        )
        print(f"CSV reports saved to: {csv_path}")

        # Generate HTML report
        html_path = report_generator.generate_html_report(
            results, stress_results, "backtest_report.html"
        )
        print(f"HTML report saved to: {html_path}")

        # Generate equity curve plot
        plot_path = report_generator.plot_equity_curve(
            results, "equity_curve.png"
        )
        print(f"Equity curve plot saved to: {plot_path}")

        # 6. Print a summary of key metrics to the console
        print("\n" + "="*50)
        print("BACKTEST RESULTS SUMMARY")
        print("="*50)
        metrics = results['metrics']
        print(f"CAGR: {metrics.get('cagr', 0):.4f}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
        print(f"Volatility: {metrics.get('volatility', 0):.4f}")
        print(f"Hit Ratio: {metrics.get('hit_ratio', 0):.4f}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.4f}")
        print(f"Turnover: {metrics.get('turnover', 0):.4f}")

        # Print stress test results summary
        print("\n" + "="*50)
        print("STRESS TEST RESULTS SUMMARY")
        print("="*50)
        for scenario, stress_result in stress_results.items():
            stress_metrics = stress_result['metrics']
            scenario_name = scenario.replace('_', ' ').title()
            print(f"\n{scenario_name}:")
            print(f"  CAGR: {stress_metrics.get('cagr', 0):.4f}")
            print(f"  Sharpe: {stress_metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  Max DD: {stress_metrics.get('max_drawdown', 0):.4f}")

        print("\n" + "="*50)
        print("Backtest demonstration completed successfully!")
        print("="*50)

    except Exception as e:
        print(f"Error running backtest: {e}")
        raise


if __name__ == "__main__":
    main()
