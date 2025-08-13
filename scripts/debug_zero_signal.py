#!/usr/bin/env python3
"""
Debug script to understand zero signal strategy behavior.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.backtest import BacktestEngine


def load_sample_data(data_path: str = 'data/large_sample_data.parquet') -> pd.Series:
    """Load sample data for backtesting."""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} data points")
    return df['close']


def main():
    """Debug zero signal strategy."""
    print("Debugging zero signal strategy...")

    # Load sample data
    prices = load_sample_data()

    # Create zero signal strategy
    signals = pd.Series(0.0, index=prices.index)
    print(f"Created zero signals with {len(signals)} data points")
    print(f"Signal values: {signals.iloc[:10].tolist()}...")

    # Test without costs
    print("\n=== Testing without transaction costs ===")
    engine_no_cost = BacktestEngine(
        transaction_cost=0.0,
        slippage=0.0,
        initial_capital=100000.0
    )

    result_no_cost = engine_no_cost.run_backtest(signals, prices)
    print(f"Final net equity without costs: {result_no_cost['net_equity'].iloc[-1]:.2f}")
    print(f"CAGR without costs: {result_no_cost['metrics']['cagr']:.6f}")

    # Test with costs
    print("\n=== Testing with transaction costs ===")
    engine_with_cost = BacktestEngine(
        transaction_cost=0.001,
        slippage=0.0005,
        initial_capital=100000.0
    )

    result_with_cost = engine_with_cost.run_backtest(signals, prices)
    print(f"Final net equity with costs: {result_with_cost['net_equity'].iloc[-1]:.2f}")
    print(f"CAGR with costs: {result_with_cost['metrics']['cagr']:.6f}")

    # Check position changes
    print("\n=== Analyzing position changes ===")
    df = pd.DataFrame({
        'signal': signals,
        'price': prices
    }).dropna()

    # Calculate positions (shifted to avoid look-ahead bias)
    df['position'] = df['signal'].shift(1).fillna(0)
    df['position_change'] = df['position'].diff().fillna(0)
    df['transaction_costs'] = np.abs(df['position_change']) * (0.001 + 0.0005)

    print(f"Position changes: {df['position_change'].iloc[:10].tolist()}...")
    print(f"Transaction costs: {df['transaction_costs'].iloc[:10].tolist()}...")
    print(f"Total transaction costs: {df['transaction_costs'].sum():.6f}")

    # Check first few rows of data
    print("\n=== First 5 rows of data ===")
    print(df.head())


if __name__ == "__main__":
    main()
