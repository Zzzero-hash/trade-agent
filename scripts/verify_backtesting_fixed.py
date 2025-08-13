#!/usr/bin/env python3
"""
Corrected verification tests for the backtesting implementation.

This script verifies:
1. Reproducibility with fixed seed
2. Performance degradation with worsening costs/slippage
3. Sanity baseline check with zero signal strategy (corrected)
"""

import os
import sys

import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.backtest import BacktestEngine


def load_sample_data(data_path: str = 'data/large_sample_data.parquet') -> pd.Series:
    """
    Load sample data for backtesting.

    Args:
        data_path: Path to the sample data file

    Returns:
        Price series
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} data points")

    # Use closing prices for the strategy
    prices = df['close']

    return prices


def create_zero_signal_strategy(prices: pd.Series) -> pd.Series:
    """
    Create a zero signal strategy (flat or random).

    Args:
        prices: Price series

    Returns:
        pandas Series with zero signals
    """
    # Create zero signals (flat strategy)
    signals = pd.Series(0.0, index=prices.index)
    print("Created zero signal strategy")
    return signals


def create_oscillating_signal_strategy(prices: pd.Series) -> pd.Series:
    """
    Create an oscillating signal strategy that generates trades.

    Args:
        prices: Price series

    Returns:
        pandas Series with oscillating signals between -1 and 1
    """
    # Create oscillating signals that will generate trades
    signals = pd.Series([1, -1] * (len(prices) // 2), index=prices.index[:len(prices)//2*2])
    if len(signals) < len(prices):
        signals = pd.concat([signals, pd.Series([1], index=[prices.index[len(signals)]])])
    print("Created oscillating signal strategy")
    return signals


def test_reproducibility(prices: pd.Series, n_runs: int = 3) -> bool:
    """
    Test reproducibility with fixed seed.

    Args:
        prices: Price series
        n_runs: Number of runs to test

    Returns:
        True if results are identical, False otherwise
    """
    print("\n=== Test 1: Reproducibility with fixed seed ===")

    # Create a simple signal strategy
    signals = create_oscillating_signal_strategy(prices)

    # Run backtest multiple times with same parameters
    results = []
    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}")
        engine = BacktestEngine(
            transaction_cost=0.001,
            slippage=0.0005,
            initial_capital=100000.0
        )
        result = engine.run_backtest(signals, prices)
        results.append(result)

    # Check if all results are identical
    cagrs = [result['metrics']['cagr'] for result in results]
    sharpe_ratios = [result['metrics']['sharpe_ratio'] for result in results]
    max_drawdowns = [result['metrics']['max_drawdown'] for result in results]

    cagr_identical = all(abs(c - cagrs[0]) < 1e-10 for c in cagrs)
    sharpe_identical = all(abs(s - sharpe_ratios[0]) < 1e-10 for s in sharpe_ratios)
    max_dd_identical = all(abs(dd - max_drawdowns[0]) < 1e-10 for dd in max_drawdowns)

    print(f"CAGR values: {cagrs}")
    print(f"Sharpe ratios: {sharpe_ratios}")
    print(f"Max drawdowns: {max_drawdowns}")

    if cagr_identical and sharpe_identical and max_dd_identical:
        print("✓ Reproducibility test PASSED")
        return True
    else:
        print("✗ Reproducibility test FAILED")
        return False


def test_cost_degradation(prices: pd.Series) -> bool:
    """
    Test performance degradation with worsening costs/slippage.

    Args:
        prices: Price series

    Returns:
        True if performance degrades with higher costs, False otherwise
    """
    print("\n=== Test 2: Performance degradation with worsening costs/slippage ===")

    # Create a strategy that generates trades
    signals = create_oscillating_signal_strategy(prices)

    # Test with different cost levels
    cost_levels = [
        (0.0001, 0.00005),  # Low costs
        (0.001, 0.0005),    # Medium costs (default)
        (0.005, 0.0025)     # High costs
    ]

    results = []
    for transaction_cost, slippage in cost_levels:
        print(f"Testing with transaction_cost={transaction_cost}, slippage={slippage}")
        engine = BacktestEngine(
            transaction_cost=transaction_cost,
            slippage=slippage,
            initial_capital=100000.0
        )
        result = engine.run_backtest(signals, prices)
        results.append(result)
        cagr = result['metrics']['cagr']
        print(f"  CAGR: {cagr:.6f}")

    # Check if performance degrades with higher costs
    cagrs = [result['metrics']['cagr'] for result in results]

    # Performance should degrade (CAGR should decrease) with higher costs
    # We expect: cagrs[0] >= cagrs[1] >= cagrs[2]
    performance_degrades = cagrs[0] >= cagrs[1] >= cagrs[2]

    print(f"CAGR values across cost levels: {cagrs}")

    if performance_degrades:
        print("✓ Cost degradation test PASSED")
        return True
    else:
        print("✗ Cost degradation test FAILED")
        return False


def test_sanity_baseline(prices: pd.Series) -> bool:
    """
    Test sanity baseline with zero signal strategy.

    Args:
        prices: Price series

    Returns:
        True if sanity checks pass, False otherwise
    """
    print("\n=== Test 3: Sanity baseline check with zero signal strategy ===")

    # Create zero signal strategy
    signals = create_zero_signal_strategy(prices)

    # Test without costs
    print("Testing without transaction costs...")
    engine_no_cost = BacktestEngine(
        transaction_cost=0.0,
        slippage=0.0,
        initial_capital=100000.0
    )
    result_no_cost = engine_no_cost.run_backtest(signals, prices)
    cagr_no_cost = result_no_cost['metrics']['cagr']
    print(f"  CAGR without costs: {cagr_no_cost:.6f}")

    # Test with costs
    print("Testing with transaction costs...")
    engine_with_cost = BacktestEngine(
        transaction_cost=0.001,
        slippage=0.0005,
        initial_capital=100000.0
    )
    result_with_cost = engine_with_cost.run_backtest(signals, prices)
    cagr_with_cost = result_with_cost['metrics']['cagr']
    print(f"  CAGR with costs: {cagr_with_cost:.6f}")

    # Check that performance ≈ 0 before costs (no trades, so no returns)
    performance_near_zero_no_cost = abs(cagr_no_cost) < 0.0001  # Very close to zero

    # Check that performance ≈ 0 after costs (no trades, so no costs)
    performance_near_zero_with_cost = abs(cagr_with_cost) < 0.0001  # Very close to zero

    print(f"Performance near zero without costs: {performance_near_zero_no_cost}")
    print(f"Performance near zero with costs: {performance_near_zero_with_cost}")

    if performance_near_zero_no_cost and performance_near_zero_with_cost:
        print("✓ Sanity baseline test PASSED")
        return True
    else:
        print("✗ Sanity baseline test FAILED")
        return False


def main():
    """Main function to run all verification tests."""
    print("Running backtesting verification tests...")

    try:
        # Load sample data
        prices = load_sample_data()

        # Run all tests
        test1_passed = test_reproducibility(prices)
        test2_passed = test_cost_degradation(prices)
        test3_passed = test_sanity_baseline(prices)

        print("\n" + "="*60)
        print("VERIFICATION TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Test 1 - Reproducibility: {'PASSED' if test1_passed else 'FAILED'}")
        print(f"Test 2 - Cost degradation: {'PASSED' if test2_passed else 'FAILED'}")
        print(f"Test 3 - Sanity baseline: {'PASSED' if test3_passed else 'FAILED'}")

        all_tests_passed = test1_passed and test2_passed and test3_passed

        if all_tests_passed:
            print("\n✓ All verification tests PASSED")
            print("\nThe backtesting implementation is working correctly.")
            print("1. Results are reproducible with fixed seed")
            print("2. Performance degrades as costs increase")
            print("3. Zero signal strategy behaves as expected (no trades, no returns)")
            return 0
        else:
            print("\n✗ Some verification tests FAILED")
            return 1

    except Exception as e:
        print(f"Error running verification tests: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
