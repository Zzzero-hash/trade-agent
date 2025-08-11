"""
Script to debug feature computation and understand NaN patterns.
"""

import numpy as np
import pandas as pd

from src.features.build import (
    compute_atr,
    compute_realized_volatility,
    compute_rsi,
    compute_z_scores,
)


def create_test_data():
    """Create sample OHLCV data for testing."""
    # Create a simple time series with predictable patterns
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducible tests

    # Create a simple price series with trend and noise
    prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, 100))

    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'close': prices,
        'volume': np.random.uniform(1000, 2000, 100)
    }, index=dates)

    # Add timezone to match data loader expectations
    df = df.tz_localize('UTC')

    return df


def main():
    """Debug feature computation."""
    df = create_test_data()

    print("ATR computation:")
    atr = compute_atr(df, window=14)
    print(f"ATR shape: {atr.shape}")
    print(f"ATR NaN count: {atr.isna().sum()}")
    print("First 20 ATR values:")
    for i in range(min(20, len(atr))):
        val = atr.iloc[i]
        print(f"  {i}: {val} (NaN: {pd.isna(val)})")

    print("\nRSI computation:")
    rsi = compute_rsi(df, window=14)
    print(f"RSI shape: {rsi.shape}")
    print(f"RSI NaN count: {rsi.isna().sum()}")
    print("First 20 RSI values:")
    for i in range(min(20, len(rsi))):
        val = rsi.iloc[i]
        print(f"  {i}: {val} (NaN: {pd.isna(val)})")

    print("\nZ-scores computation:")
    z_scores = compute_z_scores(df, window=20)
    print(f"Z-scores shape: {z_scores.shape}")
    print(f"Price Z-score NaN count: {z_scores['price_z_score'].isna().sum()}")
    print(f"Volume Z-score NaN count: {z_scores['volume_z_score'].isna().sum()}")
    print("First 25 Z-score values:")
    for i in range(min(25, len(z_scores))):
        price_val = z_scores['price_z_score'].iloc[i]
        vol_val = z_scores['volume_z_score'].iloc[i]
        print(f"  {i}: price={price_val} (NaN: {pd.isna(price_val)}), volume={vol_val} (NaN: {pd.isna(vol_val)})")

    print("\nRealized volatility computation:")
    rv = compute_realized_volatility(df, window=20)
    print(f"Realized volatility shape: {rv.shape}")
    print(f"Realized volatility NaN count: {rv.isna().sum()}")
    print("First 25 RV values:")
    for i in range(min(25, len(rv))):
        val = rv.iloc[i]
        print(f"  {i}: {val} (NaN: {pd.isna(val)})")


if __name__ == "__main__":
    main()
