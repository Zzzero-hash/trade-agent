"""
Script to verify that the feature engineering produces deterministic results.
"""

import numpy as np
import pandas as pd

from src.features.build import build_features


def main():
    """Verify deterministic results with fixed seeds."""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
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

    # Build features twice
    print("Building features first time...")
    features1 = build_features(df, horizon=5)

    print("Building features second time...")
    features2 = build_features(df, horizon=5)

    # Check if they're identical
    try:
        pd.testing.assert_frame_equal(features1, features2)
        print("SUCCESS: Results are deterministic - feature engineering "
              "produces identical results")
    except AssertionError as e:
        print("ERROR: Results are not deterministic - feature engineering "
              "produces different results")
        print(f"Error: {e}")
        return False

    # Show some statistics
    print("\nFeature statistics:")
    print(f"Number of rows: {len(features1)}")
    print(f"Number of columns: {len(features1.columns)}")
    print(f"Columns: {list(features1.columns)}")

    return True


if __name__ == "__main__":
    main()
