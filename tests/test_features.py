"""
Unit tests for the feature engineering module.
"""

import numpy as np
import pandas as pd
import pytest

from trade_agent.features.build import (
    build_features,
    compute_atr,
    compute_calendar_flags,
    compute_log_returns,
    compute_realized_volatility,
    compute_rolling_stats,
    compute_rsi,
    compute_z_scores,
    define_targets,
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
    return df.tz_localize('UTC')



def test_no_future_data_leakage() -> None:
    """Test that no feature uses future data."""
    df = create_test_data()

    # Build features
    features = build_features(df, horizon=5)

    # Check that we have features
    assert len(features) > 0
    assert len(features.columns) > 0

    # For each feature column (except targets), verify that it doesn't
    # contain information that would not be available at time t
    feature_cols = [col for col in features.columns if col not in ['mu_hat', 'sigma_hat']]

    # We'll test a few key features to ensure they don't have future data
    # For example, log_returns at time t should only depend on prices up to time t
    for i in range(10, len(features)-10):  # Skip edges due to rolling windows
        t = features.index[i]
        features.index[i-1]

        # Get the feature values at time t
        features_t = features.loc[t, feature_cols]

        # Create a modified dataframe that only contains data up to time t
        df_up_to_t = df.loc[df.index <= t].copy()

        # Recompute features with only data up to time t
        features_up_to_t = build_features(df_up_to_t, horizon=5)

        # Check if we have data for time t in the recomputed features
        if t in features_up_to_t.index:
            recomputed_features_t = features_up_to_t.loc[t, feature_cols]

            # For non-target features, values should be the same
            # (allowing for small numerical differences)
            for col in feature_cols:
                if col in recomputed_features_t and col in features_t:
                    # Skip NaN values
                    if pd.isna(features_t[col]) and pd.isna(recomputed_features_t[col]):
                        continue

                    # Check that values are close (allowing for numerical precision)
                    np.testing.assert_almost_equal(
                        float(features_t[col]),
                        float(recomputed_features_t[col]),
                        decimal=10,
                        err_msg=f"Feature {col} at time {t} uses future data"
                    )


def test_deterministic_results() -> None:
    """Test that results are deterministic with fixed seeds."""
    df = create_test_data()

    # Build features twice
    features1 = build_features(df, horizon=5)
    features2 = build_features(df, horizon=5)

    # Results should be identical
    pd.testing.assert_frame_equal(features1, features2)


def test_log_returns_computation() -> None:
    """Test log returns computation."""
    df = create_test_data()
    log_returns = compute_log_returns(df)

    # Should have same index as input
    assert log_returns.index.equals(df.index)

    # First value should be NaN due to shift
    assert pd.isna(log_returns.iloc[0])

    # Check computation for a few values
    for i in range(1, min(10, len(df))):
        expected = np.log(df['close'].iloc[i] / df['close'].iloc[i-1])
        assert abs(log_returns.iloc[i] - expected) < 1e-10


def test_rolling_stats_computation() -> None:
    """Test rolling statistics computation."""
    df = create_test_data()
    windows = [5, 10]
    rolling_stats = compute_rolling_stats(df, windows)

    # Check that we have the expected columns
    expected_cols = []
    for window in windows:
        expected_cols.extend([f'rolling_mean_{window}', f'rolling_vol_{window}'])

    for col in expected_cols:
        assert col in rolling_stats.columns

    # Check that values are shifted (no data leakage)
    # First few values should be NaN due to shift
    for col in expected_cols:
        # At least window + 1 values should be NaN (rolling window + shift)
        window = int(col.split('_')[-1])
        assert pd.isna(rolling_stats[col].iloc[:window+1]).all()


def test_atr_computation() -> None:
    """Test ATR computation."""
    df = create_test_data()
    atr = compute_atr(df, window=14)

    # Should have same index as input
    assert atr.index.equals(df.index)

    # First few values should be NaN due to rolling window
    # With window=14, we expect the first 14 values to be NaN
    assert pd.isna(atr.iloc[:14]).all()


def test_rsi_computation() -> None:
    """Test RSI computation."""
    df = create_test_data()
    rsi = compute_rsi(df, window=14)

    # Should have same index as input
    assert rsi.index.equals(df.index)

    # First few values should be NaN due to rolling window
    # With window=14, we expect the first 14 values to be NaN
    assert pd.isna(rsi.iloc[:14]).all()

    # RSI values should be between 0 and 100 (when not NaN)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_z_scores_computation() -> None:
    """Test Z-scores computation."""
    df = create_test_data()
    z_scores = compute_z_scores(df, window=20)

    # Should have same index as input
    assert z_scores.index.equals(df.index)

    # Should have expected columns
    assert 'price_z_score' in z_scores.columns
    assert 'volume_z_score' in z_scores.columns

    # First few values should be NaN due to rolling window
    # With window=20, we expect the first 20 values to be NaN
    assert pd.isna(z_scores.iloc[:20]).all().all()


def test_realized_volatility_computation() -> None:
    """Test realized volatility computation."""
    df = create_test_data()
    rv = compute_realized_volatility(df, window=20)

    # Should have same index as input
    assert rv.index.equals(df.index)

    # First few values should be NaN due to rolling window and shift
    assert pd.isna(rv.iloc[:21]).all()


def test_calendar_flags_computation() -> None:
    """Test calendar flags computation."""
    df = create_test_data()
    calendar_flags = compute_calendar_flags(df)

    # Should have same index as input
    assert calendar_flags.index.equals(df.index)

    # Should have expected columns
    expected_cols = [
        'day_of_week', 'month', 'day_of_month', 'is_monday',
        'is_friday', 'is_month_start', 'is_month_end'
    ]

    for col in expected_cols:
        assert col in calendar_flags.columns

    # First value should be NaN due to shift
    assert pd.isna(calendar_flags.iloc[0]).all()


def test_targets_definition() -> None:
    """Test targets definition."""
    df = create_test_data()
    targets = define_targets(df, horizon=5)

    # Should have same index as input
    assert targets.index.equals(df.index)

    # Should have expected columns
    assert 'mu_hat' in targets.columns
    assert 'sigma_hat' in targets.columns

    # Check that targets use future data (they should)
    # This is allowed for targets, but not for features
    compute_log_returns(df)

    # mu_hat should be the mean of future log returns
    # sigma_hat should be the std of future log returns
    # We'll just check that they're computed (not NaN for most values)
    assert not targets['mu_hat'].dropna().empty
    assert not targets['sigma_hat'].dropna().empty


if __name__ == "__main__":
    pytest.main([__file__])
