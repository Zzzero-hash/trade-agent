"""
Feature Engineering Module for Financial Time-Series Data.

This module provides functionality to compute various technical indicators
and features from OHLCV data while preventing data leakage.
"""

import argparse
import sys

import numpy as np
import pandas as pd

# Import the data loader
from src.data.loaders import load_ohlcv_data


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compute log returns from close prices.

    Parameters:
        df (pd.DataFrame): OHLCV data with 'close' column

    Returns:
        pd.Series: Log returns
    """
    return np.log(df['close'] / df['close'].shift(1))


def compute_rolling_stats(df: pd.DataFrame,
                         windows: list[int] = [20, 60]) -> pd.DataFrame:
    """
    Compute rolling mean and volatility for specified windows.

    Parameters:
        df (pd.DataFrame): OHLCV data
        windows (List[int]): List of window sizes for rolling computations

    Returns:
        pd.DataFrame: DataFrame with rolling mean and volatility features
    """
    features = pd.DataFrame(index=df.index)

    log_returns = compute_log_returns(df)

    for window in windows:
        # Rolling mean of log returns
        features[f'rolling_mean_{window}'] = (
            log_returns.rolling(window=window).mean().shift(1)
        )

        # Rolling volatility (standard deviation) of log returns
        features[f'rolling_vol_{window}'] = (
            log_returns.rolling(window=window).std().shift(1)
        )

    return features


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR) indicator.

    Parameters:
        df (pd.DataFrame): OHLCV data
        window (int): Window size for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    # Calculate True Range components
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))

    # True Range is the maximum of the three
    tr = pd.DataFrame({
        'high_low': high_low,
        'high_close': high_close,
        'low_close': low_close
    }).max(axis=1)

    # ATR is the rolling mean of True Range, shifted to prevent data leakage
    atr = tr.rolling(window=window).mean().shift(1)

    return atr


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) indicator.

    Parameters:
        df (pd.DataFrame): OHLCV data
        window (int): Window size for RSI calculation

    Returns:
        pd.Series: RSI values
    """
    # Calculate price changes
    delta = df['close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gains and losses with smoothing
    avg_gain = gain.rolling(window=window).mean().shift(1)
    avg_loss = loss.rolling(window=window).mean().shift(1)

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_z_scores(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute Z-scores for price and volume.

    Parameters:
        df (pd.DataFrame): OHLCV data
        window (int): Window size for mean and std calculation

    Returns:
        pd.DataFrame: DataFrame with Z-score features
    """
    features = pd.DataFrame(index=df.index)

    # Z-score for close price
    price_mean = df['close'].rolling(window=window).mean().shift(1)
    price_std = df['close'].rolling(window=window).std().shift(1)
    features['price_z_score'] = (df['close'] - price_mean) / price_std

    # Z-score for volume
    volume_mean = df['volume'].rolling(window=window).mean().shift(1)
    volume_std = df['volume'].rolling(window=window).std().shift(1)
    features['volume_z_score'] = (df['volume'] - volume_mean) / volume_std

    return features


def compute_realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute realized volatility (standard deviation of log returns).

    Parameters:
        df (pd.DataFrame): OHLCV data
        window (int): Window size for volatility calculation

    Returns:
        pd.Series: Realized volatility values
    """
    log_returns = compute_log_returns(df)
    realized_vol = log_returns.rolling(window=window).std().shift(1)
    return realized_vol


def compute_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute calendar-based features.

    Parameters:
        df (pd.DataFrame): OHLCV data with datetime index

    Returns:
        pd.DataFrame: DataFrame with calendar features
    """
    features = pd.DataFrame(index=df.index)

    # Day of week (0=Monday, 6=Sunday)
    features['day_of_week'] = df.index.dayofweek

    # Month of year
    features['month'] = df.index.month

    # Day of month
    features['day_of_month'] = df.index.day

    # Is Monday
    features['is_monday'] = (df.index.dayofweek == 0).astype(int)

    # Is Friday
    features['is_friday'] = (df.index.dayofweek == 4).astype(int)

    # Is month start
    features['is_month_start'] = (df.index.is_month_start).astype(int)

    # Is month end
    features['is_month_end'] = (df.index.is_month_end).astype(int)

    # Shift all features to prevent data leakage
    for col in features.columns:
        features[col] = features[col].shift(1)

    return features


def define_targets(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Define targets for supervised learning models.

    Parameters:
        df (pd.DataFrame): OHLCV data
        horizon (int): Forecast horizon for targets

    Returns:
        pd.DataFrame: DataFrame with target variables
    """
    targets = pd.DataFrame(index=df.index)

    # Calculate log returns
    log_returns = compute_log_returns(df)

    # mu_hat: k-step forward return expectation
    targets['mu_hat'] = log_returns.shift(-horizon).rolling(
        window=horizon).mean()

    # sigma_hat: k-step vol forecast
    targets['sigma_hat'] = log_returns.shift(-horizon).rolling(
        window=horizon).std()

    return targets


def build_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Build comprehensive feature set from OHLCV data.

    Parameters:
        df (pd.DataFrame): OHLCV data
        horizon (int): Forecast horizon for targets

    Returns:
        pd.DataFrame: DataFrame with all features and targets
    """
    # Set random seed for deterministic results
    np.random.seed(42)

    # Initialize features dataframe
    features = pd.DataFrame(index=df.index)

    # Compute log returns
    features['log_returns'] = compute_log_returns(df).shift(1)

    # Compute rolling statistics
    rolling_features = compute_rolling_stats(df)
    features = pd.concat([features, rolling_features], axis=1)

    # Compute ATR
    features['atr'] = compute_atr(df).shift(1)

    # Compute RSI
    features['rsi'] = compute_rsi(df).shift(1)

    # Compute Z-scores
    z_score_features = compute_z_scores(df)
    features = pd.concat([features, z_score_features], axis=1)

    # Compute realized volatility
    features['realized_vol'] = compute_realized_volatility(df).shift(1)

    # Compute calendar flags
    calendar_features = compute_calendar_flags(df)
    features = pd.concat([features, calendar_features], axis=1)

    # Define targets
    targets = define_targets(df, horizon)
    features = pd.concat([features, targets], axis=1)

    # Remove rows with NaN values (due to rolling windows and shifts)
    features = features.dropna()

    return features


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering for Financial Time-Series Data"
    )
    parser.add_argument(
        "--in",
        dest="input_file",
        required=True,
        help="Path to input parquet file with OHLCV data"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to output parquet file for feature data"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forecast horizon for targets (default: 5)"
    )

    args = parser.parse_args()

    try:
        # Load the data
        if args.input_file.endswith('.csv'):
            df = load_ohlcv_data(args.input_file)
        else:
            # For parquet files, load directly with pandas
            df = pd.read_parquet(args.input_file)

        print(f"Successfully loaded {len(df)} rows of data from {args.input_file}")

        # Build features
        features_df = build_features(df, args.horizon)
        print(f"Generated {len(features_df)} rows with {len(features_df.columns)} features")

        # Save features
        features_df.to_parquet(args.out)
        print(f"Features saved to {args.out}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
