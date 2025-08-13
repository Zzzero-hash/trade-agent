# Feature Engineering Module

## Overview

The feature engineering module (`src/features/build.py`) computes technical indicators and features from OHLCV (Open, High, Low, Close, Volume) financial data while preventing data leakage through proper time-series handling.

## Features Computed

### 1. Log Returns

- Computed as `log(close_price_t / close_price_{t-1})`
- Measures the percentage change in price on a logarithmic scale

### 2. Rolling Statistics

- **Rolling Mean**: Moving average of log returns over specified windows (20, 60 periods)
- **Rolling Volatility**: Standard deviation of log returns over specified windows (20, 60 periods)

### 3. Average True Range (ATR)

- Measures market volatility by decomposing the entire range of an asset price for a given period
- Calculated using the greatest of:
  - Current high minus current low
  - Absolute value of current high minus previous close
  - Absolute value of current low minus previous close

### 4. Relative Strength Index (RSI)

- Momentum indicator that measures the speed and change of price movements
- Values range from 0 to 100
- Traditional interpretation:
  - RSI > 70: Overbought conditions
  - RSI < 30: Oversold conditions

### 5. Z-scores

- **Price Z-score**: How many standard deviations the current price is from its moving average
- **Volume Z-score**: How many standard deviations the current volume is from its moving average

### 6. Realized Volatility

- Standard deviation of log returns over a specified window
- Measures the actual volatility experienced over the period

### 7. Calendar Flags

- **Day of week**: Integer representation (0=Monday, 6=Sunday)
- **Month**: Month of the year (1-12)
- **Day of month**: Day of the month (1-31)
- **Is Monday**: Binary flag for Monday
- **Is Friday**: Binary flag for Friday
- **Is month start**: Binary flag for first day of month
- **Is month end**: Binary flag for last day of month

## Targets

### mu_hat (Expected Return)

- k-step forward return expectation
- Calculated as the mean of future log returns over the forecast horizon

### sigma_hat (Volatility Forecast)

- k-step volatility forecast
- Calculated as the standard deviation of future log returns over the forecast horizon

## Data Leakage Prevention

All features are computed with a `shift(1)` to ensure that at time `t`, only information available up to time `t-1` is used. This prevents data leakage from future values into current feature calculations.

Targets are computed using future data (which is appropriate for targets), but they are aligned with the feature timestamps.

## Command Line Interface

The module can be run from the command line:

```bash
python -m src.features.build --in data/raw.parquet --out data/fe.parquet --horizon 5
```

### Parameters

- `--in`: Path to input parquet file with OHLCV data
- `--out`: Path to output parquet file for feature data
- `--horizon`: Forecast horizon for targets (default: 5)

## Deterministic Results

The module uses fixed random seeds to ensure deterministic results across runs with the same input data.

## Testing

Unit tests verify:

1. No data leakage (features only use past/present data)
2. Deterministic results with fixed seeds
3. Correct computation of all technical indicators
4. Proper handling of edge cases

Run tests with:

```bash
python -m pytest tests/test_features.py
```
