"""
Data processing module for time series data
This module handles data validation, feature extraction, and transformation
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import ray
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


@ray.remote
def transform_yfinance_data_remote(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms a single-symbol DataFrame with error handling."""
    try:
        if df.empty or 'Close' not in df.columns:
            return pd.DataFrame()

        # Existing transformation logic
        df = df.rename(
            columns={
                'Date': 'timestamp',
                'Symbol': 'asset',
                'Close': 'value'
            }
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df[['timestamp', 'asset', 'value']]
    except Exception as e:
        logging.error(f"Transform failed: {str(e)}")
        return pd.DataFrame()



def handle_temporal_gaps(
    df: pd.DataFrame,
    interval: str = '1D',
    interpolation_method: str = 'linear',
    ffill_limit: int | None = None,
    bfill_limit: int | None = None
) -> pd.DataFrame:
    """
    Handles temporal gaps in a DataFrame by resampling and interpolating.

    Args:
        df: Input DataFrame with 'timestamp', 'asset', 'value' columns.
        interval: The frequency to resample to (e.g., '1D' for daily,
                  '1H' for hourly).
        interpolation_method: Method for interpolation (e.g., 'linear',
                              'polynomial', 'spline').
        ffill_limit: Maximum number of consecutive NaN values to forward fill.
        bfill_limit: Maximum number of consecutive NaN values to backward fill.

    Returns:
        DataFrame with temporal gaps addressed.
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure timestamp is the index for resampling
    df = df.set_index('timestamp').sort_index()

    # Group by asset and resample each group
    processed_dfs = []
    for asset, group in df.groupby('asset'):
        # Separate numeric and non-numeric columns
        numeric_columns = group.select_dtypes(include=['number']).columns

        # Resample to the desired frequency, only on numeric columns
        resampled_group = group[numeric_columns].resample(interval).mean()
        resampled_group['asset'] = asset  # Re-add asset column

        # Apply interpolation
        if interpolation_method:
            resampled_group['value'] = resampled_group['value'].interpolate(
                method=interpolation_method
            )

        # Apply forward and backward fill with limits
        if ffill_limit is not None:
            resampled_group['value'] = resampled_group['value'].ffill(
                limit=ffill_limit
            )
        if bfill_limit is not None:
            resampled_group['value'] = resampled_group['value'].bfill(
                limit=bfill_limit
            )

        processed_dfs.append(resampled_group)

    if not processed_dfs:
        return pd.DataFrame()

    # Concatenate all processed dataframes and reset index
    final_df = pd.concat(processed_dfs).reset_index()
    return final_df[['timestamp', 'asset', 'value']]


@ray.remote
def extract_ts_features_remote(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts tsfresh features for a single symbol."""
    if df.empty:
        return pd.DataFrame()

    # tsfresh expects column_id, column_sort, column_value
    ts_features = extract_features(
        df,
        column_id='asset',
        column_sort='timestamp',
        column_value='value',
        impute_function=impute,
        show_warnings=False
    )
    return ts_features


def align_symbol_data(
    symbol_features: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Aligns and merges feature DataFrames from multiple symbols.

    Args:
        symbol_features: A dictionary where keys are symbol names and
                         values are their corresponding feature DataFrames.

    Returns:
        A single DataFrame with all features aligned.
    """
    all_features = []
    for symbol, features in symbol_features.items():
        if not features.empty:
            # Add a 'symbol' column to identify the data source
            features['symbol'] = symbol
            all_features.append(features)

    if not all_features:
        return pd.DataFrame()

    # Concatenate all features into a single DataFrame
    final_df = pd.concat(all_features, ignore_index=True)

    # Ensure a consistent column order
    cols = ['symbol'] + [col for col in final_df.columns if col != 'symbol']
    final_df = final_df[cols]

    return final_df


def validate_data_structure(df: pd.DataFrame) -> None:
    """Validate input DataFrame meets requirements"""
    required_columns = {'timestamp', 'value', 'asset'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise TypeError("Timestamp column must be datetime type")


def check_data_quality(
    df: pd.DataFrame,
    max_temporal_gap_seconds: int = 86400,  # 1 day, configurable
    max_null_percentage: float = 0.05  # Configurable threshold
) -> dict[str, Any]:
    """Perform data quality checks and return metrics"""
    quality_report = {
        'null_values': df.isnull().sum().to_dict(),
        'row_count': len(df),
        'temporal_gaps': (
            df['timestamp'].diff().dt.total_seconds() >
            max_temporal_gap_seconds
        ).sum(),
        'zero_values': (df['value'] == 0).sum()
    }

    if (quality_report['null_values'].get('value', 0) >
            len(df) * max_null_percentage):
        logger.warning("High null value percentage detected")
    if quality_report['temporal_gaps'] > 0:
        logger.error(
            "Significant temporal gaps in data detected: "
            f"{quality_report['temporal_gaps']} gaps"
        )

    return quality_report


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 15
) -> list[str]:
    """Select top features using XGBoost feature importance"""
    model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        n_estimators=100,
        random_state=42,
        enable_categorical=True
    )

    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-n_features:]
    return X.columns[indices].tolist()

def extract_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time series features using tsfresh"""
    ts_features = extract_features(
        df,
        column_id='asset',
        column_sort='timestamp',
        column_value='value',
        impute_function=impute,
        show_warnings=False
    )
    return ts_features.reset_index()

def transform_yfinance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform yfinance MultiIndex DataFrame to expected format."""
    logger.info("Transforming yfinance data structure")

    # Check if data has MultiIndex columns (typical yfinance output)
    if isinstance(df.columns, pd.MultiIndex):
        logger.info("Processing MultiIndex DataFrame")

        # Reset index to get dates as a column
        # Reset index available if needed later
        df.reset_index()

        # Reshape data to long format
        data_rows = []

        # Get all unique symbols from the MultiIndex
        symbols = df.columns.get_level_values(1).unique()
        symbols = [s for s in symbols if s and s != '']
        logger.info(f"Found {len(symbols)} symbols")

        for symbol in symbols:
            try:
                symbol_data = df.xs(symbol, level=1, axis=1, drop_level=True)

                # Use Close price as the primary value
                if 'Close' in symbol_data.columns:
                    for idx, row in symbol_data.iterrows():
                        if pd.notna(row['Close']):
                            data_rows.append({
                                'timestamp': idx,
                                'asset': symbol,
                                'value': float(row['Close'])
                            })
            except Exception as e:
                logger.warning(f"Error processing symbol {symbol}: {e}")
                continue

        # Convert to DataFrame
        transformed_df = pd.DataFrame(data_rows)
        logger.info(f"Created {len(data_rows)} data points")

    else:
        # If data is already in simple format
        logger.info("Processing simple DataFrame")
        transformed_df = df.copy()

        # Map common column names to expected format
        column_mapping = {
            'Date': 'timestamp',
            'Symbol': 'asset',
            'Close': 'value'
        }

        transformed_df = transformed_df.rename(columns=column_mapping)

    # Ensure timestamp is datetime type with proper timezone handling
    if 'timestamp' in transformed_df.columns:
        transformed_df['timestamp'] = pd.to_datetime(
            transformed_df['timestamp'], utc=True
        )

    logger.info(
        f"Data transformation completed. Shape: {transformed_df.shape}"
    )
    return transformed_df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main processing pipeline"""
    try:
        # First transform the data structure if needed
        transformed_df = transform_yfinance_data(df)

        # Handle temporal gaps
        transformed_df = handle_temporal_gaps(
            transformed_df,
            interval='1D',  # This should be configurable
            interpolation_method='linear',
            ffill_limit=5,  # This should be configurable
            bfill_limit=5,  # This should be configurable
        )

        # Data validation
        validate_data_structure(transformed_df)
        check_data_quality(transformed_df)

        # Feature engineering
        # Convert categorical variables to numeric codes
        transformed_df['asset'] = (
            transformed_df['asset'].astype('category').cat.codes
        )
        X = transformed_df.drop(columns=['timestamp', 'value'])
        y = transformed_df['value']

        # Feature selection
        selected_features = select_features(X, y)
        filtered_data = X[selected_features]

        # Time series feature extraction
        ts_features = extract_ts_features(transformed_df)

        # Combine features
        combined = pd.concat(
            [filtered_data, ts_features],
            axis=1
        ).dropna(axis=1, how='all')
        return combined  # type: ignore

    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise
