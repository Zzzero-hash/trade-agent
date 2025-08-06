import logging
from typing import Any

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def validate_data_structure(df: pd.DataFrame) -> None:
    """Validate input DataFrame meets requirements"""
    required_columns = {'timestamp', 'value', 'asset'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise TypeError("Timestamp column must be datetime type")


def check_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Perform data quality checks and return metrics"""
    quality_report = {
        'null_values': df.isnull().sum().to_dict(),
        'row_count': len(df),
        'temporal_gaps': (
            df['timestamp'].diff().dt.total_seconds() > 3600  # type: ignore
        ).sum(),
        'zero_values': (df['value'] == 0).sum()
    }

    if quality_report['null_values'].get('value', 0) > len(df)*0.05:
        logger.warning("High null value percentage detected")
    if quality_report['temporal_gaps'] > 0:
        logger.error("Significant temporal gaps in data")

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

    # Ensure timestamp is datetime type
    if 'timestamp' in transformed_df.columns:
        transformed_df['timestamp'] = pd.to_datetime(
            transformed_df['timestamp']
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
