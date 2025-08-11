"""
OHLCV Data Loader Module

This module provides functionality to load OHLCV (Open, High, Low, Close,
Volume) data from CSV files, with proper data cleaning, timezone handling,
and validation.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_ohlcv_data(
    file_path: str,
    timezone: str = "UTC",
    sort_by_date: bool = True,
    drop_duplicates: bool = True,
    forward_fill: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file with data cleaning and validation.

    Parameters:
        file_path (str): Path to the CSV file containing OHLCV data
        timezone (str): Timezone to enforce on the datetime index
            (default: "UTC")
        sort_by_date (bool): Whether to sort the data by date (default: True)
        drop_duplicates (bool): Whether to drop duplicate rows (default: True)
        forward_fill (bool): Whether to forward-fill missing values
            (default: True)

    Returns:
        pd.DataFrame: Cleaned and validated OHLCV data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If required columns are missing or data validation fails
    """
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load the data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Check for required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert timestamp to datetime and set as index
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    except Exception as e:
        raise ValueError(f"Error converting timestamp column: {e}")

    # Enforce timezone
    if df.index.tz is None:
        df = df.tz_localize(timezone)
    else:
        df = df.tz_convert(timezone)

    # Sort by date if requested
    if sort_by_date:
        df = df.sort_index()

    # Drop duplicates if requested
    if drop_duplicates:
        df = df.drop_duplicates()

    # Validate OHLCV data integrity
    # Check for negative prices or volume
    if (df[['open', 'high', 'low', 'close']] < 0).any().any():
        raise ValueError("Negative price values found in data")

    if (df['volume'] < 0).any():
        raise ValueError("Negative volume values found in data")

    # Check that high >= low for all rows
    if not (df['high'] >= df['low']).all():
        raise ValueError("High price is less than low price in some rows")

    # Forward-fill missing values if requested (but not for volume)
    if forward_fill:
        # Forward-fill price columns
        price_columns = ['open', 'high', 'low', 'close']
        df[price_columns] = df[price_columns].ffill()

        # For volume, we'll fill with 0 as it's more appropriate
        df['volume'] = df['volume'].fillna(0)

    return df


def check_data_integrity(df: pd.DataFrame) -> bool:
    """
    Run assertions to check data integrity.

    Parameters:
        df (pd.DataFrame): OHLCV data to check

    Returns:
        bool: True if all checks pass
    """
    print("Running data integrity checks...")

    # Check 1: Index is datetime with timezone
    assert isinstance(df.index, pd.DatetimeIndex), \
        "Index must be DatetimeIndex"
    assert df.index.tz is not None, "Index must have timezone information"
    print("✓ Index is datetime with timezone")

    # Check 2: Data is sorted by timestamp
    assert df.index.is_monotonic_increasing, "Data must be sorted by timestamp"
    print("✓ Data is sorted by timestamp")

    # Check 3: No duplicate timestamps
    assert not df.index.duplicated().any(), "Duplicate timestamps found"
    print("✓ No duplicate timestamps")

    # Check 4: Required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    print("✓ All required columns present")

    # Check 5: No negative prices or volume
    assert (df[['open', 'high', 'low', 'close']] >= 0).all().all(), \
        "Negative price values found"
    assert (df['volume'] >= 0).all(), "Negative volume values found"
    print("✓ No negative prices or volume")

    # Check 6: High >= Low for all rows
    assert (df['high'] >= df['low']).all(), \
        "High price is less than low price in some rows"
    print("✓ High price >= Low price for all rows")

    # Check 7: Data is not empty
    assert len(df) > 0, "Dataframe is empty"
    print("✓ Data is not empty")

    print("All data integrity checks passed!")
    return True


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="OHLCV Data Loader")
    parser.add_argument("file_path", nargs="?", help="Path to the CSV file")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run data integrity checks"
    )

    args = parser.parse_args()

    if not args.file_path:
        parser.print_help()
        sys.exit(1)

    try:
        # Load the data
        df = load_ohlcv_data(args.file_path)
        msg = f"Successfully loaded {len(df)} rows of data " \
              f"from {args.file_path}"
        print(msg)

        # Run checks if requested
        if args.check:
            check_data_integrity(df)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
