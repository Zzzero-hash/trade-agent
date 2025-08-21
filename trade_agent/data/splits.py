"""
Purged Walk-Forward Splits Module

This module provides functionality to create purged walk-forward splits for
time series data, ensuring no data leakage across time boundaries.
"""

import warnings
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd


def purged_walk_forward_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    min_gap: int = 5,
    max_gap: int = 21,
    fixed_seed: int = 42
) -> Generator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generate purged walk-forward splits with train, validation, and test sets.

    Parameters:
        data (pd.DataFrame): Time series data sorted by timestamp
        n_splits (int): Number of splits to generate (default: 5)
        train_ratio (float): Proportion of data for training (default: 0.6)
        val_ratio (float): Proportion of data for validation (default: 0.2)
        test_ratio (float): Proportion of data for testing (default: 0.2)
        gap_ratio (float): Proportion of data to gap between splits
            (default: 0.05)
        fixed_seed (int): Random seed for reproducibility (default: 42)

    Yields:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (train_data, val_data, test_data) for each split

    Raises:
        ValueError: If ratios don't sum to 1.0 or data is invalid
    """
    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    if len(data) == 0:
        raise ValueError("Data cannot be empty")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex")

    # Check if data is sorted
    if not data.index.is_monotonic_increasing:
        warnings.warn("Data is not sorted by timestamp. Sorting now.", stacklevel=2)
        data = data.sort_index()

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    if any(ratio < 0 for ratio in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("Ratios must be non-negative")

    # Set random seed for reproducibility
    np.random.seed(fixed_seed)

    # Calculate split sizes
    total_length = len(data)
    split_size = total_length // n_splits

    # Calculate gap size
    gap_size = min_gap  # Use defined min_gap parameter instead of undefined gap_ratio

    # Generate splits
    for i in range(n_splits):
        # Calculate start and end indices for this split
        split_start = i * split_size
        split_end = min((i + 1) * split_size, total_length)

        # Ensure we have enough data for all sets
        if split_end - split_start < 3:
            warnings.warn(f"Split {i} has insufficient data. Skipping.", stacklevel=2)
            continue

        # Calculate boundaries for train, val, test with gaps
        train_end = split_start + int((split_end - split_start) * train_ratio)
        val_end = train_end + int((split_end - split_start) * val_ratio)

        # Apply gaps to prevent data leakage
        train_start = split_start
        train_end_purged = max(train_start + 1, train_end - gap_size)

        val_start = min(train_end + gap_size, val_end - gap_size)
        val_end_purged = max(val_start + 1, val_end - gap_size)

        test_start = min(val_end + gap_size, split_end)
        test_end_purged = split_end

        # Extract data segments
        if train_end_purged > train_start:
            train_data = data.iloc[train_start:train_end_purged]
        else:
            train_data = pd.DataFrame()

        if val_end_purged > val_start:
            val_data = data.iloc[val_start:val_end_purged]
        else:
            val_data = pd.DataFrame()

        if test_end_purged > test_start:
            test_data = data.iloc[test_start:test_end_purged]
        else:
            test_data = pd.DataFrame()

        # Yield the split
        yield train_data, val_data, test_data


def strict_temporal_split_generator(
    data: pd.DataFrame,
    window_size: int,
    step_size: int = 1,
    min_samples: int = 10,
    fixed_seed: int = 42
) -> Generator[tuple[pd.DataFrame, Any, pd.DatetimeIndex], None, None]:
    """
    Generate strict temporal splits that yield (X_window_t, y_t, t_index)
    with no future information leakage.

    Parameters:
        data (pd.DataFrame): Time series data sorted by timestamp
        window_size (int): Size of the rolling window for features
        step_size (int): Step size between windows (default: 1)
        min_samples (int): Minimum samples required for a window (default: 10)
        fixed_seed (int): Random seed for reproducibility (default: 42)

    Yields:
        Tuple[pd.DataFrame, Any, pd.DatetimeIndex]:
            (X_window_t, y_t, t_index) where:
            - X_window_t: Feature window of size window_size
            - y_t: Target value at time t (next time step after window)
            - t_index: Timestamp index of the target

    Raises:
        ValueError: If data is invalid or window_size is too large
    """
    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    if len(data) == 0:
        raise ValueError("Data cannot be empty")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex")

    # Check if data is sorted
    if not data.index.is_monotonic_increasing:
        warnings.warn("Data is not sorted by timestamp. Sorting now.", stacklevel=2)
        data = data.sort_index()

    if window_size <= 0:
        raise ValueError("Window size must be positive")

    if window_size >= len(data):
        raise ValueError("Window size must be smaller than data length")

    if step_size <= 0:
        raise ValueError("Step size must be positive")

    # Set random seed for reproducibility
    np.random.seed(fixed_seed)

    # Generate temporal splits
    for i in range(window_size, len(data), step_size):
        # Extract feature window (X_window_t)
        X_window_t = data.iloc[i - window_size:i]

        # Extract target value (y_t) - next time step after window
        y_t = data.iloc[i]

        # Extract timestamp index of target
        t_index = data.index[i:i+1]

        # Check minimum samples requirement
        if len(X_window_t) < min_samples:
            continue

        # Yield the temporal split
        yield X_window_t, y_t, t_index


# Example usage function
def example_usage() -> None:
    """
    Example of how to use the purged walk-forward splits and temporal
    generator.
    """
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    data = pd.DataFrame({
        'open': np.random.rand(1000) * 100,
        'high': np.random.rand(1000) * 100 + 10,
        'low': np.random.rand(1000) * 100 - 10,
        'close': np.random.rand(1000) * 100,
        'volume': np.random.rand(1000) * 10000
    }, index=dates)


    # Generate splits
    split_count = 0
    for train, val, test in purged_walk_forward_splits(
        data,
        n_splits=3,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    ):
        split_count += 1

        if len(train) > 0:
            pass
        if len(val) > 0:
            pass
        if len(test) > 0:
            pass


    # Generate temporal splits
    window_count = 0
    for X_window, _y_target, _t_index in strict_temporal_split_generator(
        data,
        window_size=10,
        step_size=5
    ):
        window_count += 1
        if window_count > 3:  # Only show first 3 for brevity
            break
        f"  X_window time range: {X_window.index[0]} " \
                    f"to {X_window.index[-1]}"


if __name__ == "__main__":
    example_usage()
