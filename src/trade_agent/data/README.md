# Data Layer Implementation

This directory contains the implementation of the data layer for the trading system, including data loading and splitting functionality.

## Modules

### loaders.py

This module provides functionality to load OHLCV (Open, High, Low, Close, Volume) data from CSV files with proper data cleaning, timezone handling, and validation.

#### Features:

- Load OHLCV data from CSV files
- Enforce timezone (default UTC)
- Sort data by timestamp
- Drop duplicate rows
- Forward-fill missing values where appropriate
- Data integrity checks

#### Usage:

```bash
# Load data and run integrity checks
python -m src.data.loaders path/to/data.csv --check

# Load data only
python -m src.data.loaders path/to/data.csv
```

#### API:

```python
from src.data.loaders import load_ohlcv_data, check_data_integrity

# Load data
df = load_ohlcv_data('path/to/data.csv')

# Run checks
check_data_integrity(df)
```

### splits.py

This module provides functionality to create purged walk-forward splits for time series data, ensuring no data leakage across time boundaries.

#### Features:

- Purged walk-forward splits with train, validation, and test sets
- Strict temporal splits that yield (X_window_t, y_t, t_index) with no future information leakage
- Deterministic splits with fixed seeds
- Gap periods between splits to prevent data leakage

#### Usage:

```bash
# Run example splits
python -m src.data.splits
```

#### API:

```python
from src.data.splits import purged_walk_forward_splits, strict_temporal_split_generator
import pandas as pd

# Load your data
data = pd.read_csv('path/to/data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.set_index('timestamp')

# Generate purged walk-forward splits
for train, val, test in purged_walk_forward_splits(data, n_splits=5):
    # Process each split
    pass

# Generate strict temporal splits
for X_window, y_target, t_index in strict_temporal_split_generator(data, window_size=10):
    # Process each temporal split
    pass
```

## Requirements

- Python 3.7+
- pandas
- numpy

## Data Format

The expected CSV format for OHLCV data:

```csv
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,100.0,110.0,90.0,105.0,1000
2023-01-02 00:00:00,105.0,115.0,95.0,110.0,1200
...
```

## Running Tests

To verify the implementation works correctly:

```bash
# Test data loading
python -m src.data.loaders data/sample_data.csv --check

# Test data splitting
python -m src.data.splits
```
