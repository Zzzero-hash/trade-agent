"""
Dynamic TA-Lib wrapper for technical analysis indicators.

This module provides automatic access to all TA-Lib functions
with proper error handling, input validation, and pandas integration
"""
from __future__ import annotations

import pandas as pd


class TALibError(Exception):
    """Base exception for TA-Lib wrapper errors."""
    pass


class InsufficientDataError(TALibError):
    """Raised when insufficient data is provided to a TA-Lib function."""
    pass


def _validate_input_data(
    data: pd.Series | pd.DataFrame,
    required_columns: list[str] | None = None,
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Validate and prepare input data for TA-Lib functions.
    """
    if isinstance(data, pd.Series):
        df = data.to_frame(name=data.name or 'value')
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TALibError(f"Input data must be pandas Series or DataFrame, got {type(data)}")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise TALibError(f"Missing required columns: {missing_cols}")

    if len(df) < min_periods:
        raise InsufficientDataError(f"Insufficient data: {len(df)} periods < {min_periods} required"
        )

    # Handle timezone-aware timestamps
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df

def _to_pandas_result(
    result: Union[np.ndarray, list, tuple],
    index: pd.Index,
    name: List[str]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Convert TA-Lib result to pandas objects with proper indexing.
    """
    if isinstance(result, list | tuple):
        if len(result) == 1:
            # Single outptu - return Series
            return pd.Series(result[0], index=index, name=names[0] if names else 'indicator')
        # Multiple outputs = return DataFrame
        if not names or len(names) != len(result):
            names = [f'indicator_{i}' for i in range(len(result))]
            data_dict = {name: result[i] for i, name in enumerate(names)}
            return pd.DataFrame(data_dict, index=index)
    else:
        # Single array result
        name = names[0] if names else 'indicator'
        return pd.Series(result, index=index, name=name)
    return None
