"""
Dynamic TA-Lib wrapper for technical analysis indicators.

This module provides automatic access to all TA-Lib functions
with proper error handling, input validation, and pandas integration
"""
from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
import talib


class TALibError(Exception):
    """Base exception for TA-Lib wrapper errors."""
    pass


class InsufficientDataError(TALibError):
    """Raised when insufficient data is provided to a TA-Lib function."""
    pass


def _validate_input_data(
    data: pd.Series | pd.DataFrame | tuple,
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
    elif isinstance(data, tuple) and len(data) == 4:
        # Handle OHLC tuple - create DataFrame from the 4 series
        if all(isinstance(item, pd.Series) for item in data):
            df = pd.DataFrame({
                'open': data[0],
                'high': data[1],
                'low': data[2],
                'close': data[3]
            })
        else:
            raise TALibError(f"Tuple input must contain 4 pandas Series, got {type(data)}")
    else:
        raise TALibError(f"Input data must be pandas Series, DataFrame, or tuple of 4 Series, got {type(data)}")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise TALibError(f"Missing required columns: {missing_cols}")

    if len(df) < min_periods:
        raise InsufficientDataError(f"Insufficient data: {len(df)} periods < {min_periods} required")

    # Handle timezone-aware timestamps
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def _to_pandas_result(
    result: np.ndarray | list | tuple,
    index: pd.Index,
    names: list[str] | None = None
) -> pd.Series | pd.DataFrame:
    """
    Convert TA-Lib result to pandas objects with proper indexing.
    """
    if isinstance(result, list | tuple):
        if len(result) == 1:
            # Single output - return Series
            name = names[0] if names else 'indicator'
            return pd.Series(result[0], index=index, name=name)
        # Multiple outputs = return DataFrame
        if not names or len(names) != len(result):
            names = [f'indicator_{i}' for i in range(len(result))]
        data_dict = {name: result[i] for i, name in enumerate(names)}
        return pd.DataFrame(data_dict, index=index)
    # Single array result
    name = names[0] if names else 'indicator'
    return pd.Series(result, index=index, name=name)


class TALibWrapper:
    """Dynamic wrapper for TA-Lib functions with pandas integration."""

    def __init__(self) -> None:
        self._function_cache = {}
        self._initialize_functions()

    def _initialize_functions(self) -> None:
        """Initialize all available TA-Lib functions."""
        # Get all functions from talib
        for name in dir(talib):
            func = getattr(talib, name)
            if callable(func) and not name.startswith('_'):
                # Check if it's a TA-Lib function by looking at its signature
                try:
                    sig = inspect.signature(func)
                    if 'real' in sig.parameters or 'open' in sig.parameters:
                        self._function_cache[name] = func
                except (ValueError, TypeError):
                    # Skip functions that cannot be inspected
                    continue

    def get_available_functions(self) -> list[str]:
        """Get list of all available TA-Lib functions."""
        return sorted(list(self._function_cache.keys()))

    def get_function_info(self, function_name: str) -> dict[str, Any]:
        """Get information about a specific TA-Lib function."""
        if function_name not in self._function_cache:
            raise TALibError(f"Function '{function_name}' not found. Available functions are: {self.get_available_functions()}")

        func = self._function_cache[function_name]
        try:
            sig = inspect.signature(func)
            return {
                'name': function_name,
                'signature': str(sig),
                'docstring': func.__doc__ or "No documentation available",
                'parameters': list(sig.parameters.keys())
            }
        except (ValueError, TypeError):
            return {
                'name': function_name,
                'signature': 'unknown',
                'docstring': 'unknown',
                'parameters': []
            }

    def call_with_params(self, function_name: str, data: pd.Series | pd.DataFrame | tuple,
                         params: dict[str, Any] = None) -> pd.Series | pd.DataFrame:
        """
        Call a TA-Lib function with parameters.

        Args:
            function_name: Name of the TA-Lib function to call
            data: Input data (pandas Series or DataFrame)
            params: Dictionary of parameters for the function

        Returns:
            pandas Series or DataFrame with the result
        """
        if function_name not in self._function_cache:
            raise TALibError(f"Function '{function_name}' not found. Available functions are: {self.get_available_functions()}")

        params = params or {}
        func = self._function_cache[function_name]

        # Convert pandas data to numpy arrays for TA-Lib
        df = _validate_input_data(data)

        # Prepare arguments based on function signature
        sig = inspect.signature(func)
        args = []
        kwargs = params.copy()

        # Handle common TA-Lib patterns
        if 'real' in sig.parameters:
            # Check if this is a function that specifically requires 'close' data
            close_required_functions = ['MACD', 'RSI', 'BBANDS', 'STOCH', 'STOCHF', 'STOCHRSI']
            if function_name in close_required_functions:
                # These functions specifically require 'close' price data
                if 'close' in df.columns:
                    kwargs['real'] = df['close'].values.astype(np.float64)
                elif len(df.columns) == 1:
                    # For single column data, use that column (common case for Series input)
                    kwargs['real'] = df.iloc[:, 0].values.astype(np.float64)
                else:
                    raise TALibError(
                        f"Function '{function_name}' requires 'close' price data but DataFrame "
                        f"has columns: {list(df.columns)}. Please provide a DataFrame with a 'close' column "
                        f"or a single-column Series."
                    )
            elif len(df.columns) == 1:
                # Single column - use it directly (unless it's a close-required function)
                kwargs['real'] = df.iloc[:, 0].values.astype(np.float64)
            elif 'close' in df.columns:
                # If we have a 'close' column, use that for 'real' parameter
                kwargs['real'] = df['close'].values.astype(np.float64)
            else:
                # For other functions, try to use the first column
                kwargs['real'] = df.iloc[:, 0].values.astype(np.float64)
        elif all(col in sig.parameters for col in ['open', 'high', 'low', 'close']):
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise TALibError(
                    f"Missing required columns for OHLC function: {missing_cols}"
                )
            for col in required_cols:
                kwargs[col] = df[col].values.astype(np.float64)
        elif 'close' in sig.parameters:
            if 'close' in df.columns:
                kwargs['close'] = df['close'].values.astype(np.float64)
            elif len(df.columns) == 1:
                # For functions that require 'close' but we have single column, use that column
                kwargs['close'] = df.iloc[:, 0].values.astype(np.float64)
            else:
                raise TALibError(
                    "Function requires 'close' parameter but suitable column not found"
                )

        # Handle volume parameter
        if 'volume' in sig.parameters and 'volume' in df.columns:
            kwargs['volume'] = df['volume'].values.astype(np.float64)

        # Handle OHLC functions that take multiple positional arguments
        # Some TA-Lib functions expect positional arguments rather than keyword args
        if (len(args) == 0 and len(kwargs) == 0 and
            all(col in df.columns for col in ['open', 'high', 'low', 'close'])):
            # This is likely an OHLC function that needs positional args
            required_cols = ['open', 'high', 'low', 'close']
            if all(col in sig.parameters for col in required_cols):
                args = [df[col].values.astype(np.float64) for col in required_cols]
                kwargs = {}

        # Check minimum data requirements for TA-Lib functions
        min_periods_needed = params.get('timeperiod', 1)
        if len(df) < min_periods_needed and 'real' in kwargs:
            # For simple indicator functions, check if we have enough data
            raise InsufficientDataError(
                f"Insufficient data for function '{function_name}': "
                f"{len(df)} periods < {min_periods_needed} required"
            )

        # Special handling for functions that require specific columns but don't use keyword args
        if (len(args) == 0 and len(kwargs) == 0 and
            'close' in sig.parameters and 'close' not in df.columns and len(df.columns) == 1):
            # This might be a function that expects 'close' but we have a single column
            # Try to map the single column to 'close'
            kwargs['close'] = df.iloc[:, 0].values.astype(np.float64)

        # Special handling for edge cases
        timeperiod = params.get('timeperiod', 1)
        if (function_name == 'SMA' and timeperiod == 1 and len(df) == 1 and
            'real' in kwargs and len(kwargs['real']) == 1):
            # Handle SMA with timeperiod=1 and single value - return the value directly
            return pd.Series([kwargs['real'][0]], index=df.index, name='indicator')

        try:
            result = func(*args, **kwargs)
            return _to_pandas_result(result, df.index)
        except Exception as e:
            raise TALibError(f"Error calling function '{function_name}': {str(e)}")

    def __getattr__(self, name: str):
        """Enable dynamic access to TA-Lib functions."""
        if name in self._function_cache:
            def wrapper(*args, **kwargs):
                # Handle OHLC functions that take multiple positional arguments
                if len(args) == 4 and all(isinstance(arg, pd.Series | np.ndarray) for arg in args):
                    # This is likely an OHLC function call with 4 separate arguments
                    return self.call_with_params(name, args, kwargs)
                if len(args) == 1 and isinstance(args[0], pd.Series | pd.DataFrame):
                    # This is a normal single data argument call
                    return self.call_with_params(name, args[0], kwargs)
                raise TALibError(f"Invalid arguments for function '{name}'. Expected either 1 data argument or 4 OHLC arguments.")
            return wrapper
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Create global instance for easy access
talib_wrapper = TALibWrapper()

# Export main classes and functions
__all__ = ['TALibWrapper',
           'TALibError',
           'InsufficientDataError',
           'talib_wrapper']
