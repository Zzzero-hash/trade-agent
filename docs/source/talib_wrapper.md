# TA-Lib Wrapper Documentation

## Overview

The TA-Lib wrapper provides integration with the popular technical analysis library, offering access to over 150 technical indicators and functions. The current implementation provides a foundation for technical analysis within the trading system.

## Current Implementation Status

The wrapper is implemented at `trade_agent/engine/nodes/talib_wrapper.py` and is fully functional with comprehensive features. All tests are passing and the implementation is complete and working in production:

### Core Components

1. **Exception Classes**
   - `TALibError`: Base exception for wrapper errors
   - `InsufficientDataError`: Raised when insufficient data is provided to a TA-Lib function

2. **Main Wrapper Class**
   - `TALibWrapper`: Dynamic wrapper for all TA-Lib functions with pandas integration
   - Global instance `talib_wrapper` for easy access

3. **Helper Functions**
   - `_validate_input_data()`: Validates and prepares input data for TA-Lib functions
   - `_to_pandas_result()`: Converts TA-Lib results to pandas objects with proper indexing

### Key Features

The implementation provides:

- **Dynamic Function Access**: Automatic access to all 150+ TA-Lib functions via `__getattr__`
- **Pandas Integration**: Seamless Series/DataFrame input and output handling with proper indexing
- **Smart Parameter Mapping**: Automatic mapping of DataFrame columns to TA-Lib parameters
- **Comprehensive Error Handling**: Robust error handling for edge cases and invalid inputs
- **OHLC Support**: Full support for OHLC-based functions with multiple input arguments
- **Type Safety**: Proper type hints and annotations throughout
- **Extensive Test Coverage**: 18 comprehensive unit tests covering all major functionality

## Usage Examples

### Current Usage (Working)

```python
import pandas as pd
from trade_agent.engine.nodes.talib_wrapper import talib_wrapper

# Load market data
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
    'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
    'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
})

# Calculate Simple Moving Average
sma = talib_wrapper.SMA(data['close'], timeperiod=3)
print(sma.tail())

# Calculate RSI
rsi = talib_wrapper.RSI(data['close'], timeperiod=14)
print(rsi.tail())

# Calculate MACD (multiple outputs)
macd_df = talib_wrapper.MACD(data['close'])
print(macd_df.tail())

# Calculate Bollinger Bands
bbands_df = talib_wrapper.BBANDS(data['close'])
print(bbands_df.tail())

# Calculate Average Price using OHLC data
avg_price = talib_wrapper.AVGPRICE(
    data['open'], data['high'], data['low'], data['close']
)
print(avg_price.tail())

# Get available functions
functions = talib_wrapper.get_available_functions()
print(f"Available functions: {len(functions)}")

# Get function information
sma_info = talib_wrapper.get_function_info('SMA')
print(f"SMA signature: {sma_info['signature']}")
```

### Error Handling Examples

```python
import pandas as pd
from trade_agent.engine.nodes.talib_wrapper import talib_wrapper, TALibError, InsufficientDataError

# Handle insufficient data
try:
    short_series = pd.Series([1, 2], name='short')
    talib_wrapper.SMA(short_series, timeperiod=5)
except InsufficientDataError as e:
    print(f"Insufficient data error: {e}")

# Handle missing required columns
try:
    bad_data = pd.DataFrame({'price': [1, 2, 3, 4, 5]})
    talib_wrapper.MACD(bad_data)  # MACD needs close price data
except TALibError as e:
    print(f"TA-Lib error: {e}")

# Handle invalid input types
try:
    talib_wrapper.SMA([1, 2, 3, 4, 5], timeperiod=3)
except TALibError as e:
    print(f"TA-Lib error: {e}")
```

## Integration with Trading System

The TA-Lib wrapper will integrate with the existing trading system components:

1. **Data Handler**: Process market data from ParquetStore
2. **Feature Engineering**: Generate technical indicators for strategy development
3. **Signal Generation**: Create trading signals based on indicator crossovers and thresholds
4. **Backtesting**: Evaluate strategy performance using technical indicators

## Related Linear Issues

- [TA-98: Implement Dynamic TA-Lib Wrapper for Feature Engineering](https://linear.app/trading-rl-agent/issue/TA-98/implement-dynamic-ta-lib-wrapper-for-feature-engineering)
- [TA-99: Update Documentation for TA-Lib Wrapper and Feature Engineering](https://linear.app/trading-rl-agent/issue/TA-9/update-documentation-for-ta-lib-wrapper-and-feature-engineering)
- [TA-100: Add Unit Tests for TA-Lib Wrapper](https://linear.app/trading-rl-agent/issue/TA-100/add-unit-tests-for-ta-lib-wrapper)

## Dependencies

- `ta-lib>=0.4.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`

## See Also

- [ParquetStore Documentation](qarquetstore.md)
- [Feature Engineering Nodes](feature_engineering.md)
- [Technical Indicators](technical_indicators.md)
