"""
Unit tests for the TA-Lib wrapper.
"""
import numpy as np
import pandas as pd
import pytest

from trade_agent.engine.nodes.talib_wrapper import (
    InsufficientDataError,
    TALibError,
    TALibWrapper,
    talib_wrapper,
)


class TestTALibWrapper:

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample price data for testing."""
        # Create more data points to accommodate longer periods like RSI(14)
        open_prices = list(range(100, 120))  # 20 data points
        high_prices = [o + 5 for o in open_prices]
        low_prices = [o - 5 for o in open_prices]
        close_prices = [o + 2 for o in open_prices]
        volume_data = [1000 + (i * 100) for i in range(20)]

        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume_data
        })

    @pytest.fixture
    def sample_series(self) -> pd.Series:
        """Create sample series data for testing."""
        return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='price')

    def test_initialization(self) -> None:
        """Test that wrapper initializes correctly."""
        wrapper = TALibWrapper()
        assert len(wrapper.get_available_functions()) > 0
        # Check that common functions are available
        assert 'SMA' in wrapper.get_available_functions()
        assert 'RSI' in wrapper.get_available_functions()

    def test_get_function_info(self) -> None:
        """Test function info retrieval."""
        wrapper = TALibWrapper()
        info = wrapper.get_function_info('SMA')
        assert info['name'] == 'SMA'
        assert 'signature' in info
        assert 'parameters' in info
        assert len(info['parameters']) > 0

    def test_get_function_info_not_found(self) -> None:
        """Test error handling for non-existent functions."""
        wrapper = TALibWrapper()
        with pytest.raises(TALibError):
            wrapper.get_function_info('NONEXISTENT_FUNCTION')

    def test_sma_with_series(self, sample_series) -> None:
        """Test SMA calculation with Series input."""
        result = talib_wrapper.SMA(sample_series, timeperiod=3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_series)
        # First 2 values should be NaN due to minimum period
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])

    def test_sma_with_dataframe(self, sample_data) -> None:
        """Test SMA calculation with DataFrame input."""
        result = talib_wrapper.SMA(sample_data['close'], timeperiod=3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == 'indicator'

    def test_rsi_calculation(self, sample_data) -> None:
        """Test RSI calculation."""
        result = talib_wrapper.RSI(sample_data['close'], timeperiod=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        # Check that values are in reasonable range
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(0 <= val <= 100 for val in valid_values)

    def test_macd_calculation(self, sample_data) -> None:
        """Test MACD calculation (multiple outputs)."""
        result = talib_wrapper.MACD(sample_data['close'])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        # MACD should have 3 columns
        assert len(result.columns) == 3

    def test_bbands_calculation(self, sample_data) -> None:
        """Test Bollinger Bands calculation."""
        result = talib_wrapper.BBANDS(sample_data['close'])
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 3  # Upper, Middle, Lower bands

    def test_ohlc_functions(self, sample_data) -> None:
        """Test OHLC-based functions."""
        result = talib_wrapper.AVGPRICE(
            sample_data['open'], sample_data['high'],
            sample_data['low'], sample_data['close']
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_invalid_input_type(self) -> None:
        """Test error handling for invalid input types."""
        with pytest.raises(TALibError):
            talib_wrapper.SMA([1, 2, 3, 4, 5], timeperiod=3)

    def test_insufficient_data(self) -> None:
        """Test error handling for insufficient data."""
        short_series = pd.Series([1, 2], name='short')
        with pytest.raises(InsufficientDataError):
            talib_wrapper.SMA(short_series, timeperiod=5)

    def test_missing_required_columns(self) -> None:
        """Test error handling for missing required columns."""
        data = pd.DataFrame({'price': [1, 2, 3, 4, 5]})
        # MACD should work with single column data now (maps to 'close')
        result = talib_wrapper.MACD(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)

    def test_call_with_params(self, sample_data) -> None:
        """Test calling functions with parameter dictionary."""
        result = talib_wrapper.call_with_params(
            'SMA', sample_data['close'], {'timeperiod': 3})
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_dynamic_function_access(self, sample_series) -> None:
        """Test dynamic function access via __getattr__."""
        # This tests that functions can be accessed as attributes
        result = talib_wrapper.SMA(sample_series, timeperiod=3)
        assert isinstance(result, pd.Series)

    def test_timezone_handling(self) -> None:
        """Test handling of timezone-aware data."""
        tz_data = pd.Series(
            [1, 2, 3, 4, 5],
            index=pd.date_range('2023-01-01', periods=5, tz='UTC'),
            name='tz_series'
        )
        result = talib_wrapper.SMA(tz_data, timeperiod=3)
        assert isinstance(result, pd.Series)
        # Result should have timezone-naive index
        assert result.index.tz is None

    def test_nan_handling(self) -> None:
        """Test handling of NaN values in input data."""
        data_with_nan = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
                                  name='nan_series')
        result = talib_wrapper.SMA(data_with_nan, timeperiod=3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data_with_nan)

    def test_edge_case_single_value(self) -> None:
        """Test handling of single value data."""
        single_value = pd.Series([5], name='single')
        result = talib_wrapper.SMA(single_value, timeperiod=1)
        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.iloc[0] == 5


def test_global_instance() -> None:
    """Test that global instance works correctly."""
    assert isinstance(talib_wrapper, TALibWrapper)
    assert len(talib_wrapper.get_available_functions()) > 0
