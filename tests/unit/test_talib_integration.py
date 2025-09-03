"""
Integration tests for the TA-Lib wrapper with the existing trading pipeline.
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


class TestTALibIntegration:
    """Test TA-Lib wrapper integration with existing trading pipeline."""

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Create realistic market data for integration testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducible results

        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)  # Daily returns
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLC data
        open_prices = prices[:-1]
        close_prices = prices[1:]
        high_prices = [max(o, c) * (1 + abs(np.random.normal(0, 0.01)))
                       for o, c in zip(open_prices, close_prices, strict=False)]
        low_prices = [min(o, c) * (1 - abs(np.random.normal(0, 0.01)))
                      for o, c in zip(open_prices, close_prices, strict=False)]
        volumes = [int(np.random.uniform(1000, 10000)) for _ in range(99)]

        return pd.DataFrame({
            'timestamp': dates[1:],
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }).set_index('timestamp')

    def test_sma_integration(self, sample_market_data) -> None:
        """Test SMA integration with realistic market data."""
        # Calculate 20-period SMA
        sma_20 = talib_wrapper.SMA(sample_market_data['close'], timeperiod=20)

        assert isinstance(sma_20, pd.Series)
        assert len(sma_20) == len(sample_market_data)
        # First 19 values should be NaN due to minimum period
        assert all(pd.isna(sma_20.iloc[:19]))
        # Remaining values should be numeric
        valid_values = sma_20.iloc[19:].dropna()
        assert len(valid_values) > 0
        assert all(isinstance(val, int | float) for val in valid_values)

    def test_rsi_integration(self, sample_market_data) -> None:
        """Test RSI integration with realistic market data."""
        # Calculate 14-period RSI
        rsi_14 = talib_wrapper.RSI(sample_market_data['close'], timeperiod=14)

        assert isinstance(rsi_14, pd.Series)
        assert len(rsi_14) == len(sample_market_data)
        # Check that values are in reasonable range (0-100)
        valid_values = rsi_14.dropna()
        if len(valid_values) > 0:
            assert all(0 <= val <= 100 for val in valid_values)

    def test_macd_integration(self, sample_market_data) -> None:
        """Test MACD integration with realistic market data."""
        # Calculate MACD
        macd_result = talib_wrapper.MACD(sample_market_data['close'])

        assert isinstance(macd_result, pd.DataFrame)
        assert len(macd_result) == len(sample_market_data)
        # MACD should have 3 columns
        assert len(macd_result.columns) == 3
        # All columns should have the same length
        for col in macd_result.columns:
            assert len(macd_result[col]) == len(sample_market_data)

    def test_bbands_integration(self, sample_market_data) -> None:
        """Test Bollinger Bands integration with realistic market data."""
        # Calculate Bollinger Bands
        bbands_result = talib_wrapper.BBANDS(sample_market_data['close'])

        assert isinstance(bbands_result, pd.DataFrame)
        assert len(bbands_result) == len(sample_market_data)
        # BBANDS should have 3 columns (upper, middle, lower)
        assert len(bbands_result.columns) == 3
        # Upper band should be >= middle band >= lower band
        valid_rows = bbands_result.dropna()
        if len(valid_rows) > 0:
            assert all(row.iloc[0] >= row.iloc[1] >= row.iloc[2]
                      for _, row in valid_rows.iterrows())

    def test_ohlc_integration(self, sample_market_data) -> None:
        """Test OHLC-based functions integration."""
        # Test AVGPRICE
        avg_price = talib_wrapper.AVGPRICE(
            sample_market_data['open'], sample_market_data['high'],
            sample_market_data['low'], sample_market_data['close']
        )

        assert isinstance(avg_price, pd.Series)
        assert len(avg_price) == len(sample_market_data)
        # AVGPRICE should be between high and low
        valid_rows = pd.concat([
            sample_market_data[['high', 'low']],
            avg_price.to_frame('avg')
        ], axis=1).dropna()

        if len(valid_rows) > 0:
            assert all(row['low'] <= row['avg'] <= row['high']
                      for _, row in valid_rows.iterrows())

    def test_multiple_indicators_pipeline(self, sample_market_data) -> None:
        """Test integration of multiple indicators in a pipeline."""
        # Create a feature engineering pipeline
        features = pd.DataFrame(index=sample_market_data.index)

        # Add various technical indicators
        features['sma_10'] = talib_wrapper.SMA(sample_market_data['close'], timeperiod=10)
        features['sma_20'] = talib_wrapper.SMA(sample_market_data['close'], timeperiod=20)
        features['rsi_14'] = talib_wrapper.RSI(sample_market_data['close'], timeperiod=14)
        features['macd_line'] = talib_wrapper.MACD(sample_market_data['close']).iloc[:, 0]

        # Calculate Bollinger Bands
        bbands = talib_wrapper.BBANDS(sample_market_data['close'])
        features['bb_upper'] = bbands.iloc[:, 0]
        features['bb_middle'] = bbands.iloc[:, 1]
        features['bb_lower'] = bbands.iloc[:, 2]

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_market_data)
        # Should have 7 feature columns (MACD returns 3 columns, so we get 7 total)
        assert len(features.columns) == 7
        # All columns should have proper indexing
        assert features.index.equals(sample_market_data.index)

    def test_timezone_aware_data_integration(self, sample_market_data) -> None:
        """Test integration with timezone-aware market data."""
        # Create timezone-aware data
        tz_data = sample_market_data.copy()
        tz_data.index = tz_data.index.tz_localize('UTC')

        # Test that indicators work with timezone-aware data
        sma_result = talib_wrapper.SMA(tz_data['close'], timeperiod=10)
        rsi_result = talib_wrapper.RSI(tz_data['close'], timeperiod=14)

        assert isinstance(sma_result, pd.Series)
        assert isinstance(rsi_result, pd.Series)
        assert len(sma_result) == len(tz_data)
        assert len(rsi_result) == len(tz_data)
        # Result indices should be timezone-naive
        assert sma_result.index.tz is None
        assert rsi_result.index.tz is None

    def test_nan_data_integration(self, sample_market_data) -> None:
        """Test integration with NaN-containing market data."""
        # Introduce some NaN values
        nan_data = sample_market_data.copy()
        nan_data.loc[nan_data.index[5]:nan_data.index[10], 'close'] = np.nan

        # Test that indicators handle NaN data gracefully
        sma_result = talib_wrapper.SMA(nan_data['close'], timeperiod=5)
        rsi_result = talib_wrapper.RSI(nan_data['close'], timeperiod=14)

        assert isinstance(sma_result, pd.Series)
        assert isinstance(rsi_result, pd.Series)
        assert len(sma_result) == len(nan_data)
        assert len(rsi_result) == len(nan_data)
        # Results should maintain proper indexing
        assert sma_result.index.equals(nan_data.index)
        assert rsi_result.index.equals(nan_data.index)

    def test_insufficient_data_integration(self) -> None:
        """Test integration error handling for insufficient data."""
        # Create very short data series
        short_data = pd.Series([100, 101], name='short_close')

        # Test that appropriate errors are raised
        with pytest.raises(InsufficientDataError):
            talib_wrapper.SMA(short_data, timeperiod=5)

        with pytest.raises(InsufficientDataError):
            talib_wrapper.RSI(short_data, timeperiod=14)

    def test_invalid_input_integration(self) -> None:
        """Test integration error handling for invalid inputs."""
        # Test with invalid data types
        with pytest.raises(TALibError):
            talib_wrapper.SMA([100, 101, 102], timeperiod=3)

        # Test with empty data
        empty_series = pd.Series([], name='empty', dtype=float)
        with pytest.raises(InsufficientDataError):
            talib_wrapper.SMA(empty_series, timeperiod=3)


def test_global_wrapper_integration() -> None:
    """Test that global wrapper instance integrates properly."""
    # Test that global instance is accessible and functional
    assert isinstance(talib_wrapper, TALibWrapper)

    # Test basic functionality
    sample_data = pd.Series([100, 101, 102, 103, 104, 105], name='test_close')
    sma_result = talib_wrapper.SMA(sample_data, timeperiod=3)

    assert isinstance(sma_result, pd.Series)
    assert len(sma_result) == len(sample_data)
    # First 2 values should be NaN due to minimum period
    assert pd.isna(sma_result.iloc[0])
    assert pd.isna(sma_result.iloc[1])
    assert not pd.isna(sma_result.iloc[2])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
