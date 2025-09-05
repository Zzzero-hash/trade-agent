#!/usr/bin/env python3
"""
Test script for yfinance data collector integration.
"""
import os
import sys


# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trade_agent.plugins.builtins import (
    save_yfinance_data_to_store,
    yfinance_data_collector,
)


def test_yfinance_collector():
    """Test the yfinance data collector function."""

    # Test with a simple symbol
    result = yfinance_data_collector(
        symbol="AAPL",
        period="5d",
        interval="1d"
    )


    if result.get("success"):

        # Show first few data points
        if result.get("data"):
            for _i, _point in enumerate(result["data"][:3]):
                pass
    else:
        pass

    return result.get("success", False)


def test_save_to_store():
    """Test saving yfinance data to ParquetStore."""

    result = save_yfinance_data_to_store(
        symbol="TSLA",
        period="1mo",
        interval="1d",
        store_path="data"
    )


    if result.get("success"):
        pass
    else:
        pass

    return result.get("success", False)


if __name__ == "__main__":

    # Test collector
    collector_success = test_yfinance_collector()

    # Test save to store
    save_success = test_save_to_store()

    collector_status = "✅ PASS" if collector_success else "❌ FAIL"
    save_status = "✅ PASS" if save_success else "❌ FAIL"

    if collector_success and save_success:
        sys.exit(0)
    else:
        sys.exit(1)
