#!/usr/bin/env python3
"""
Test script for the data API endpoints
"""
import asyncio


async def test_data_api() -> None:
    """Test the data API endpoints"""

    # Test 1: List series (should be empty initially)
    try:
        from interface.api.routers.data import list_series
        result = await list_series()
    except Exception:
        pass

    # Test 2: Write sample data
    try:
        from interface.api.routers.data import DataWriteRequest, write_data

        # Create sample data
        sample_data = [
            {
                "timestamp": "2025-08-27T00:00:00Z",
                "open": 50000.0,
                "high": 51000.0,
                "low": 49500.0,
                "close": 50500.0,
                "volume": 1000.0
            },
            {
                "timestamp": "2025-08-27T01:00:00Z",
                "open": 50500.0,
                "high": 51500.0,
                "low": 50000.0,
                "close": 51000.0,
                "volume": 1200.0
            }
        ]

        request = DataWriteRequest(
            symbol="BTC-USD",
            timeframe="1h",
            data=sample_data
        )

        result = await write_data(request)
    except Exception:
        pass

    # Test 3: List series again (should show our data now)
    try:
        from interface.api.routers.data import list_series
        result = await list_series()
        for _series in result:
            pass
    except Exception:
        pass

    # Test 4: Read data back
    try:
        from interface.api.routers.data import DataReadRequest, read_data

        request = DataReadRequest(
            symbol="BTC-USD",
            timeframe="1h"
        )

        result = await read_data(request)
        if result:
            pass
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(test_data_api())
