"""
Built-in plugins for trade-agent system.
"""
from typing import Any

import pandas as pd
import yfinance as yf

from trade_agent.engine.nodes.data_handler import ParquetStore


def simple_memory_collector() -> dict[str, Any]:
    """
    Simple in-memory data collector for testing purposes.
    """
    return {
        "name": "memory",
        "description": "Simple in-memory data collector"
    }


def make_hybrid_agent() -> None:
    """
    Factory function for hybrid RL agent.
    """
    pass


class InMemoryBroker:
    """
    Simple in-memory broker for testing.
    """
    pass


def yfinance_data_collector(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None
) -> dict[str, Any]:
    """
    Collect data from Yahoo Finance using yfinance library.

    Args:
        symbol: Stock/crypto symbol (e.g., 'AAPL', 'BTC-USD')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y',
                '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m',
                  '1h', '1d', '5d', '1wk', '1mo', '3mo')
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)

    Returns:
        Dict containing the collected data and metadata
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)

        # Download data
        if start_date and end_date:
            # Use date range
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
        else:
            # Use period
            data = ticker.history(
                period=period,
                interval=interval
            )

        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")

        # Convert to the format expected by ParquetStore
        # Reset index to make timestamp a column
        data = data.reset_index()

        # Rename columns to match expected format
        data = data.rename(columns={
            'Datetime': 'timestamp',
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Ensure all required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert timestamp to datetime with UTC timezone
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)

        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)

        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "data": data.to_dict('records'),
            "rows": len(data),
            "success": True
        }

    except Exception as e:
        return {
            "symbol": symbol,
            "error": str(e),
            "success": False
        }


def save_yfinance_data_to_store(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    store_path: str = "data"
) -> dict[str, Any]:
    """
    Collect data from Yahoo Finance and save it to ParquetStore.

    Args:
        symbol: Stock/crypto symbol (e.g., 'AAPL', 'BTC-USD')
        period: Time period for data collection
        interval: Data interval
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        store_path: Path to the data store

    Returns:
        Dict containing operation result and metadata
    """
    try:
        # Collect data
        result = yfinance_data_collector(symbol, period, interval, start_date, end_date)

        if not result.get("success", False):
            return result

        # Convert data to DataFrame
        data = pd.DataFrame(result["data"])

        # Save to ParquetStore
        store = ParquetStore(store_path)
        meta = store.write(data, symbol, interval)

        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "rows": meta.rows,
            "start": meta.start.isoformat(),
            "end": meta.end.isoformat(),
            "last_updated": meta.last_updated.isoformat(),
            "success": True
        }

    except Exception as e:
        return {
            "symbol": symbol,
            "error": str(e),
            "success": False
        }
