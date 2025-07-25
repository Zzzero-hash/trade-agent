"""
Professional Data Feeds Integration

Industry-grade data providers for production trading systems.
Supports Alpaca, Yahoo Finance, and extensible to Bloomberg/Refinitiv.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from trade_agent.data.market_calendar import classify_portfolio_assets, get_trading_calendar

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Alpaca Trade API not available. Install with: pip install alpaca-trade-api",
    )

try:
    from finrl.finrl_meta.data_processors.processor_alpaca import AlpacaProcessor

    FINRL_ALPACA_AVAILABLE = True
except ImportError:
    FINRL_ALPACA_AVAILABLE = False
    logging.getLogger(__name__).warning("FinRL Alpaca processor not available")

import yfinance as yf

from trade_agent.data.features import generate_features

# Check for optional dependency: alpha_vantage
try:
    from alpha_vantage.timeseries import TimeSeries

    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Alpha Vantage not available. Install with: pip install alpha-vantage",
    )

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "CCXT not available. Install with: pip install ccxt",
    )

logger = logging.getLogger(__name__)


class ProfessionalDataProvider:
    """
    Industry-grade data provider with multiple professional feeds.

    Supports:
    - Alpaca Markets (commission-free trading data)
    - Yahoo Finance (fallback for development)
    - Extensible to Bloomberg/Refinitiv, Interactive Brokers
    """

    def __init__(self, provider: str = "alpaca", **kwargs: Any) -> None:
        """
        Initialize professional data provider.

        Args:
            provider: Data provider ('alpaca', 'yahoo', 'bloomberg')
            **kwargs: Provider-specific configuration
        """
        self.provider = provider
        self.config = kwargs
        self.cache_dir = Path(kwargs.get("cache_dir", "data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = kwargs.get("cache_ttl", 86400)  # 24 hours in seconds

        if provider == "alpaca":
            self._init_alpaca()
        elif provider == "yahoo":
            self._init_yahoo()
        elif provider == "bloomberg":
            self._init_bloomberg()
        elif provider == "alpha_vantage":
            self._init_alpha_vantage()
        elif provider == "ccxt":
            self._init_ccxt()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _init_alpaca(self) -> None:
        """Initialize Alpaca Markets data feed."""
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca Trade API required: pip install alpaca-trade-api")

        # Try to get configuration from the unified config system first
        try:
            from trade_agent.core.unified_config import UnifiedConfig

            config = UnifiedConfig()
            api_key = config.alpaca_api_key
            secret_key = config.alpaca_secret_key
            base_url = config.alpaca_base_url or "https://paper-api.alpaca.markets"
        except Exception:
            # Fallback to direct environment variable access
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY",
            )
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.market_client = StockHistoricalDataClient(api_key, secret_key)

        # Initialize FinRL processor if available
        if FINRL_ALPACA_AVAILABLE:
            self.finrl_processor = AlpacaProcessor(
                API_KEY=api_key,
                API_SECRET=secret_key,
                API_BASE_URL=base_url,
            )

        logger.info("Alpaca Markets data provider initialized")

    def _init_yahoo(self) -> None:
        """Initialize Yahoo Finance data feed (development/fallback)."""
        logger.info("Yahoo Finance data provider initialized (development mode)")

    def _init_bloomberg(self) -> None:
        """Initialize Bloomberg data feed (enterprise)."""
        # Placeholder for Bloomberg API integration
        # Requires Bloomberg Terminal and blpapi library
        logger.info("Bloomberg data provider initialized (placeholder)")
        raise NotImplementedError("Bloomberg integration requires enterprise license")

    def _init_alpha_vantage(self) -> None:
        """Initialize Alpha Vantage data feed."""
        if not ALPHA_VANTAGE_AVAILABLE:
            raise ImportError("Alpha Vantage required: pip install alpha-vantage")

        # Try to get configuration from the unified config system first
        try:
            from trade_agent.core.unified_config import UnifiedConfig

            config = UnifiedConfig()
            api_key = config.alphavantage_api_key
        except Exception:
            # Fallback to direct environment variable access
            api_key = os.getenv("ALPHA_VANTAGE_KEY")

        if not api_key:
            raise ValueError("Alpha Vantage API key required. Set ALPHA_VANTAGE_KEY environment variable.")

        self.ts = TimeSeries(key=api_key, output_format="pandas")
        logger.info("Alpha Vantage data provider initialized")

    def _init_ccxt(self) -> None:
        """Initialize CCXT exchange data feed."""
        if not CCXT_AVAILABLE:
            raise ImportError("CCXT required: pip install ccxt")

        exchange_name = self.config.get("exchange", "binance").lower()
        api_key = os.getenv(f"{exchange_name.upper()}_API_KEY")
        secret = os.getenv(f"{exchange_name.upper()}_SECRET")

        if not api_key or not secret:
            raise ValueError(
                f"{exchange_name.upper()} API credentials required. Set {exchange_name.upper()}_API_KEY and _SECRET",
            )

        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": secret,
            },
        )
        logger.info(f"CCXT {exchange_name} data provider initialized")

    def get_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1Day",
        include_features: bool = True,
        align_mixed_portfolio: bool = True,
        alignment_strategy: str = "last_known_value",
    ) -> pd.DataFrame:
        """
        Get professional market data with optional feature engineering.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data frequency ('1Min', '5Min', '1Hour', '1Day')
            include_features: Whether to generate technical features
            align_mixed_portfolio: Whether to align timestamps for mixed crypto/traditional portfolios
            alignment_strategy: Strategy for crypto alignment ('last_known_value', 'forward_fill', 'interpolate')

        Returns:
            DataFrame with OHLCV data and optional technical features
        """
        # Generate cache key using hash to avoid filename length issues
        import hashlib
        symbols_str = "_".join(sorted(symbols))
        symbols_hash = hashlib.sha256(symbols_str.encode(), usedforsecurity=False).hexdigest()[:12]  # Use first 12 chars of hash
        cache_key = f"{symbols_hash}_{start_date}_{end_date}_{timeframe}_{self.provider}.parquet"
        cache_path = self.cache_dir / cache_key

        # Check cache
        if cache_path.exists():
            mod_time = cache_path.stat().st_mtime
            if time.time() - mod_time < self.cache_ttl:
                logger.info(f"Loading from cache: {cache_path}")
                data = pd.read_parquet(cache_path)

                # Ensure 'symbol' and 'timestamp' columns exist. If not, cache is invalid.
                if "symbol" in data.columns and "timestamp" in data.columns:
                    if include_features and not data.empty:
                        logger.info("Generating technical features...")
                        data = self._add_technical_features(data)
                    return data

                logger.warning(
                    f"Cached data at {cache_path} is missing required columns. Refetching."
                )

        if self.provider == "alpaca":
            data = self._get_alpaca_data(symbols, start_date, end_date, timeframe)
        elif self.provider == "yahoo":
            data = self._get_yahoo_data(symbols, start_date, end_date)
        elif self.provider == "alpha_vantage":
            data = self._get_alpha_vantage_data(symbols, start_date, end_date, timeframe)
        elif self.provider == "ccxt":
            data = self._get_ccxt_data(symbols, start_date, end_date, timeframe)
        else:
            raise NotImplementedError(
                f"Data retrieval not implemented for {self.provider}",
            )

        # Handle mixed portfolio timestamp alignment
        if align_mixed_portfolio and not data.empty:
            # Auto-detect if crypto is present and apply appropriate strategy
            data = self._align_mixed_portfolio_timestamps(data, symbols, alignment_strategy)

        if include_features and not data.empty:
            logger.info("Generating technical features...")
            data = self._add_technical_features(data)

        # Cache the result
        if not data.empty:
            data.to_parquet(cache_path)
            logger.info(f"Cached data to: {cache_path}")

        return data

    def _get_alpaca_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """Get data from Alpaca Markets."""
        try:
            # Use FinRL processor if available (recommended)
            if FINRL_ALPACA_AVAILABLE and hasattr(self, "finrl_processor"):
                return self.finrl_processor.download_data(
                    ticker_list=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    time_interval=timeframe,
                )

            # Fallback to direct Alpaca API
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            timeframe_mapping = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame.Minute,
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
            }
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe_mapping.get(timeframe, TimeFrame.Day),
                start=pd.to_datetime(start_date).tz_localize("America/New_York"),
                end=pd.to_datetime(end_date).tz_localize("America/New_York"),
            )
            bars = self.market_client.get_stock_bars(request_params).df
            if not bars.empty:
                bars.reset_index(inplace=True)
                bars.rename(
                    columns={"timestamp": "date", "volume": "volume"},
                    inplace=True,
                )
                from .utils import normalize_timestamps

                bars = normalize_timestamps(bars, timestamp_column="date")
                return bars
            logger.error("No data retrieved from Alpaca")
            return pd.DataFrame()

        except APIError as e:
            logger.exception(f"Alpaca API error during data retrieval: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.exception(f"Alpaca data retrieval error: {e}")
            return pd.DataFrame()

    def _normalize_timestamp_column(self, timestamp_col: pd.Series, timezone: str = "America/New_York") -> pd.Series:
        """
        Normalize timestamp column to consistent timezone format.

        Args:
            timestamp_col: Series of timestamps
            timezone: Target timezone for normalization

        Returns:
            Normalized timestamp series
        """
        from .utils import _normalize_timestamp_series
        return _normalize_timestamp_series(timestamp_col, timezone)

    def _get_yahoo_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Get data from Yahoo Finance (fallback/development)."""
        try:
            all_data = []

            # Add timeout and rate limiting for API calls
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from concurrent.futures import TimeoutError as FutureTimeoutError

            def fetch_symbol_data(symbol: str) -> tuple[str, pd.DataFrame]:
                """Fetch data for a single symbol with timeout."""
                try:
                    ticker = yf.Ticker(symbol)
                    # Add timeout to prevent hanging
                    data = ticker.history(start=start_date, end=end_date, timeout=30)
                    return symbol, data
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    return symbol, pd.DataFrame()

            # Process symbols in parallel with timeout
            max_workers = min(10, len(symbols))  # Limit concurrent requests
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(fetch_symbol_data, symbol): symbol
                    for symbol in symbols
                }

                # Collect results with timeout
                for future in as_completed(future_to_symbol, timeout=300):  # 5 minute total timeout
                    try:
                        symbol, data = future.result(timeout=60)  # 1 minute per symbol
                        if not data.empty:
                            data.reset_index(inplace=True)
                            data["symbol"] = symbol
                            data.rename(
                                columns={
                                    "Date": "timestamp",
                                    "Open": "open",
                                    "High": "high",
                                    "Low": "low",
                                    "Close": "close",
                                    "Volume": "volume",
                                },
                                inplace=True,
                            )
                            # Select relevant columns
                            data = data[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
                            all_data.append(data)
                    except (FutureTimeoutError, TimeoutError):
                        symbol = future_to_symbol[future]
                        logger.warning(f"Timeout fetching data for {symbol}")
                        continue
                    except Exception as e:
                        symbol = future_to_symbol[future]
                        logger.warning(f"Error fetching data for {symbol}: {e}")
                        continue

            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                # Handle timezone-aware datetime conversion consistently using utility function
                from .utils import normalize_timestamps
                combined_data = normalize_timestamps(combined_data, timestamp_column="timestamp")
                return combined_data.sort_values(["symbol", "timestamp"]).reset_index(
                    drop=True,
                )
            return pd.DataFrame()

        except Exception as e:
            logger.exception(f"Yahoo Finance data retrieval error: {e}")
            return pd.DataFrame()

    def _get_alpha_vantage_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Get data from Alpha Vantage."""
        try:
            all_data = []
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            for symbol in symbols:
                if timeframe == "1Day":
                    df, _ = self.ts.get_daily_adjusted(symbol, outputsize="full")
                else:
                    # For intraday, map timeframe to Alpha Vantage interval
                    interval_map = {
                        "1Min": "1min",
                        "5Min": "5min",
                        "15Min": "15min",
                        "30Min": "30min",
                        "1Hour": "60min",
                    }
                    interval = interval_map.get(timeframe, "1min")
                    df, _ = self.ts.get_intraday(symbol, interval=interval, outputsize="full")

                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                # Rename columns
                df = df.rename(
                    columns={
                        "1. open": "open",
                        "2. high": "high",
                        "3. low": "low",
                        "4. close": "close",
                        "5. adjusted close": "adjusted_close",
                        "6. volume": "volume",
                        "7. dividend amount": "dividend",
                        "8. split coefficient": "split",
                    },
                )

                df["symbol"] = symbol
                df.index.name = "date"
                df = df.reset_index()
                # Handle timezone-aware datetime conversion
                try:
                    df["date"] = pd.to_datetime(df["date"], utc=True)
                except ValueError:
                    # If utc=True fails, try without timezone info
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

                # Filter by date range
                df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

                all_data.append(df)

            if all_data:
                return pd.concat(all_data, ignore_index=True)
            return pd.DataFrame()

        except Exception as e:
            logger.exception(f"Alpha Vantage data retrieval error: {e}")
            return pd.DataFrame()

    def _get_ccxt_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Get data from CCXT exchange."""
        try:
            all_data = []
            start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
            int(pd.to_datetime(end_date).timestamp() * 1000)

            tf_map = {
                "1Min": "1m",
                "5Min": "5m",
                "15Min": "15m",
                "30Min": "30m",
                "1Hour": "1h",
                "1Day": "1d",
            }
            ccxt_tf = tf_map.get(timeframe, "1d")

            for symbol in symbols:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=ccxt_tf, since=start_ts, limit=None)
                if not ohlcv:
                    logger.warning(f"No data for {symbol}")
                    continue

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df[(df["timestamp"] <= pd.to_datetime(end_date))]
                df["symbol"] = symbol
                all_data.append(df)

            if all_data:
                return pd.concat(all_data, ignore_index=True)
            return pd.DataFrame()

        except Exception as e:
            logger.exception(f"CCXT data retrieval error: {e}")
            return pd.DataFrame()

    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features to market data."""
        try:
            enhanced_data = []

            for symbol in data["symbol"].unique():
                symbol_data = data[data["symbol"] == symbol].copy()

                # Generate unified features (exactly 78 features)
                symbol_data = generate_features(symbol_data)
                enhanced_data.append(symbol_data)

            return pd.concat(enhanced_data, ignore_index=True)

        except Exception as e:
            logger.exception(f"Feature generation error: {e}")
            return data

    def get_real_time_data(self, symbols: list[str]) -> dict[str, Any]:
        """
        Get real-time market data for live trading.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dictionary with real-time market data
        """
        if self.provider == "alpaca":
            return self._get_alpaca_real_time(symbols)
        if self.provider == "alpha_vantage":
            return self._get_alpha_vantage_real_time(symbols)
        if self.provider == "ccxt":
            return self._get_ccxt_real_time(symbols)
        raise NotImplementedError(
            f"Real-time data not available for {self.provider}",
        )

    def _get_alpaca_real_time(self, symbols: list[str]) -> dict[str, Any]:
        """Get real-time data from Alpaca."""
        try:
            quotes = self.market_client.get_latest_stock_quotes(symbols)
            return {
                symbol: {
                    "bid_price": float(quote.bid_price),
                    "ask_price": float(quote.ask_price),
                    "bid_size": int(quote.bid_size),
                    "ask_size": int(quote.ask_size),
                    "timestamp": quote.timestamp,
                }
                for symbol, quote in quotes.items()
            }
        except APIError as e:
            logger.exception(f"Alpaca API error during real-time data retrieval: {e}")
            return {}
        except Exception as e:
            logger.exception(f"Real-time data error: {e}")
            return {}

    def _get_alpha_vantage_real_time(self, symbols: list[str]) -> dict[str, Any]:
        """Get real-time quotes from Alpha Vantage."""
        try:
            quotes = {}
            for symbol in symbols:
                data, _ = self.ts.get_quote(symbol)
                quotes[symbol] = {
                    "price": float(data["05. price"]),
                    "previous_close": float(data["08. previous close"]),
                    "volume": int(data["06. volume"]),
                    "timestamp": data["07. latest trading day"],
                }
            return quotes
        except Exception as e:
            logger.exception(f"Alpha Vantage real-time error: {e}")
            return {}

    def _get_ccxt_real_time(self, symbols: list[str]) -> dict[str, Any]:
        """Get real-time ticker from CCXT."""
        try:
            quotes = {}
            for symbol in symbols:
                ticker = self.exchange.fetch_ticker(symbol)
                quotes[symbol] = {
                    "bid": ticker["bid"],
                    "ask": ticker["ask"],
                    "last": ticker["last"],
                    "volume": ticker["baseVolume"],
                    "timestamp": pd.to_datetime(ticker["timestamp"], unit="ms"),
                }
            return quotes
        except Exception as e:
            logger.exception(f"CCXT real-time error: {e}")
            return {}

    def _align_mixed_portfolio_timestamps(self, data: pd.DataFrame, symbols: list[str], alignment_strategy: str = "last_known_value") -> pd.DataFrame:
        """
        Align timestamps for mixed crypto/traditional portfolios.

        This method automatically detects crypto presence and applies appropriate
        alignment strategy. Uses last_known_value as default for optimal data integrity.

        Args:
            data: DataFrame with market data
            symbols: List of symbols in the portfolio
            alignment_strategy: Strategy for crypto alignment ('last_known_value', 'forward_fill', 'interpolate')

        Returns:
            DataFrame with aligned timestamps
        """
        try:
            # Get trading calendar
            calendar = get_trading_calendar()

            # Classify portfolio assets
            classification = classify_portfolio_assets(symbols)

            if not classification["mixed"]:
                logger.info("No mixed portfolio detected, skipping timestamp alignment")
                return data

            logger.info(f"Mixed portfolio detected: {len(classification['crypto'])} crypto, {len(classification['traditional'])} traditional")
            logger.info(f"Auto-applying alignment strategy: {alignment_strategy}")

            # Ensure we have timestamp and symbol columns
            if "timestamp" not in data.columns or "symbol" not in data.columns:
                logger.warning("Missing timestamp or symbol columns, skipping alignment")
                return data

            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                data["timestamp"] = pd.to_datetime(data["timestamp"])

            # Use the calendar's alignment method with specified strategy
            aligned_data = calendar.align_data_timestamps(data, symbols, alignment_strategy)

            # Add metadata about the alignment
            if not aligned_data.empty and "alignment_method" in aligned_data.columns:
                methods_used = aligned_data["alignment_method"].value_counts()
                logger.info(f"Alignment methods applied: {dict(methods_used)}")

            logger.info(f"Timestamp alignment complete: {len(data)} -> {len(aligned_data)} rows")
            return aligned_data

        except Exception as e:
            logger.exception(f"Error aligning mixed portfolio timestamps: {e}")
            return data

    def validate_connection(self) -> bool:
        """Validate connection to data provider."""
        try:
            if self.provider == "alpaca":
                account = self.trading_client.get_account()
                logger.info(f"Alpaca connection validated. Account: {account.id}")
                return True
            if self.provider == "yahoo":
                # Test with a simple query
                test_data = yf.download("AAPL", period="1d")
                return not test_data.empty
            if self.provider == "alpha_vantage":
                # Test with a simple query
                data, _ = self.ts.get_quote("AAPL")
                return not data.empty
            if self.provider == "ccxt":
                # Test with a simple query (e.g., fetch_ticker for a symbol)
                try:
                    ticker = self.exchange.fetch_ticker("BTC/USDT")
                    return ticker is not None
                except Exception as e:
                    logger.warning(f"CCXT connection test failed: {e}")
                    return False
            return False

        except Exception as e:
            logger.exception(f"Connection validation failed: {e}")
            return False


# CLI functionality moved to unified CLI in trading_rl_agent.cli
