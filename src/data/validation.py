"""
Comprehensive Symbol Validation System

This module provides a robust framework for validating financial symbols
across multiple asset classes and data sources, ensuring data quality
and reliability for the trading system.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import requests
import yfinance as yf

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Symbol validation status enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    DELISTED = "delisted"
    SYMBOL_CHANGED = "symbol_changed"
    DATA_UNAVAILABLE = "data_unavailable"
    API_ERROR = "api_error"
    UNKNOWN = "unknown"


class AssetClass(Enum):
    """Asset class enumeration."""
    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    FUTURE = "future"
    REIT = "reit"
    VOLATILITY = "volatility"
    FOREX = "forex"


@dataclass
class ValidationResult:
    """Result of symbol validation."""
    symbol: str
    status: ValidationStatus
    asset_class: AssetClass
    current_name: Optional[str] = None
    suggested_symbol: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    last_traded_date: Optional[datetime] = None
    data_quality_score: float = 0.0
    validation_timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Configuration for symbol validation."""
    # API configuration
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    max_workers: int = 10

    # Data quality thresholds
    min_trading_days: int = 14  # More realistic (2-3 weeks of trading days)
    min_volume_threshold: float = 100   # Lower threshold for broader coverage
    max_days_since_last_trade: int = 7  # Should have traded within a week

    # Asset class specific settings
    crypto_suffix_required: bool = True
    futures_suffix_required: bool = True

    # Validation rules
    validate_company_info: bool = True
    validate_trading_status: bool = True
    validate_data_availability: bool = True
    validate_metadata_accuracy: bool = True

    # Caching
    cache_results: bool = True
    cache_duration_hours: int = 24
    cache_file: str = "symbol_validation_cache.json"


class SymbolValidator:
    """Comprehensive symbol validation system."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the validator with configuration."""
        self.config = config or ValidationConfig()
        self.cache: dict[str, ValidationResult] = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36'
            )
        })

        # Load cache if it exists
        self._load_cache()

        logger.info("SymbolValidator initialized with config: %s", self.config)

    def _load_cache(self) -> None:
        """Load validation results from cache."""
        try:
            if os.path.exists(self.config.cache_file):
                with open(self.config.cache_file) as f:
                    cache_data = json.load(f)

                for symbol, data in cache_data.items():
                    # Convert datetime strings back to datetime objects
                    if 'validation_timestamp' in data:
                        data['validation_timestamp'] = datetime.fromisoformat(
                            data['validation_timestamp']
                        )
                    if 'last_traded_date' in data and data['last_traded_date']:
                        data['last_traded_date'] = datetime.fromisoformat(
                            data['last_traded_date']
                        )

                    # Convert enums
                    data['status'] = ValidationStatus(data['status'])
                    data['asset_class'] = AssetClass(data['asset_class'])

                    self.cache[symbol] = ValidationResult(**data)

                logger.info(
                    f"Loaded {len(self.cache)} cached validation results"
                )
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}

    def _save_cache(self) -> None:
        """Save validation results to cache."""
        try:
            cache_data = {}
            for symbol, result in self.cache.items():
                data = {
                    'symbol': result.symbol,
                    'status': result.status.value,
                    'asset_class': result.asset_class.value,
                    'current_name': result.current_name,
                    'suggested_symbol': result.suggested_symbol,
                    'exchange': result.exchange,
                    'currency': result.currency,
                    'sector': result.sector,
                    'market_cap': result.market_cap,
                    'last_traded_date': (
                        result.last_traded_date.isoformat()
                        if result.last_traded_date else None
                    ),
                    'data_quality_score': result.data_quality_score,
                    'validation_timestamp': (
                        result.validation_timestamp.isoformat()
                    ),
                    'error_message': result.error_message,
                    'metadata': result.metadata
                }
                cache_data[symbol] = data

            with open(self.config.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.info(
                f"Saved {len(cache_data)} validation results to cache"
            )
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _is_cache_valid(self, result: ValidationResult) -> bool:
        """Check if cached result is still valid."""
        if not self.config.cache_results:
            return False

        cache_age = datetime.now() - result.validation_timestamp
        return cache_age < timedelta(hours=self.config.cache_duration_hours)

    def _get_asset_class_from_symbol(self, symbol: str) -> AssetClass:
        """Determine asset class from symbol format."""
        symbol_upper = symbol.upper()

        # Check for forex pairs (e.g., EURUSD=X, GBPUSD=X)
        if symbol_upper.endswith('=X') or symbol_upper in [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF'
        ]:
            return AssetClass.FOREX
        elif symbol_upper.endswith('-USD') or symbol_upper.endswith('-EUR'):
            return AssetClass.CRYPTO
        elif symbol_upper.endswith('=F'):
            return AssetClass.FUTURE
        elif symbol_upper in ['VXX', 'BATS:VIX', 'VIX']:
            return AssetClass.VOLATILITY
        elif symbol_upper in ['VNQ', 'SCHH', 'REET']:
            return AssetClass.REIT
        elif symbol_upper in [
            'SPY', 'QQQ', 'IVV', 'VOO', 'DIA', 'IWM',
            'VTI', 'VEA', 'VWO', 'QQQM'
        ]:
            return AssetClass.ETF
        else:
            return AssetClass.EQUITY

    def validate_symbol(
        self, symbol: str, asset_class: Optional[AssetClass] = None
    ) -> ValidationResult:
        """Validate a single symbol."""
        # Check cache first
        if symbol in self.cache and self._is_cache_valid(self.cache[symbol]):
            logger.debug(f"Using cached result for {symbol}")
            return self.cache[symbol]

        # Determine asset class if not provided
        if asset_class is None:
            asset_class = self._get_asset_class_from_symbol(symbol)

        logger.info(
            f"Validating symbol: {symbol} (Asset class: {asset_class.value})"
        )

        # Initialize result
        result = ValidationResult(
            symbol=symbol,
            status=ValidationStatus.UNKNOWN,
            asset_class=asset_class
        )

        try:
            # Validate using Yahoo Finance
            result = self._validate_with_yfinance(result)

            # Asset class specific validation
            if asset_class == AssetClass.EQUITY:
                result = self._validate_equity_specific(result)
            elif asset_class == AssetClass.CRYPTO:
                result = self._validate_crypto_specific(result)
            elif asset_class == AssetClass.ETF:
                result = self._validate_etf_specific(result)
            elif asset_class == AssetClass.FUTURE:
                result = self._validate_future_specific(result)
            elif asset_class == AssetClass.FOREX:
                result = self._validate_forex_specific(result)

            # Calculate data quality score
            result.data_quality_score = self._calculate_quality_score(result)

            # Cache the result
            self.cache[symbol] = result

        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            result.status = ValidationStatus.API_ERROR
            result.error_message = str(e)

        return result

    def _validate_with_yfinance(
        self, result: ValidationResult
    ) -> ValidationResult:
        """Validate symbol using Yahoo Finance API."""
        try:
            ticker = yf.Ticker(result.symbol)

            # Get basic info
            info = ticker.info
            if not info or 'symbol' not in info:
                result.status = ValidationStatus.DATA_UNAVAILABLE
                result.error_message = "No data available from Yahoo Finance"
                return result

            # Update basic information
            result.current_name = (
                info.get('longName') or info.get('shortName')
            )
            result.exchange = info.get('exchange')
            result.currency = info.get('currency')
            result.sector = info.get('sector')
            result.market_cap = info.get('marketCap')

            # Get historical data to check trading status
            end_date = datetime.now()
            start_date = end_date - timedelta(
                days=self.config.min_trading_days * 2 + 5
            )

            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                result.status = ValidationStatus.DATA_UNAVAILABLE
                result.error_message = "No recent trading data available"
                return result

            # Check last trading date
            # Get last trade date and ensure UTC timezone
            last_trade_date = hist.index[-1].to_pydatetime()

            # Handle timezone information with safe attribute access
            hist_tz = getattr(hist.index, 'tz', None) or timezone.utc

            # Ensure trade date has timezone info
            if last_trade_date.tzinfo is None:
                last_trade_date = last_trade_date.replace(tzinfo=hist_tz)

            # Convert all timestamps to UTC for consistent comparison
            utc_trade_date = last_trade_date.astimezone(timezone.utc)
            result.last_traded_date = utc_trade_date

            # Get current time in UTC
            now_utc = datetime.now(timezone.utc)

            # Calculate days difference using UTC dates
            days_since_last_trade = (now_utc - utc_trade_date).days

            logger.debug(
                "Timezone Debug: Hist=%s, TradeDate=%s, Now=%s, DaysSince=%d",
                str(hist_tz),
                utc_trade_date.isoformat(),
                now_utc.isoformat(),
                days_since_last_trade
            )

            if days_since_last_trade > self.config.max_days_since_last_trade:
                result.status = ValidationStatus.DELISTED
                result.error_message = (
                    f"Last traded {days_since_last_trade} days ago"
                )
            else:
                # Check data quality
                trading_days = len(hist)
                avg_volume = (
                    hist['Volume'].mean() if 'Volume' in hist.columns else 0
                )

                if trading_days < self.config.min_trading_days:
                    result.status = ValidationStatus.INVALID
                    result.error_message = (
                        f"Insufficient trading history: {trading_days} days"
                    )
                elif (result.asset_class != AssetClass.FOREX and
                      avg_volume < self.config.min_volume_threshold):
                    # Skip volume validation for forex pairs as they don't
                    # have meaningful volume
                    result.status = ValidationStatus.INVALID
                    result.error_message = f"Low average volume: {avg_volume}"
                else:
                    result.status = ValidationStatus.VALID

            # Store additional metadata
            result.metadata = {
                'trading_days': len(hist),
                'avg_volume': (
                    float(hist['Volume'].mean())
                    if 'Volume' in hist.columns else 0
                ),
                'days_since_last_trade': days_since_last_trade,
                'price_range_52w': {
                    'high': float(info.get('fiftyTwoWeekHigh', 0) or 0),
                    'low': float(info.get('fiftyTwoWeekLow', 0) or 0)
                }
            }

        except Exception as e:
            result.status = ValidationStatus.API_ERROR
            result.error_message = f"Yahoo Finance API error: {str(e)}"

        return result

    def _validate_equity_specific(
        self, result: ValidationResult
    ) -> ValidationResult:
        """Perform equity-specific validation."""
        if result.status != ValidationStatus.VALID:
            return result

        # Check if it's actually an equity (not ETF, REIT, etc.)
        if result.metadata.get('quoteType') == 'ETF':
            result.asset_class = AssetClass.ETF

        # Validate sector information for equities
        if not result.sector and result.asset_class == AssetClass.EQUITY:
            logger.warning(
                f"No sector information for equity {result.symbol}"
            )

        return result

    def _validate_crypto_specific(
        self, result: ValidationResult
    ) -> ValidationResult:
        """Perform cryptocurrency-specific validation."""
        if (not result.symbol.endswith('-USD') and
                self.config.crypto_suffix_required):
            result.status = ValidationStatus.INVALID
            result.error_message = (
                "Crypto symbol should end with -USD for Yahoo Finance"
            )

        return result

    def _validate_etf_specific(
        self, result: ValidationResult
    ) -> ValidationResult:
        """Perform ETF-specific validation."""
        # ETFs should have certain characteristics
        if result.status == ValidationStatus.VALID:
            # Check if it's actually an ETF
            if result.metadata.get('quoteType') != 'ETF':
                logger.warning(
                    f"Symbol {result.symbol} classified as ETF "
                    f"but may not be"
                )

        return result

    def _validate_future_specific(
        self, result: ValidationResult
    ) -> ValidationResult:
        """Perform futures-specific validation."""
        if (not result.symbol.endswith('=F') and
                self.config.futures_suffix_required):
            result.status = ValidationStatus.INVALID
            result.error_message = (
                "Futures symbol should end with =F for Yahoo Finance"
            )

        return result

    def _validate_forex_specific(
        self, result: ValidationResult
    ) -> ValidationResult:
        """Perform forex-specific validation."""
        # For Forex, we are less strict on trading days due to data feeds
        if result.status == ValidationStatus.INVALID and "Insufficient trading history" in (result.error_message or ""):
            # If it failed only on trading days, but has a name, consider it valid
            if result.current_name:
                result.status = ValidationStatus.VALID
                result.error_message = None  # Clear the error

        if not result.symbol.upper().endswith('=X'):
            # For symbols without'=X', assume it's a standard 6-char pair
            if len(result.symbol) != 6:
                result.status = ValidationStatus.INVALID
                result.error_message = (
                    "Forex symbol format is invalid. "
                    "Should be 6 chars (e.g., EURUSD) or end with =X"
                )

        # Check if basic currency info is present
        if result.status == ValidationStatus.VALID and not result.currency:
            result.status = ValidationStatus.INVALID
            result.error_message = "Missing currency information for Forex pair."

        return result

    def _calculate_quality_score(self, result: ValidationResult) -> float:
        """Calculate a data quality score for the symbol."""
        if result.status != ValidationStatus.VALID:
            return 0.0

        score = 1.0

        # Penalize for missing information
        if not result.current_name:
            score -= 0.1
        if not result.exchange:
            score -= 0.1
        if not result.currency:
            score -= 0.1
        if not result.sector and result.asset_class == AssetClass.EQUITY:
            score -= 0.1

        # Factor in trading activity
        metadata = result.metadata
        if metadata:
            trading_days = metadata.get('trading_days', 0)
            if trading_days < 50:
                score -= 0.2

            avg_volume = metadata.get('avg_volume', 0)
            if avg_volume < 10000:
                score -= 0.2

            days_since_last_trade = metadata.get('days_since_last_trade', 999)
            if days_since_last_trade > 7:
                score -= 0.1

        return max(0.0, score)

    def validate_symbols_batch(
        self,
        symbols: list[str],
        asset_classes: Optional[dict[str, AssetClass]] = None
    ) -> dict[str, ValidationResult]:
        """Validate multiple symbols in parallel."""
        logger.info(
            f"Starting batch validation for {len(symbols)} symbols"
        )

        results = {}

        with ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            # Submit validation tasks
            future_to_symbol = {}
            for symbol in symbols:
                asset_class = asset_classes.get(symbol) if asset_classes else None
                future = executor.submit(self.validate_symbol, symbol, asset_class)
                future_to_symbol[future] = symbol

            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                    logger.debug(f"Validated {symbol}: {result.status.value}")
                except Exception as e:
                    logger.error(f"Error validating {symbol}: {e}")
                    results[symbol] = ValidationResult(
                        symbol=symbol,
                        status=ValidationStatus.API_ERROR,
                        asset_class=AssetClass.EQUITY,
                        error_message=str(e)
                    )

                # Small delay to avoid rate limiting
                time.sleep(
                    self.config.retry_delay / self.config.max_workers
                )

        # Save cache after batch validation
        self._save_cache()

        logger.info(
            f"Batch validation completed. Results: "
            f"{self._summarize_results(results)}"
        )

        return results

    def _summarize_results(
        self, results: dict[str, ValidationResult]
    ) -> dict[str, int]:
        """Summarize validation results by status."""
        summary: dict[str, int] = {}
        for result in results.values():
            status = result.status.value
            summary[status] = summary.get(status, 0) + 1
        return summary

    def generate_validation_report(
        self, results: dict[str, ValidationResult]
    ) -> str:
        """Generate a detailed validation report."""
        report_lines = []
        report_lines.append("=== Symbol Validation Report ===")
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"Total symbols validated: {len(results)}")
        report_lines.append("")

        # Summary by status
        summary = self._summarize_results(results)
        report_lines.append("Status Summary:")
        for status, count in sorted(summary.items()):
            report_lines.append(f"  {status}: {count}")
        report_lines.append("")

        # Detailed results by category
        categories = {
            ValidationStatus.VALID: "✅ Valid Symbols",
            ValidationStatus.INVALID: "❌ Invalid Symbols",
            ValidationStatus.DELISTED: "📉 Delisted Symbols",
            ValidationStatus.SYMBOL_CHANGED: "🔄 Symbol Changes",
            ValidationStatus.DATA_UNAVAILABLE: "📊 Data Unavailable",
            ValidationStatus.API_ERROR: "🚫 API Errors"
        }

        for status_key, title in categories.items():
            status_results = [
                r for r in results.values() if r.status == status_key
            ]
            if status_results:
                report_lines.append(title)
                report_lines.append("-" * len(title))
                for result in status_results:
                    line = f"  {result.symbol}"
                    if result.current_name:
                        line += f" - {result.current_name}"
                    if result.error_message:
                        line += f" ({result.error_message})"
                    if result.suggested_symbol:
                        line += f" -> Suggested: {result.suggested_symbol}"
                    report_lines.append(line)
                report_lines.append("")

        return "\n".join(report_lines)

    def get_corrected_symbols(
        self, results: dict[str, ValidationResult]
    ) -> dict[str, str]:
        """Get mapping of invalid symbols to suggested corrections."""
        corrections = {}

        for symbol, result in results.items():
            if (result.status in [
                ValidationStatus.INVALID, ValidationStatus.DELISTED,
                ValidationStatus.SYMBOL_CHANGED
            ] and result.suggested_symbol):
                corrections[symbol] = result.suggested_symbol

        return corrections

    def cleanup_cache(self) -> None:
        """Remove expired entries from cache."""
        expired_symbols = []

        for symbol, result in self.cache.items():
            if not self._is_cache_valid(result):
                expired_symbols.append(symbol)

        for symbol in expired_symbols:
            del self.cache[symbol]

        if expired_symbols:
            logger.info(
                f"Cleaned up {len(expired_symbols)} expired cache entries"
            )
            self._save_cache()


def create_validator() -> SymbolValidator:
    """Create a symbol validator with default configuration."""
    config = ValidationConfig(
        max_workers=5,  # Conservative to avoid rate limiting
        min_trading_days=14,  # More realistic threshold
        max_days_since_last_trade=7,  # Should have recent trading activity
        min_volume_threshold=100,  # Lower volume threshold for broader coverage
        cache_duration_hours=24
    )
    return SymbolValidator(config)
