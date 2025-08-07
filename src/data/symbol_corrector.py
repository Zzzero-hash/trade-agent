"""
Symbol Correction System

This module provides functionality to validate and correct symbols
in the MarketUniverse, ensuring all symbols are valid and up-to-date.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from .ingestion import Instrument, MarketUniverse
from .validation import AssetClass, SymbolValidator, ValidationStatus, create_validator

logger = logging.getLogger(__name__)


class SymbolCorrector:
    """Corrects and updates symbols in the MarketUniverse."""

    def __init__(self, validator: Optional[SymbolValidator] = None):
        """Initialize with a validator instance."""
        self.validator = validator or create_validator()
        self.corrections: dict[str, str] = {}
        self.replacements: dict[str, list[Instrument]] = {}

    def validate_market_universe(self) -> dict[str, Any]:
        """Validate all symbols in the current MarketUniverse."""
        logger.info("Starting comprehensive validation of MarketUniverse")

        # Get all instruments
        all_instruments = MarketUniverse.get_all_instruments()

        # Group by asset class for better reporting
        symbols_by_class: dict[AssetClass, list[str]] = {}
        for instrument in all_instruments:
            asset_class = self._map_instrument_to_asset_class(instrument)
            if asset_class not in symbols_by_class:
                symbols_by_class[asset_class] = []
            symbols_by_class[asset_class].append(instrument.symbol)

        # Validate each asset class
        all_results = {}
        validation_summary = {}

        for asset_class, symbols in symbols_by_class.items():
            logger.info(f"Validating {len(symbols)} {asset_class.value} symbols")

            # Create asset class mapping for batch validation
            asset_class_map = {symbol: asset_class for symbol in symbols}

            # Validate symbols
            results = self.validator.validate_symbols_batch(
                symbols, asset_class_map
            )

            all_results.update(results)
            validation_summary[asset_class.value] = self.validator._summarize_results(results)

        # Generate comprehensive report
        report = self.validator.generate_validation_report(all_results)

        # Identify corrections needed
        invalid_symbols = self._identify_corrections_needed(all_results)

        return {
            'validation_results': all_results,
            'validation_summary': validation_summary,
            'validation_report': report,
            'invalid_symbols': invalid_symbols,
            'total_symbols': len(all_results),
            'validation_timestamp': datetime.now().isoformat()
        }

    def _map_instrument_to_asset_class(self, instrument: Instrument) -> AssetClass:
        """Map an Instrument to its AssetClass enum."""
        asset_class_map = {
            'equity': AssetClass.EQUITY,
            'etf': AssetClass.ETF,
            'crypto': AssetClass.CRYPTO,
            'future': AssetClass.FUTURE,
            'reit': AssetClass.REIT,
            'volatility': AssetClass.VOLATILITY,
            'forex': AssetClass.FOREX
        }
        return asset_class_map.get(instrument.asset_class, AssetClass.EQUITY)

    def _identify_corrections_needed(self, results: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Identify symbols that need correction."""
        corrections_needed = {}

        for symbol, result in results.items():
            # Type narrowing: ensure result has the expected ValidationResult attributes
            if hasattr(result, 'status') and result.status in [
                ValidationStatus.INVALID,
                ValidationStatus.DELISTED,
                ValidationStatus.DATA_UNAVAILABLE,
                ValidationStatus.SYMBOL_CHANGED
            ]:
                corrections_needed[symbol] = {
                    'status': result.status.value,
                    'error_message': result.error_message,
                    'suggested_symbol': result.suggested_symbol,
                    'current_name': result.current_name,
                    'asset_class': result.asset_class.value,
                    'last_traded_date': result.last_traded_date.isoformat() if result.last_traded_date else None,
                    'data_quality_score': result.data_quality_score
                }

        return corrections_needed

    def suggest_symbol_replacements(self, invalid_symbols: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
        """Suggest replacement symbols for invalid ones."""
        suggestions = {}

        for symbol, info in invalid_symbols.items():
            asset_class = info['asset_class']
            suggestions[symbol] = self._get_replacement_suggestions(symbol, asset_class, info)

        return suggestions

    def _get_replacement_suggestions(self, symbol: str, asset_class: str, info: dict[str, Any]) -> list[str]:
        """Get replacement suggestions for a specific symbol."""
        suggestions = []

        # If we have a suggested symbol from validation, use it
        if info.get('suggested_symbol'):
            suggestions.append(info['suggested_symbol'])

        # Asset class specific suggestions
        if asset_class == 'equity':
            suggestions.extend(self._suggest_equity_replacements(symbol, info))
        elif asset_class == 'etf':
            suggestions.extend(self._suggest_etf_replacements(symbol, info))
        elif asset_class == 'crypto':
            suggestions.extend(self._suggest_crypto_replacements(symbol, info))
        elif asset_class == 'forex':
            suggestions.extend(self._suggest_forex_replacements(symbol, info))

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)

        return unique_suggestions[:3]  # Return top 3 suggestions

    def _suggest_equity_replacements(self, symbol: str, info: dict[str, Any]) -> list[str]:
        """Suggest equity replacement symbols."""
        suggestions = []

        # Common symbol transformations
        if '.' in symbol:
            # Try without the suffix (e.g., BRK.B -> BRK-B)
            suggestions.append(symbol.replace('.', '-'))

        # If it's a class B share, try class A
        if symbol.endswith('-B'):
            suggestions.append(symbol[:-2] + '-A')
        elif symbol.endswith('.B'):
            suggestions.append(symbol[:-2] + '.A')

        return suggestions

    def _suggest_etf_replacements(self, symbol: str, info: dict[str, Any]) -> list[str]:
        """Suggest ETF replacement symbols."""
        suggestions = []

        # Common ETF alternatives
        etf_alternatives = {
            'SPY': ['VOO', 'IVV', 'SPLG'],
            'QQQ': ['QQQM', 'VGT', 'VUG'],
            'IWM': ['VB', 'VTEB', 'VBK'],
            'VTI': ['ITOT', 'SWTSX', 'FZROX'],
            'VEA': ['IEFA', 'EFA', 'FTIHX'],
            'VWO': ['IEMG', 'EEM', 'FPADX']
        }

        if symbol in etf_alternatives:
            suggestions.extend(etf_alternatives[symbol])

        return suggestions

    def _suggest_crypto_replacements(self, symbol: str, info: dict[str, Any]) -> list[str]:
        """Suggest crypto replacement symbols."""
        suggestions = []

        # Ensure proper crypto format for Yahoo Finance
        if not symbol.endswith('-USD'):
            base_symbol = symbol.replace('-USD', '').replace('-USDT', '')
            suggestions.append(f"{base_symbol}-USD")

        return suggestions

    def _suggest_forex_replacements(self, symbol: str, info: dict[str, Any]) -> list[str]:
        """Suggest forex replacement symbols."""
        suggestions = []

        # Ensure proper forex format for Yahoo Finance
        if not symbol.endswith('=X'):
            base_symbol = symbol.replace('=X', '')
            suggestions.append(f"{base_symbol}=X")

        # Common forex pair alternatives
        forex_alternatives = {
            'EURUSD=X': ['EUR=X', 'USD=X'],
            'GBPUSD=X': ['GBP=X', 'USD=X'],
            'USDJPY=X': ['USD=X', 'JPY=X'],
            'AUDUSD=X': ['AUD=X', 'USD=X'],
            'USDCAD=X': ['USD=X', 'CAD=X'],
            'USDCHF=X': ['USD=X', 'CHF=X']
        }

        if symbol in forex_alternatives:
            suggestions.extend(forex_alternatives[symbol])

        return suggestions

    def apply_corrections(self, corrections: dict[str, str]) -> dict[str, Any]:
        """Apply symbol corrections to the MarketUniverse."""
        logger.info(f"Applying {len(corrections)} symbol corrections")

        correction_results: dict[str, Any] = {
            'applied_corrections': [],
            'failed_corrections': [],
            'validation_results': {}
        }

        # Validate new symbols before applying corrections
        new_symbols = list(corrections.values())
        validation_results = self.validator.validate_symbols_batch(new_symbols)

        correction_results['validation_results'] = validation_results

        for old_symbol, new_symbol in corrections.items():
            result = validation_results.get(new_symbol)

            if result and hasattr(result, 'status') and result.status == ValidationStatus.VALID:
                correction_results['applied_corrections'].append({
                    'old_symbol': old_symbol,
                    'new_symbol': new_symbol,
                    'new_name': result.current_name,
                    'data_quality_score': result.data_quality_score
                })
                logger.info(f"Correction applied: {old_symbol} -> {new_symbol}")
            else:
                error_msg = result.error_message if result else "Symbol not found"
                correction_results['failed_corrections'].append({
                    'old_symbol': old_symbol,
                    'new_symbol': new_symbol,
                    'error': error_msg
                })
                logger.warning(f"Correction failed: {old_symbol} -> {new_symbol} ({error_msg})")

        return correction_results

    def generate_corrected_universe_code(self, corrections: dict[str, str]) -> str:
        """Generate updated MarketUniverse code with corrections applied."""
        logger.info("Generating corrected MarketUniverse code")

        # This would generate the updated code for the MarketUniverse class
        # with corrections applied. For now, return a summary.

        correction_summary = []
        correction_summary.append("# Symbol Corrections Applied:")
        correction_summary.append("# " + "="*50)

        for old_symbol, new_symbol in corrections.items():
            correction_summary.append(f"# {old_symbol} -> {new_symbol}")

        correction_summary.append("")
        correction_summary.append("# Apply these corrections to src/data/ingestion.py")
        correction_summary.append("# in the MarketUniverse class methods")

        return "\n".join(correction_summary)

    def save_validation_report(self, validation_data: dict[str, Any], filename: str = "symbol_validation_report.txt"):
        """Save validation report to file."""
        try:
            with open(filename, 'w') as f:
                f.write("SYMBOL VALIDATION REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {validation_data['validation_timestamp']}\n")
                f.write(f"Total symbols validated: {validation_data['total_symbols']}\n\n")

                # Summary by asset class
                f.write("VALIDATION SUMMARY BY ASSET CLASS\n")
                f.write("-" * 40 + "\n")
                for asset_class, summary in validation_data['validation_summary'].items():
                    f.write(f"\n{asset_class.upper()}:\n")
                    for status, count in summary.items():
                        f.write(f"  {status}: {count}\n")

                f.write("\n" + "="*50 + "\n")
                f.write("DETAILED VALIDATION REPORT\n")
                f.write("="*50 + "\n")
                f.write(validation_data['validation_report'])

                # Invalid symbols section
                if validation_data['invalid_symbols']:
                    f.write("\n" + "="*50 + "\n")
                    f.write("SYMBOLS REQUIRING CORRECTION\n")
                    f.write("="*50 + "\n")

                    for symbol, info in validation_data['invalid_symbols'].items():
                        f.write(f"\n{symbol} ({info['asset_class']}):\n")
                        f.write(f"  Status: {info['status']}\n")
                        f.write(f"  Error: {info['error_message']}\n")
                        if info['current_name']:
                            f.write(f"  Name: {info['current_name']}\n")
                        if info['suggested_symbol']:
                            f.write(f"  Suggested: {info['suggested_symbol']}\n")
                        f.write(f"  Quality Score: {info['data_quality_score']:.2f}\n")

            logger.info(f"Validation report saved to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")
            return None


def run_symbol_validation() -> dict[str, Any]:
    """Run comprehensive symbol validation."""
    corrector = SymbolCorrector()
    return corrector.validate_market_universe()


def main():
    """Main function to run symbol validation and correction."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("Starting symbol validation and correction process")

    # Create corrector
    corrector = SymbolCorrector()

    # Validate current universe
    validation_data = corrector.validate_market_universe()

    # Save validation report
    report_file = corrector.save_validation_report(validation_data)

    # Print summary
    print("\n" + "="*60)
    print("SYMBOL VALIDATION SUMMARY")
    print("="*60)
    print(f"Total symbols validated: {validation_data['total_symbols']}")
    print(f"Invalid symbols found: {len(validation_data['invalid_symbols'])}")

    if validation_data['invalid_symbols']:
        print("\nSymbols requiring attention:")
        for symbol, info in validation_data['invalid_symbols'].items():
            print(f"  {symbol} ({info['asset_class']}): {info['status']}")

        # Get suggestions
        suggestions = corrector.suggest_symbol_replacements(validation_data['invalid_symbols'])

        print("\nSuggested replacements:")
        for symbol, symbol_suggestions in suggestions.items():
            if symbol_suggestions:
                print(f"  {symbol} -> {', '.join(symbol_suggestions)}")

    print(f"\nDetailed report saved to: {report_file}")
    print("="*60)

    return validation_data


if __name__ == "__main__":
    main()
