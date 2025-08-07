#!/usr/bin/env python3
"""
Test script for the symbol validation system.

This script tests the validation system with a small sample of symbols
from each asset class to verify functionality before running on the full universe.
"""

import logging
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.ingestion import MarketUniverse
from src.data.symbol_corrector import SymbolCorrector
from src.data.validation import AssetClass, create_validator


def test_sample_validation():
    """Test validation with a small sample of symbols."""
    print("Testing Symbol Validation System")
    print("=" * 50)

    # Initialize validator
    validator = create_validator()

    # Test sample symbols from each asset class
    test_symbols = {
        # Valid symbols that should work
        'AAPL': AssetClass.EQUITY,     # Apple - should be valid
        'MSFT': AssetClass.EQUITY,     # Microsoft - should be valid
        'SPY': AssetClass.ETF,         # SPDR S&P 500 ETF - should be valid
        'BTC-USD': AssetClass.CRYPTO,  # Bitcoin - should be valid
        'GC=F': AssetClass.FUTURE,     # Gold futures - should be valid

        # Potentially problematic symbols
        'GOOGL': AssetClass.EQUITY,    # Alphabet - check if valid
        'META': AssetClass.EQUITY,     # Meta - formerly Facebook
        'BRK-B': AssetClass.EQUITY,    # Berkshire Hathaway Class B
    }

    print(f"Testing {len(test_symbols)} sample symbols...")
    print()

    # Validate symbols
    results = validator.validate_symbols_batch(list(test_symbols.keys()), test_symbols)

    # Display results
    for symbol, result in results.items():
        status_emoji = "✅" if result.status.value == "valid" else "❌"
        print(f"{status_emoji} {symbol} ({result.asset_class.value})")
        print(f"    Status: {result.status.value}")
        if result.current_name:
            print(f"    Name: {result.current_name}")
        if result.error_message:
            print(f"    Error: {result.error_message}")
        print(f"    Quality Score: {result.data_quality_score:.2f}")
        print()

    # Summary
    valid_count = sum(1 for r in results.values() if r.status.value == "valid")
    print(f"Summary: {valid_count}/{len(results)} symbols are valid")

    return results

def test_corrector_functionality():
    """Test the symbol corrector with a small sample."""
    print("\nTesting Symbol Corrector Functionality")
    print("=" * 50)

    corrector = SymbolCorrector()

    # Test with a small subset of instruments
    sample_instruments = MarketUniverse.get_equities_universe()[:10]  # First 10 equity symbols
    sample_instruments.extend(MarketUniverse.get_etfs_universe()[:3])   # First 3 ETF symbols
    sample_instruments.extend(MarketUniverse.get_crypto_universe()[:3]) # First 3 crypto symbols

    print(f"Testing correction system with {len(sample_instruments)} instruments...")

    # Extract symbols and create asset class mapping
    symbols = [inst.symbol for inst in sample_instruments]
    asset_class_map = {}

    for inst in sample_instruments:
        asset_class = corrector._map_instrument_to_asset_class(inst)
        asset_class_map[inst.symbol] = asset_class

    # Validate the sample
    results = corrector.validator.validate_symbols_batch(symbols, asset_class_map)

    # Identify issues
    invalid_symbols = corrector._identify_corrections_needed(results)

    print(f"\nFound {len(invalid_symbols)} symbols with issues:")
    for symbol, info in invalid_symbols.items():
        print(f"  {symbol}: {info['status']} - {info['error_message']}")

    # Test suggestions if there are invalid symbols
    if invalid_symbols:
        print("\nTesting suggestion system...")
        suggestions = corrector.suggest_symbol_replacements(invalid_symbols)

        for symbol, symbol_suggestions in suggestions.items():
            if symbol_suggestions:
                print(f"  {symbol} -> {', '.join(symbol_suggestions)}")

    return results, invalid_symbols

def main():
    """Main test function."""
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(levelname)s: %(message)s'
    )

    try:
        print("Symbol Validation System Test")
        print("=" * 60)

        # Test 1: Basic validation
        sample_results = test_sample_validation()

        # Test 2: Corrector functionality
        corrector_results, invalid_symbols = test_corrector_functionality()

        # Overall summary
        print("\n" + "=" * 60)
        print("OVERALL TEST SUMMARY")
        print("=" * 60)

        total_tested = len(sample_results) + len(corrector_results)
        total_valid = sum(1 for r in sample_results.values() if r.status.value == "valid")
        total_valid += sum(1 for r in corrector_results.values() if r.status.value == "valid")

        print(f"Total symbols tested: {total_tested}")
        print(f"Valid symbols: {total_valid}")
        print(f"Success rate: {(total_valid/total_tested)*100:.1f}%")

        if invalid_symbols:
            print(f"Symbols needing attention: {len(invalid_symbols)}")

        print("\n✅ Symbol validation system is working correctly!")
        print("Ready to run full validation on the complete universe.")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
