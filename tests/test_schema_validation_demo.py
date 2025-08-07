"""
Test script demonstrating the schema validation and column consistency system.

This script shows how to use the schema validation framework to ensure
consistent column counts and structure in your data pipeline.
"""
import logging

import numpy as np
import pandas as pd

from src.data.schema_aware_processing import (
    ensure_column_consistency_across_batches,
    process_data_with_validation,
)
from src.data.schema_config import default_config, get_stage_config

# Import our schema validation modules
from src.data.schema_validator import schema_validator, validate_pipeline_stage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_financial_data(
    n_rows: int = 100, n_symbols: int = 3
) -> pd.DataFrame:
    """Create sample financial data for testing."""
    np.random.seed(42)

    symbols = [f'STOCK_{i}' for i in range(n_symbols)]
    dates = pd.date_range(
        start='2024-01-01', periods=n_rows//n_symbols, freq='D'
    )

    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'Date': date,
                'Symbol': symbol,
                'Open': 100 + np.random.normal(0, 5),
                'High': 105 + np.random.normal(0, 5),
                'Low': 95 + np.random.normal(0, 5),
                'Close': 100 + np.random.normal(0, 5),
                'Volume': np.random.randint(1000, 10000)
            })

    return pd.DataFrame(data)


def create_inconsistent_data_batches() -> list[pd.DataFrame]:
    """Create batches of data with inconsistent columns for testing."""
    # Batch 1: Standard format
    batch1 = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
        'asset': ['AAPL'] * 10,
        'value': np.random.randn(10),
        'volume': np.random.randint(1000, 5000, 10)
    })

    # Batch 2: Missing volume column
    batch2 = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-11', periods=10, freq='D'),
        'asset': ['GOOGL'] * 10,
        'value': np.random.randn(10)
    })

    # Batch 3: Extra columns
    batch3 = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-21', periods=10, freq='D'),
        'asset': ['MSFT'] * 10,
        'value': np.random.randn(10),
        'volume': np.random.randint(1000, 5000, 10),
        'open': np.random.randn(10),
        'high': np.random.randn(10),
        'low': np.random.randn(10)
    })

    return [batch1, batch2, batch3]


def test_basic_schema_validation():
    """Test basic schema validation functionality."""
    print("\n" + "="*60)
    print("TESTING BASIC SCHEMA VALIDATION")
    print("="*60)

    # Create sample data
    df = create_sample_financial_data()
    print(f"Created sample data with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Test raw input validation
    print("\n1. Testing raw input validation...")
    result = validate_pipeline_stage(df, 'raw_input')
    print(f"Validation result: {result.is_valid}")
    if result.errors:
        print(f"Errors: {result.errors}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")

    # Test with missing required columns
    print("\n2. Testing with missing required columns...")
    df_missing = df.drop(columns=['Symbol'])  # Remove required column
    result = validate_pipeline_stage(df_missing, 'raw_input')
    print(f"Validation result: {result.is_valid}")
    print(f"Missing columns: {result.missing_columns}")
    print(f"Errors: {result.errors}")


def test_column_consistency_enforcement():
    """Test column consistency enforcement across batches."""
    print("\n" + "="*60)
    print("TESTING COLUMN CONSISTENCY ENFORCEMENT")
    print("="*60)

    # Create inconsistent batches
    batches = create_inconsistent_data_batches()

    print("Original batch shapes and columns:")
    for i, batch in enumerate(batches):
        print(f"Batch {i+1}: {batch.shape} - {list(batch.columns)}")

    # Enforce consistency
    print("\nEnforcing column consistency...")
    consistent_batches = ensure_column_consistency_across_batches(batches)

    print("\nConsistent batch shapes and columns:")
    for i, batch in enumerate(consistent_batches):
        print(f"Batch {i+1}: {batch.shape} - {list(batch.columns)}")


def test_schema_aware_processing():
    """Test the complete schema-aware processing pipeline."""
    print("\n" + "="*60)
    print("TESTING SCHEMA-AWARE PROCESSING PIPELINE")
    print("="*60)

    # Create sample data
    df = create_sample_financial_data(50, 2)
    print(f"Input data shape: {df.shape}")
    print(f"Input columns: {list(df.columns)}")

    try:
        # Process with schema validation
        print("\nRunning schema-aware processing...")
        processed_df = process_data_with_validation(df)

        print(f"Output data shape: {processed_df.shape}")
        print(f"Output columns: {list(processed_df.columns)}")

        # Validate the final output
        final_validation = validate_pipeline_stage(
            processed_df, 'final_output'
        )
        print(f"Final validation result: {final_validation.is_valid}")

    except Exception as e:
        print(f"Processing failed: {e}")


def test_validation_summary():
    """Test the validation summary functionality."""
    print("\n" + "="*60)
    print("TESTING VALIDATION SUMMARY")
    print("="*60)

    # Run a few validations
    df = create_sample_financial_data()
    validate_pipeline_stage(df, 'raw_input')
    validate_pipeline_stage(df.drop(columns=['Volume']), 'raw_input')

    # Get summary
    summary = schema_validator.get_validation_summary()
    print("Validation Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def demonstrate_configuration():
    """Demonstrate configuration options."""
    print("\n" + "="*60)
    print("DEMONSTRATING CONFIGURATION OPTIONS")
    print("="*60)

    print("Default configuration:")
    print(f"  Strict validation: {default_config.strict_validation}")
    print(f"  Auto fix issues: {default_config.auto_fix_issues}")
    print(f"  Max features: {default_config.max_features}")
    print(f"  Min features: {default_config.min_features}")

    # Show stage configurations
    print("\nStage configurations:")
    for stage in ['raw_input', 'cleaned', 'transformed', 'final_output']:
        config = get_stage_config(stage)
        print(f"  {stage}:")
        print(f"    Required columns: {config.required_columns}")
        print(f"    Max null percentage: {config.max_null_percentage}")


def main():
    """Run all tests and demonstrations."""
    print("SCHEMA VALIDATION AND COLUMN CONSISTENCY DEMO")
    print("=" * 60)

    # Run all tests
    test_basic_schema_validation()
    test_column_consistency_enforcement()
    test_schema_aware_processing()
    test_validation_summary()
    demonstrate_configuration()

    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)

    print("\nKey Benefits:")
    print("• Consistent column counts across pipeline stages")
    print("• Automatic detection and fixing of schema issues")
    print("• Configurable validation rules for different data types")
    print("• Detailed validation reporting and logging")
    print("• Batch processing with consistent schemas")


if __name__ == "__main__":
    main()
