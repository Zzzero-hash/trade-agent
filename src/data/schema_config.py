"""
Configuration for schema validation and column consistency management.

This module provides configuration options for the schema validation system,
allowing users to customize validation rules and column consistency behavior.
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class SchemaConfig:
    """Configuration for schema validation behavior."""

    # Global settings
    strict_validation: bool = False  # If True, fail on any validation error
    auto_fix_issues: bool = True     # If True, attempt to fix issues
    log_validation_details: bool = True  # If True, log detailed info

    # Column consistency settings
    enforce_column_order: bool = True   # If True, enforce column order
    allow_extra_columns: bool = True    # If True, allow extra columns
    remove_extra_columns: bool = False  # If True, remove extra columns

    # Feature engineering settings
    max_features: int = 50              # Maximum features for model input
    min_features: int = 5               # Minimum features for model input
    preserve_feature_columns: bool = True  # If True, maintain feature set

    # Data type validation
    enforce_dtypes: bool = True         # If True, enforce expected data types
    auto_convert_dtypes: bool = True    # If True, attempt dtype conversion

    # Pipeline stage settings
    validate_raw_input: bool = True
    validate_transformed: bool = True
    validate_cleaned: bool = True
    validate_feature_engineered: bool = True
    validate_final_output: bool = True


@dataclass
class PipelineStageConfig:
    """Configuration for a specific pipeline stage."""
    stage_name: str
    required_columns: set[str]
    optional_columns: set[str]
    expected_dtypes: dict[str, str]
    min_rows: int = 1
    max_null_percentage: float = 0.1


# Default configuration instance
default_config = SchemaConfig()


# Common column sets for different data types
FINANCIAL_DATA_COLUMNS = {
    'timestamp', 'asset', 'value', 'volume', 'open', 'high', 'low', 'close'
}

TECHNICAL_INDICATOR_COLUMNS = {
    'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd',
    'bollinger_upper', 'bollinger_lower'
}

FEATURE_COLUMNS_TEMPLATE = [
    'asset_encoded',
    'price_change_1d',
    'price_change_5d',
    'volume_ratio',
    'volatility_5d',
    'rsi_14',
    'sma_ratio_5_20',
    'bollinger_position',
    'trading_volume_trend',
    'price_momentum'
]


def get_stage_config(stage_name: str) -> PipelineStageConfig:
    """
    Get configuration for a specific pipeline stage.

    Args:
        stage_name: Name of the pipeline stage

    Returns:
        PipelineStageConfig for the specified stage
    """
    stage_configs = {
        'raw_input': PipelineStageConfig(
            stage_name='raw_input',
            required_columns={'timestamp', 'value', 'asset'},
            optional_columns={
                'Symbol', 'Date', 'Close', 'Open', 'High', 'Low', 'Volume'
            },
            expected_dtypes={
                'timestamp': 'datetime64[ns]',
                'value': 'float64',
                'asset': 'object'
            },
            min_rows=1,
            max_null_percentage=0.1
        ),

        'cleaned': PipelineStageConfig(
            stage_name='cleaned',
            required_columns={'timestamp', 'value', 'asset'},
            optional_columns=set(),
            expected_dtypes={
                'timestamp': 'datetime64[ns]',
                'value': 'float64',
                'asset': 'object'
            },
            min_rows=1,
            max_null_percentage=0.05
        ),

        'transformed': PipelineStageConfig(
            stage_name='transformed',
            required_columns={'timestamp', 'value', 'asset'},
            optional_columns=set(),
            expected_dtypes={
                'timestamp': 'datetime64[ns]',
                'value': 'float64',
                'asset': 'int64'  # After categorical encoding
            },
            min_rows=1,
            max_null_percentage=0.05
        ),

        'feature_engineered': PipelineStageConfig(
            stage_name='feature_engineered',
            required_columns=set(),  # Variable based on feature selection
            optional_columns=set(),
            expected_dtypes={},
            min_rows=1,
            max_null_percentage=0.2  # More lenient for feature columns
        ),

        'final_output': PipelineStageConfig(
            stage_name='final_output',
            required_columns=set(),  # Will be determined dynamically
            optional_columns=set(),
            expected_dtypes={},
            min_rows=1,
            max_null_percentage=0.1
        )
    }

    if stage_name not in stage_configs:
        raise ValueError(f"Unknown pipeline stage: {stage_name}")

    return stage_configs[stage_name]


def update_config(**kwargs: Any) -> SchemaConfig:
    """
    Update the default configuration with new values.

    Args:
        **kwargs: Configuration parameters to update

    Returns:
        Updated SchemaConfig instance
    """
    config_dict = default_config.__dict__.copy()
    config_dict.update(kwargs)
    return SchemaConfig(**config_dict)
