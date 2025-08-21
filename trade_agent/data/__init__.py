"""
Trade Agent Data Engine

Unified data pipeline framework for financial data processing, validation,
cleaning, feature engineering, and quality monitoring.
"""

# Configuration classes
from .config import (
    CleaningConfig,
    DataPipelineConfig,
    DataSourceConfig,
    FeatureConfig,
    QualityConfig,
    StorageConfig,
    ValidationConfig,
    create_data_source_config,
)

# Pipeline orchestration
from .orchestrator import DataOrchestrator

# Registry and tracking
from .registry import DataRegistry


__all__ = [
    # Configuration
    'DataPipelineConfig',
    'DataSourceConfig',
    'ValidationConfig',
    'CleaningConfig',
    'FeatureConfig',
    'StorageConfig',
    'QualityConfig',
    'create_data_source_config',

    # Core components
    'DataRegistry',
    'DataOrchestrator'
]

__version__ = '1.0.0'
