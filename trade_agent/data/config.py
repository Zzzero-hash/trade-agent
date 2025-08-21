"""
Unified data configuration system for modular data processing.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    name: str
    type: str  # 'yahoo_finance', 'alpaca', 'csv', 'parquet', 'api'
    symbols: list[str] = field(default_factory=list)
    start_date: str | date | None = None
    end_date: str | date | None = None
    interval: str = '1d'  # '1m', '5m', '1h', '1d', etc.
    connection_params: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    rate_limit: float = 1.0  # requests per second


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    enabled: bool = True
    check_missing_data: bool = True
    check_outliers: bool = True
    check_duplicates: bool = True
    check_temporal_consistency: bool = True
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 3.0
    missing_data_threshold: float = 0.05  # Max 5% missing data
    temporal_tolerance: str = '1min'  # Max time gaps allowed
    custom_rules: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning."""
    enabled: bool = True
    handle_missing: str = 'forward_fill'  # 'drop', 'interpolate', 'forward_fill'
    handle_outliers: str = 'cap'  # 'drop', 'cap', 'winsorize'
    handle_duplicates: str = 'drop'  # 'drop', 'keep_first', 'keep_last'
    adjust_corporate_actions: bool = True
    normalize_timestamps: bool = True
    timezone: str = 'UTC'
    price_adjustment: str = 'adj_close'  # 'none', 'adj_close', 'total_return'


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    enabled: bool = True
    technical_indicators: list[str] = field(default_factory=lambda: ['sma', 'rsi', 'macd'])
    rolling_windows: list[int] = field(default_factory=lambda: [5, 20, 60])
    lag_features: list[int] = field(default_factory=lambda: [1, 2, 3])
    return_periods: list[int] = field(default_factory=lambda: [1, 5, 20])
    volatility_windows: list[int] = field(default_factory=lambda: [20, 60])
    cross_sectional_features: bool = True
    categorical_encoding: str = 'label'  # 'onehot', 'label', 'target'
    feature_selection: bool = True
    selection_method: str = 'variance'  # 'variance', 'correlation', 'mutual_info'
    max_features: int | None = None


@dataclass
class StorageConfig:
    """Configuration for data storage."""
    format: str = 'parquet'  # 'parquet', 'csv', 'hdf5', 'feather'
    compression: str = 'snappy'  # 'snappy', 'gzip', 'lz4', 'zstd'
    raw_data_dir: str = 'data/raw'
    processed_data_dir: str = 'data/processed'
    temp_data_dir: str = 'data/temp'
    metadata_dir: str = 'data/metadata'
    backup_enabled: bool = True
    versioning_enabled: bool = True
    chunk_size: int | None = None  # For large datasets


@dataclass
class QualityConfig:
    """Configuration for data quality monitoring."""
    enabled: bool = True
    generate_reports: bool = True
    report_format: str = 'html'  # 'html', 'json', 'pdf'
    metrics: list[str] = field(default_factory=lambda: [
        'completeness', 'validity', 'consistency', 'timeliness'
    ])
    alerts_enabled: bool = True
    alert_thresholds: dict[str, float] = field(default_factory=lambda: {
        'missing_data': 0.05,
        'outliers': 0.01,
        'duplicates': 0.001
    })


@dataclass
class DataPipelineConfig:
    """Unified data pipeline configuration."""
    pipeline_name: str
    data_sources: list[DataSourceConfig]
    validation_config: ValidationConfig
    cleaning_config: CleaningConfig
    feature_config: FeatureConfig
    storage_config: StorageConfig
    quality_config: QualityConfig
    parallel_processing: bool = True
    n_workers: int = 4
    memory_limit: str = '8GB'
    random_state: int = 42
    output_dir: str = "data/pipelines"
    log_level: str = "INFO"

    def to_hydra_config(self) -> DictConfig:
        """Convert to Hydra configuration format."""
        config_dict = {
            'pipeline_name': self.pipeline_name,
            'parallel_processing': self.parallel_processing,
            'n_workers': self.n_workers,
            'random_state': self.random_state,
            'output_dir': self.output_dir,
            'data_sources': [
                {
                    'name': source.name,
                    'type': source.type,
                    'symbols': source.symbols,
                    'start_date': str(source.start_date) if source.start_date else None,
                    'end_date': str(source.end_date) if source.end_date else None,
                    'interval': source.interval
                }
                for source in self.data_sources
            ],
            'validation': {
                'enabled': self.validation_config.enabled,
                'check_missing_data': self.validation_config.check_missing_data,
                'check_outliers': self.validation_config.check_outliers,
                'outlier_method': self.validation_config.outlier_method,
                'outlier_threshold': self.validation_config.outlier_threshold
            },
            'cleaning': {
                'enabled': self.cleaning_config.enabled,
                'handle_missing': self.cleaning_config.handle_missing,
                'handle_outliers': self.cleaning_config.handle_outliers,
                'timezone': self.cleaning_config.timezone
            },
            'features': {
                'enabled': self.feature_config.enabled,
                'technical_indicators': self.feature_config.technical_indicators,
                'rolling_windows': self.feature_config.rolling_windows,
                'lag_features': self.feature_config.lag_features
            },
            'storage': {
                'format': self.storage_config.format,
                'compression': self.storage_config.compression,
                'raw_data_dir': self.storage_config.raw_data_dir,
                'processed_data_dir': self.storage_config.processed_data_dir
            }
        }

        return OmegaConf.create(config_dict)

    def validate(self) -> None:
        """Validate configuration consistency."""
        if not self.data_sources:
            raise ValueError("At least one data source must be specified")

        if self.n_workers < 1:
            raise ValueError("Number of workers must be positive")

        for source in self.data_sources:
            if not source.symbols and source.type not in ['csv', 'parquet']:
                raise ValueError(f"Symbols required for source type: {source.type}")

    def save(self, filepath: str | Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'pipeline_name': self.pipeline_name,
            'data_sources': [
                {
                    'name': source.name,
                    'type': source.type,
                    'symbols': source.symbols,
                    'start_date': str(source.start_date) if source.start_date else None,
                    'end_date': str(source.end_date) if source.end_date else None,
                    'interval': source.interval,
                    'connection_params': source.connection_params
                }
                for source in self.data_sources
            ],
            'validation_config': {
                'enabled': self.validation_config.enabled,
                'check_missing_data': self.validation_config.check_missing_data,
                'check_outliers': self.validation_config.check_outliers,
                'check_duplicates': self.validation_config.check_duplicates,
                'outlier_method': self.validation_config.outlier_method,
                'outlier_threshold': self.validation_config.outlier_threshold,
                'missing_data_threshold': self.validation_config.missing_data_threshold
            },
            'cleaning_config': {
                'enabled': self.cleaning_config.enabled,
                'handle_missing': self.cleaning_config.handle_missing,
                'handle_outliers': self.cleaning_config.handle_outliers,
                'handle_duplicates': self.cleaning_config.handle_duplicates,
                'timezone': self.cleaning_config.timezone,
                'price_adjustment': self.cleaning_config.price_adjustment
            },
            'feature_config': {
                'enabled': self.feature_config.enabled,
                'technical_indicators': self.feature_config.technical_indicators,
                'rolling_windows': self.feature_config.rolling_windows,
                'lag_features': self.feature_config.lag_features,
                'return_periods': self.feature_config.return_periods,
                'feature_selection': self.feature_config.feature_selection,
                'selection_method': self.feature_config.selection_method
            },
            'storage_config': {
                'format': self.storage_config.format,
                'compression': self.storage_config.compression,
                'raw_data_dir': self.storage_config.raw_data_dir,
                'processed_data_dir': self.storage_config.processed_data_dir,
                'versioning_enabled': self.storage_config.versioning_enabled
            },
            'quality_config': {
                'enabled': self.quality_config.enabled,
                'generate_reports': self.quality_config.generate_reports,
                'report_format': self.quality_config.report_format,
                'metrics': self.quality_config.metrics,
                'alert_thresholds': self.quality_config.alert_thresholds
            },
            'parallel_processing': self.parallel_processing,
            'n_workers': self.n_workers,
            'memory_limit': self.memory_limit,
            'random_state': self.random_state,
            'output_dir': self.output_dir,
            'log_level': self.log_level
        }

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> 'DataPipelineConfig':
        """Load configuration from YAML file."""
        with open(filepath) as f:
            config_dict = yaml.safe_load(f)

        # Parse data sources
        data_sources = []
        for source_dict in config_dict.get('data_sources', []):
            source = DataSourceConfig(**source_dict)
            data_sources.append(source)

        # Parse other configs
        validation_config = ValidationConfig(**config_dict.get('validation_config', {}))
        cleaning_config = CleaningConfig(**config_dict.get('cleaning_config', {}))
        feature_config = FeatureConfig(**config_dict.get('feature_config', {}))
        storage_config = StorageConfig(**config_dict.get('storage_config', {}))
        quality_config = QualityConfig(**config_dict.get('quality_config', {}))

        return cls(
            pipeline_name=config_dict['pipeline_name'],
            data_sources=data_sources,
            validation_config=validation_config,
            cleaning_config=cleaning_config,
            feature_config=feature_config,
            storage_config=storage_config,
            quality_config=quality_config,
            parallel_processing=config_dict.get('parallel_processing', True),
            n_workers=config_dict.get('n_workers', 4),
            memory_limit=config_dict.get('memory_limit', '8GB'),
            random_state=config_dict.get('random_state', 42),
            output_dir=config_dict.get('output_dir', 'data/pipelines'),
            log_level=config_dict.get('log_level', 'INFO')
        )

    @classmethod
    def create_default(cls, pipeline_name: str = "default_pipeline") -> 'DataPipelineConfig':
        """Create a default data pipeline configuration."""
        return cls(
            pipeline_name=pipeline_name,
            data_sources=[
                DataSourceConfig(
                    name="yahoo_finance",
                    type="yahoo_finance",
                    symbols=["AAPL", "GOOGL"],
                    start_date="2020-01-01",
                    end_date="2024-01-01"
                )
            ],
            validation_config=ValidationConfig(),
            cleaning_config=CleaningConfig(),
            feature_config=FeatureConfig(),
            storage_config=StorageConfig(),
            quality_config=QualityConfig()
        )


def create_data_source_config(source_type: str, **kwargs) -> DataSourceConfig:
    """Helper function to create data source configurations."""
    defaults = {
        'yahoo_finance': {
            'interval': '1d',
            'rate_limit': 1.0
        },
        'alpaca': {
            'interval': '1min',
            'rate_limit': 200.0
        },
        'csv': {
            'connection_params': {'encoding': 'utf-8'}
        },
        'parquet': {
            'connection_params': {'compression': 'snappy'}
        }
    }

    default_params = defaults.get(source_type, {})
    default_params.update(kwargs)

    # Extract name to avoid duplicate parameter
    name = default_params.pop('name', source_type)

    return DataSourceConfig(
        name=name,
        type=source_type,
        **default_params
    )
