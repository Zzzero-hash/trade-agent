"""
Data pipeline orchestrator for end-to-end data processing execution.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataPipelineConfig
from .registry import DataRegistry


class DataOrchestrator:
    """Orchestrates end-to-end data pipeline execution."""

    def __init__(self, config: DataPipelineConfig,
                 registry: DataRegistry | None = None) -> None:
        """Initialize data orchestrator."""
        self.config = config
        self.registry = registry or DataRegistry()
        self.run_id = None
        self.results = {}

    # Legacy logging removed; silent operation retained.

    def run_full_pipeline(self) -> dict[str, Any]:
        """Execute the complete data pipeline."""
    # (logging removed)

        try:
            # Register pipeline run
            self.run_id = self.registry.register_pipeline_run(self.config)
            self.registry.update_run_status(self.run_id, 'running')

            # Execute pipeline stages
            ingestion_results = self._run_data_ingestion()
            self.results['ingestion'] = ingestion_results

            validation_results = self._run_data_validation(ingestion_results)
            self.results['validation'] = validation_results

            cleaning_results = self._run_data_cleaning(ingestion_results)
            self.results['cleaning'] = cleaning_results

            if self.config.feature_config.enabled:
                feature_results = self._run_feature_engineering(
                    cleaning_results
                )
                self.results['features'] = feature_results
            else:
                feature_results = cleaning_results

            storage_results = self._run_data_storage(feature_results)
            self.results['storage'] = storage_results

            if self.config.quality_config.enabled:
                quality_results = self._run_quality_monitoring(feature_results)
                self.results['quality'] = quality_results

            # Mark as completed
            self.registry.update_run_status(self.run_id, 'completed')
            # (logging removed)

            return self.results

        except Exception as e:  # noqa: PERF203 (single except acceptable)
            error_msg = str(e)
            # (logging removed)
            if self.run_id:
                self.registry.update_run_status(self.run_id, 'failed', error_msg)
            raise

    def _run_data_ingestion(self) -> dict[str, Any]:
        """Execute data ingestion stage."""
    # (logging removed)

        ingestion_results = {}

        if (
            self.config.parallel_processing
            and len(self.config.data_sources) > 1
        ):
            # Parallel ingestion
            with ThreadPoolExecutor(
                max_workers=self.config.n_workers
            ) as executor:
                future_to_source = {
                    executor.submit(self._ingest_single_source, source): source
                    for source in self.config.data_sources
                }

                for future in as_completed(future_to_source):
                    source = future_to_source[future]
                    try:
                        result = future.result()
                        ingestion_results[source.name] = result
                    except Exception:
                        raise
        else:
            # Sequential ingestion
            for source in self.config.data_sources:
                result = self._ingest_single_source(source)
                ingestion_results[source.name] = result

        # Log lineage
        self.registry.log_data_lineage(
            self.run_id,
            'data_ingestion',
            parameters={'n_sources': len(self.config.data_sources)}
        )

        return ingestion_results

    def _ingest_single_source(self, source) -> dict[str, Any]:
        """Ingest data from a single source."""
    # (logging removed)

        try:
            if source.type == 'yahoo_finance':
                data = self._ingest_yahoo_finance(source)
            elif source.type == 'alpaca':
                data = self._ingest_alpaca(source)
            elif source.type == 'csv':
                data = self._ingest_csv(source)
            elif source.type == 'parquet':
                data = self._ingest_parquet(source)
            else:
                raise ValueError(f"Unsupported source type: {source.type}")

            # Save raw data
            raw_path = self._save_raw_data(source.name, data)

            # Log to registry
            self.registry.log_data_source(
                self.run_id,
                source.name,
                source.type,
                source.symbols,
                str(source.start_date) if source.start_date else "",
                str(source.end_date) if source.end_date else "",
                len(data),
                str(raw_path),
                raw_path.stat().st_size if raw_path.exists() else 0
            )

            return {
                'data': data,
                'path': str(raw_path),
                'records': len(data),
                'columns': list(data.columns),
                'date_range': {
                    'start': (
                        str(data.index.min())
                        if hasattr(data.index, 'min')
                        else None
                    ),
                    'end': (
                        str(data.index.max())
                        if hasattr(data.index, 'max')
                        else None
                    ),
                },
            }

        except Exception:
            raise

    def _ingest_yahoo_finance(self, source) -> pd.DataFrame:
        """Ingest data from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance package required for Yahoo Finance data"
            )

        data_frames = []
        for symbol in source.symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=source.start_date,
                end=source.end_date,
                interval=source.interval
            )
            df['symbol'] = symbol
            data_frames.append(df)

            # Rate limiting
            import time
            time.sleep(1.0 / source.rate_limit)

        return pd.concat(data_frames, axis=0)

    def _ingest_alpaca(self, source) -> pd.DataFrame:
        """Ingest data from Alpaca."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError:
            raise ImportError("alpaca-py package required for Alpaca data")

        # This is a placeholder - would need actual implementation
        warnings.warn("Alpaca data ingestion not fully implemented", stacklevel=2)

        # Create sample data structure
        dates = pd.date_range(start=source.start_date, end=source.end_date, freq='D')
        return pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 105,
            'low': np.random.randn(len(dates)).cumsum() + 95,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates)),
            'symbol': source.symbols[0] if source.symbols else 'SAMPLE'
        }, index=dates)


    def _ingest_csv(self, source) -> pd.DataFrame:
        """Ingest data from CSV file."""
        file_path = source.connection_params.get('file_path')
        if not file_path:
            raise ValueError("file_path required in connection_params for CSV source")

        return pd.read_csv(file_path, **source.connection_params)

    def _ingest_parquet(self, source) -> pd.DataFrame:
        """Ingest data from Parquet file."""
        file_path = source.connection_params.get('file_path')
        if not file_path:
            raise ValueError("file_path required in connection_params for Parquet source")

        return pd.read_parquet(file_path, **source.connection_params)

    def _save_raw_data(self, source_name: str, data: pd.DataFrame) -> Path:
        """Save raw data to storage."""
        output_dir = Path(self.config.storage_config.raw_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{source_name}_{self.run_id}.{self.config.storage_config.format}"
        file_path = output_dir / filename

        if self.config.storage_config.format == 'parquet':
            data.to_parquet(
                file_path,
                compression=self.config.storage_config.compression
            )
        elif self.config.storage_config.format == 'csv':
            data.to_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {self.config.storage_config.format}")

        return file_path

    def _run_data_validation(self, ingestion_results: dict[str, Any]) -> dict[str, Any]:
        """Execute data validation stage."""
        if not self.config.validation_config.enabled:
            return {}

    # (logging removed)
        validation_results = {}

        for source_name, source_result in ingestion_results.items():
            data = source_result['data']
            source_validation = self._validate_single_dataset(data, source_name)
            validation_results[source_name] = source_validation

        # Log lineage
        self.registry.log_data_lineage(
            self.run_id,
            'data_validation',
            parameters={'validation_config': self.config.validation_config.__dict__}
        )

        return validation_results

    def _validate_single_dataset(self, data: pd.DataFrame, source_name: str) -> dict[str, Any]:
        """Validate a single dataset."""
        validation_results = {}

        # Check missing data
        if self.config.validation_config.check_missing_data:
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            missing_passed = missing_ratio <= self.config.validation_config.missing_data_threshold

            validation_results['missing_data'] = {
                'ratio': missing_ratio,
                'threshold': self.config.validation_config.missing_data_threshold,
                'passed': missing_passed
            }

            self.registry.log_validation_result(
                self.run_id, 'validation', 'missing_data', missing_passed,
                score=missing_ratio, details=validation_results['missing_data']
            )

        # Check duplicates
        if self.config.validation_config.check_duplicates:
            duplicate_count = data.duplicated().sum()
            duplicate_ratio = duplicate_count / len(data)
            duplicate_passed = duplicate_ratio < 0.01  # Less than 1%

            validation_results['duplicates'] = {
                'count': duplicate_count,
                'ratio': duplicate_ratio,
                'passed': duplicate_passed
            }

            self.registry.log_validation_result(
                self.run_id, 'validation', 'duplicates', duplicate_passed,
                score=duplicate_ratio, details=validation_results['duplicates']
            )

        # Check outliers
        if self.config.validation_config.check_outliers:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outlier_results = {}

            for col in numeric_cols:
                if self.config.validation_config.outlier_method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((data[col] < Q1 - 1.5 * IQR) |
                               (data[col] > Q3 + 1.5 * IQR)).sum()
                elif self.config.validation_config.outlier_method == 'zscore':
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outliers = (z_scores > self.config.validation_config.outlier_threshold).sum()
                else:
                    outliers = 0

                outlier_ratio = outliers / len(data)
                outlier_results[col] = {
                    'count': outliers,
                    'ratio': outlier_ratio
                }

            avg_outlier_ratio = np.mean([r['ratio'] for r in outlier_results.values()])
            outlier_passed = avg_outlier_ratio < 0.05  # Less than 5%

            validation_results['outliers'] = {
                'by_column': outlier_results,
                'average_ratio': avg_outlier_ratio,
                'passed': outlier_passed
            }

            self.registry.log_validation_result(
                self.run_id, 'validation', 'outliers', outlier_passed,
                score=avg_outlier_ratio, details=validation_results['outliers']
            )

        return validation_results

    def _run_data_cleaning(self, ingestion_results: dict[str, Any]) -> dict[str, Any]:
        """Execute data cleaning stage."""
        if not self.config.cleaning_config.enabled:
            return ingestion_results

    # (logging removed)
        cleaning_results = {}

        for source_name, source_result in ingestion_results.items():
            data = source_result['data'].copy()
            cleaned_data = self._clean_single_dataset(data)

            cleaning_results[source_name] = {
                **source_result,
                'data': cleaned_data,
                'cleaning_applied': True
            }

        # Log lineage
        self.registry.log_data_lineage(
            self.run_id,
            'data_cleaning',
            parameters={'cleaning_config': self.config.cleaning_config.__dict__}
        )

        return cleaning_results

    def _clean_single_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean a single dataset."""
    # (logging removed)

        # Handle duplicates
        if self.config.cleaning_config.handle_duplicates == 'drop':
            data = data.drop_duplicates()
        elif self.config.cleaning_config.handle_duplicates == 'keep_first':
            data = data.drop_duplicates(keep='first')
        elif self.config.cleaning_config.handle_duplicates == 'keep_last':
            data = data.drop_duplicates(keep='last')

        # Handle missing data
        if self.config.cleaning_config.handle_missing == 'forward_fill':
            data = data.fillna(method='ffill')
        elif self.config.cleaning_config.handle_missing == 'drop':
            data = data.dropna()
        elif self.config.cleaning_config.handle_missing == 'interpolate':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].interpolate()

        # Handle outliers
        if self.config.cleaning_config.handle_outliers == 'cap':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

        # Normalize timestamps
        if self.config.cleaning_config.normalize_timestamps:
            if hasattr(data.index, 'tz_localize'):
                try:
                    if data.index.tz is None:
                        data.index = data.index.tz_localize(self.config.cleaning_config.timezone)
                    else:
                        data.index = data.index.tz_convert(self.config.cleaning_config.timezone)
                except Exception:
                    pass

        return data

    def _run_feature_engineering(self, cleaning_results: dict[str, Any]) -> dict[str, Any]:
        """Execute feature engineering stage."""
    # (logging removed)

        # Import the existing feature engineering module
        try:
            from trade_agent.features.build import (
                compute_rolling_stats,
                compute_technical_indicators,
            )
        except ImportError:
            pass
            return cleaning_results

        feature_results = {}

        for source_name, source_result in cleaning_results.items():
            data = source_result['data'].copy()

            # Apply feature engineering
            if 'close' in data.columns:
                # Add rolling statistics
                rolling_features = compute_rolling_stats(
                    data, windows=self.config.feature_config.rolling_windows
                )
                data = pd.concat([data, rolling_features], axis=1)

                # Add technical indicators
                if hasattr(self, '_compute_technical_indicators'):
                    tech_features = self._compute_technical_indicators(data)
                    data = pd.concat([data, tech_features], axis=1)

            feature_results[source_name] = {
                **source_result,
                'data': data,
                'features_applied': True
            }

        # Log lineage
        self.registry.log_data_lineage(
            self.run_id,
            'feature_engineering',
            parameters={'feature_config': self.config.feature_config.__dict__}
        )

        return feature_results

    def _run_data_storage(self, processing_results: dict[str, Any]) -> dict[str, Any]:
        """Execute data storage stage."""
    # (logging removed)

        storage_results = {}
        output_dir = Path(self.config.storage_config.processed_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for source_name, source_result in processing_results.items():
            data = source_result['data']

            # Save processed data
            filename = f"{source_name}_processed_{self.run_id}.{self.config.storage_config.format}"
            file_path = output_dir / filename

            if self.config.storage_config.format == 'parquet':
                data.to_parquet(
                    file_path,
                    compression=self.config.storage_config.compression
                )
            elif self.config.storage_config.format == 'csv':
                data.to_csv(file_path)

            storage_results[source_name] = {
                **source_result,
                'processed_path': str(file_path),
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            }

        # Log lineage
        self.registry.log_data_lineage(
            self.run_id,
            'data_storage',
            output_path=str(output_dir),
            parameters={'storage_config': self.config.storage_config.__dict__}
        )

        return storage_results

    def _run_quality_monitoring(self, processing_results: dict[str, Any]) -> dict[str, Any]:
        """Execute quality monitoring stage."""
    # (logging removed)

        quality_results = {}

        for source_name, source_result in processing_results.items():
            data = source_result['data']

            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(data, source_name)
            quality_results[source_name] = metrics

        return quality_results

    def _calculate_quality_metrics(self, data: pd.DataFrame, source_name: str) -> dict[str, Any]:
        """Calculate quality metrics for a dataset."""
        metrics = {}

        # Completeness
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        metrics['completeness'] = completeness

        self.registry.log_quality_metric(
            self.run_id, f"{source_name}_completeness", completeness,
            threshold=0.95, passed=completeness >= 0.95
        )

        # Validity (basic check for numeric ranges)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        validity_scores = []

        for col in numeric_cols:
            if col in ['open', 'high', 'low', 'close']:
                # Price columns should be positive
                valid_ratio = (data[col] > 0).sum() / len(data)
                validity_scores.append(valid_ratio)
            elif col == 'volume':
                # Volume should be non-negative
                valid_ratio = (data[col] >= 0).sum() / len(data)
                validity_scores.append(valid_ratio)

        validity = np.mean(validity_scores) if validity_scores else 1.0
        metrics['validity'] = validity

        self.registry.log_quality_metric(
            self.run_id, f"{source_name}_validity", validity,
            threshold=0.98, passed=validity >= 0.98
        )

        # Consistency (check for logical relationships)
        consistency_scores = []

        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Low
            consistency_scores.append(
                (data['high'] >= data['low']).sum() / len(data)
            )
            # High should be >= Open and Close
            consistency_scores.append(
                ((data['high'] >= data['open']) & (data['high'] >= data['close'])).sum() / len(data)
            )
            # Low should be <= Open and Close
            consistency_scores.append(
                ((data['low'] <= data['open']) & (data['low'] <= data['close'])).sum() / len(data)
            )

        consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        metrics['consistency'] = consistency

        self.registry.log_quality_metric(
            self.run_id, f"{source_name}_consistency", consistency,
            threshold=0.99, passed=consistency >= 0.99
        )

        # Timeliness (check for data freshness)
        if hasattr(data.index, 'max'):
            try:
                latest_date = pd.to_datetime(data.index.max())
                days_old = (pd.Timestamp.now() - latest_date).days
                timeliness = max(0, 1 - days_old / 365)  # Decay over a year
            except Exception:
                timeliness = 0.5  # Neutral score if can't determine
        else:
            timeliness = 0.5

        metrics['timeliness'] = timeliness

        self.registry.log_quality_metric(
            self.run_id, f"{source_name}_timeliness", timeliness,
            threshold=0.8, passed=timeliness >= 0.8
        )

        return metrics
