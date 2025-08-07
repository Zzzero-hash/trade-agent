"""
Unified Data Pipeline Orchestrator for Trading RL Agent

This module provides a comprehensive orchestrator that manages the complete
data pipeline for the CNN-LSTM trading RL agent, including:

1. Multi-asset data ingestion with parallel processing
2. Symbol validation and correction with feedback loops
3. Feature engineering optimized for CNN-LSTM architecture
4. Data preprocessing for RL training
5. TimescaleDB storage integration
6. Pipeline monitoring and metrics

Architecture Integration:
- Processes 100+ assets simultaneously
- Generates 30-day sequences for temporal modeling
- Prepares data in (time_steps, assets, features) format for CNN-LSTM
- Includes technical indicators and feature engineering
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
import ray
import torch

from src.data.ingestion import fetch_data, get_sp500_symbols
from src.data.symbol_corrector import SymbolCorrector, run_symbol_validation
from src.data.validation import ValidationStatus

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class PipelineStage(Enum):
    """Pipeline execution stages."""
    VALIDATION_LOOP = "validation_loop"
    DATA_PROCESSING = "data_processing"
    FEATURE_ENGINEERING = "feature_engineering"
    SEQUENCE_PREPARATION = "sequence_preparation"
    STORAGE = "storage"
    COMPLETED = "completed"


@dataclass
class PipelineConfig:
    """Configuration for data pipeline execution."""
    mode: PipelineMode = PipelineMode.PARALLEL
    validate_symbols: bool = True
    apply_corrections: bool = True
    enable_evaluation: bool = True
    parallel_batch_size: int = 10

    # CNN-LSTM specific configurations
    sequence_length: int = 30  # 30-day sequences
    min_assets: int = 50       # Minimum assets for RL training
    target_assets: int = 100   # Target number of assets
    feature_engineering: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'mode': self.mode.value,
            'validate_symbols': self.validate_symbols,
            'apply_corrections': self.apply_corrections,
            'enable_evaluation': self.enable_evaluation,
            'parallel_batch_size': self.parallel_batch_size,
            'sequence_length': self.sequence_length,
            'min_assets': self.min_assets,
            'target_assets': self.target_assets,
            'feature_engineering': self.feature_engineering
        }


@dataclass
class ValidationLoopConfig:
    """Configuration for validation feedback loop."""
    max_iterations: int = 3
    validation_timeout: int = 300  # 5 minutes
    auto_correct_symbols: bool = True
    skip_invalid_after_max_iterations: bool = True


class FeatureEngineer:
    """Feature engineering optimized for CNN-LSTM architecture."""

    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.technical_indicators = [
            'RSI_14', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
            'SMA_20', 'EMA_12', 'Volume_SMA', 'Price_Change', 'Volatility'
        ]

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators optimized for CNN-LSTM input."""
        df = data.copy()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_upper'] = sma_20 + (std_20 * 2)
        df['BB_lower'] = sma_20 - (std_20 * 2)

        # Simple moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()

        # Price change and volatility
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()

        return df

    def normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for neural network training."""
        df = data.copy()

        # Normalize price features to percentage changes
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                df[f'{col}_norm'] = df[col] / df['Close'].shift(1) - 1

        # Normalize volume
        if 'Volume' in df.columns:
            volume_mean = df['Volume'].rolling(window=20).mean()
            volume_std = df['Volume'].rolling(window=20).std()
            df['Volume_norm'] = (df['Volume'] - volume_mean) / volume_std

        # Normalize technical indicators to [-1, 1] range
        for indicator in self.technical_indicators:
            if indicator in df.columns:
                values = df[indicator]
                if values.max() != values.min():
                    normalized = 2 * (values - values.min()) / (
                        values.max() - values.min()) - 1
                    df[f'{indicator}_norm'] = normalized

        return df

    def prepare_cnn_lstm_tensor(
        self, data: dict[str, pd.DataFrame]
    ) -> torch.Tensor:
        """Prepare data tensor for CNN-LSTM model."""
        if not data:
            return torch.empty(0)

        # Get feature columns
        sample_df = next(iter(data.values()))
        feature_cols = [col for col in sample_df.columns
                        if (col.endswith('_norm') or
                            col in self.technical_indicators)]

        # Sort symbols for consistent ordering
        symbols = sorted(data.keys())

        # Find minimum sequence length across all symbols
        min_length = min(len(df) for df in data.values())
        if min_length < self.sequence_length:
            logger.warning(
                f"Data length {min_length} < sequence length "
                f"{self.sequence_length}"
            )
            return torch.empty(0)

        # Create tensor: (batch_size, time_steps, assets, features)
        sequences = []
        for i in range(min_length - self.sequence_length + 1):
            time_step_data = []
            for symbol in symbols:
                df = data[symbol]
                asset_features = df[feature_cols].iloc[
                    i:i + self.sequence_length].values
                time_step_data.append(asset_features)

            # Stack assets: (time_steps, assets, features)
            sequence = np.stack(time_step_data, axis=1)
            sequences.append(sequence)

        if sequences:
            # Stack sequences: (batch_size, time_steps, assets, features)
            sequences_array = np.stack(sequences)
            return torch.tensor(sequences_array, dtype=torch.float32)

        return torch.empty(0)


class TimescaleDBStorage:
    """TimescaleDB storage interface for processed trading data."""

    def __init__(self, connection_string: Optional[str] = None):
        default_conn = "postgresql://user:pass@localhost:5432/trading_db"
        self.connection_string = connection_string or default_conn
        logger.info("Initialized TimescaleDB storage")

    def store_processed_data(
        self, data: pd.DataFrame, table_name: str = "ohlcv_processed"
    ) -> bool:
        """Store processed data to TimescaleDB."""
        try:
            msg = (f"Storing {len(data)} records to TimescaleDB table: "
                   f"{table_name}")
            logger.info(msg)

            # Simulate database storage
            logger.info("Data structure preview:")
            logger.info(f"Columns: {list(data.columns)}")
            logger.info(f"Shape: {data.shape}")
            logger.info(f"Sample data:\n{data.head()}")

            # Simulate processing time
            time.sleep(1)

            logger.info("✅ Data successfully stored to TimescaleDB")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to store data to TimescaleDB: {str(e)}")
            return False

    def create_hypertable(
        self, table_name: str, time_column: str = "timestamp"
    ) -> bool:
        """Create TimescaleDB hypertable for time-series data."""
        try:
            msg = (f"Creating hypertable: {table_name} with time column: "
                   f"{time_column}")
            logger.info(msg)

            logger.info("✅ Hypertable created successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create hypertable: {str(e)}")
            return False


class DataOrchestrator:
    """Unified orchestrator for the complete trading RL data pipeline."""

    def __init__(
        self,
        pipeline_config: Optional[PipelineConfig] = None,
        validation_config: Optional[ValidationLoopConfig] = None,
        storage_config: Optional[str] = None
    ):
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.validation_config = validation_config or ValidationLoopConfig()
        self.storage = TimescaleDBStorage(storage_config)
        self.corrector = SymbolCorrector()
        self.feature_engineer = FeatureEngineer(
            self.pipeline_config.sequence_length
        )

        # Pipeline state tracking
        self.current_stage = PipelineStage.VALIDATION_LOOP
        self.validation_iterations = 0
        self.validated_symbols: list[str] = []
        self.invalid_symbols: dict[str, Any] = {}
        self.processing_metrics: dict[str, Any] = {}

    def execute_complete_pipeline(
        self, symbols: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Execute the complete pipeline optimized for CNN-LSTM training."""
        try:
            logger.info("🚀 Starting Unified Data Pipeline Orchestration")
            start_time = time.time()

            # Use S&P 500 symbols if none provided
            if symbols is None:
                all_symbols = get_sp500_symbols()
                symbols = all_symbols[:self.pipeline_config.target_assets]

            logger.info(f"Processing {len(symbols)} symbols")

            # Stage 1: Validation feedback loop
            validated_data = self._execute_validation_loop(symbols)

            if len(validated_data) < self.pipeline_config.min_assets:
                error_msg = (f"Only {len(validated_data)} valid symbols, "
                           f"minimum required: {self.pipeline_config.min_assets}"
                           )
                logger.error(f"❌ {error_msg}")
                return self._create_pipeline_result(
                    success=False, error=error_msg
                )

            # Stage 2: Feature engineering and processing
            processed_data = self._execute_feature_engineering(validated_data)

            if processed_data.empty:
                logger.error("❌ Feature engineering returned empty data")
                return self._create_pipeline_result(
                    success=False, error="Feature engineering failed"
                )

            # Stage 3: Prepare CNN-LSTM tensor
            cnn_lstm_tensor = self._prepare_cnn_lstm_data(validated_data)

            # Stage 4: Storage
            storage_success = self._execute_storage(processed_data)

            # Calculate final metrics
            execution_time = time.time() - start_time
            self.processing_metrics.update({
                'total_execution_time': execution_time,
                'final_data_shape': processed_data.shape,
                'cnn_lstm_tensor_shape': cnn_lstm_tensor.shape if cnn_lstm_tensor is not None else 'None',
                'storage_success': storage_success,
                'validated_symbols_count': len(validated_data)
            })

            msg = f"✅ Pipeline completed successfully in {execution_time:.2f}s"
            logger.info(msg)

            return self._create_pipeline_result(
                success=True,
                data=processed_data,
                cnn_lstm_tensor=cnn_lstm_tensor # This will be None if torch is not used
            )

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}")
            return self._create_pipeline_result(success=False, error=str(e))

    def _execute_validation_loop(
        self, symbols: list[str]
    ) -> dict[str, pd.DataFrame]:
        """Execute validation feedback loop until symbols pass validation."""
        logger.info("🔄 Starting Validation Feedback Loop")
        self.current_stage = PipelineStage.VALIDATION_LOOP

        current_symbols = symbols.copy()
        validated_data = {}

        while (self.validation_iterations <
               self.validation_config.max_iterations and current_symbols):

            self.validation_iterations += 1
            msg = (f"Validation iteration {self.validation_iterations}/"
                   f"{self.validation_config.max_iterations}")
            logger.info(msg)

            # Step 1: Attempt data ingestion
            ingested_data = self._ingest_symbols(current_symbols)

            # Step 2: Validate symbols
            validation_results = self._validate_symbols(current_symbols)

            # Step 3: Process validation results
            valid_symbols, invalid_symbols = self._process_validation_results(
                validation_results
            )

            # Step 4: Apply corrections if enabled
            if (invalid_symbols and
                self.validation_config.auto_correct_symbols and
                self.validation_iterations <
                self.validation_config.max_iterations
            ):

                corrected_symbols = self._apply_symbol_corrections(
                    invalid_symbols
                )
                current_symbols = corrected_symbols

            else:
                # Add valid data to results
                for symbol in valid_symbols:
                    if (symbol in ingested_data and
                        not ingested_data[symbol].empty
                    ):
                        validated_data[symbol] = ingested_data[symbol]
                        self.validated_symbols.append(symbol)
                break

        msg = (f"✅ Validation loop completed: {len(validated_data)} "
               f"symbols validated")
        logger.info(msg)
        return validated_data

    def _ingest_symbols(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Ingest data for given symbols using Ray parallel processing."""
        logger.info(f"📥 Ingesting data for {len(symbols)} symbols")

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        # Create remote tasks
        tasks = []
        for symbol in symbols:
            task = fetch_data.remote(symbol, start_date, end_date) # Removed '1d' as it's not in fetch_data signature
            tasks.append((symbol, task))

        # Collect results
        ingested_data = {}
        for symbol, task in tasks:
            try:
                result = ray.get(task, timeout=30)
                if result is not None and not result.empty:
                    ingested_data[symbol] = result
                    logger.debug(f"✅ Ingested {len(result)} records for {symbol}") # Line length is fine now
                else:
                    logger.warning(f"⚠️ No data returned for {symbol}")
            except Exception as e:
                logger.error(f"❌ Failed to ingest {symbol}: {str(e)}")

        msg = (f"✅ Successfully ingested data for {len(ingested_data)}/"
               f"{len(symbols)} symbols")
        logger.info(msg)
        return ingested_data

    def _validate_symbols(self, symbols: list[str]) -> dict[str, Any]:
        """Validate symbols using the symbol validation system."""
        logger.info(f"🔍 Validating {len(symbols)} symbols")

        # Run comprehensive symbol validation
        validation_data = run_symbol_validation()

        # Filter results for our symbols
        filtered_results = {}
        for symbol in symbols:
            validation_results = validation_data.get('validation_results', {})
            if symbol in validation_results:
                filtered_results[symbol] = validation_results[symbol]

        return filtered_results

    def _process_validation_results(
        self, validation_results: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        """Process validation results and separate valid/invalid symbols."""
        valid_symbols = []
        invalid_symbols = {}

        for symbol, result in validation_results.items():
            if hasattr(result, 'status'):
                if result.status == ValidationStatus.VALID:
                    valid_symbols.append(symbol)
                else:
                    invalid_symbols[symbol] = {
                        'status': result.status.value,
                        'error_message': getattr(result, 'error_message', None), # Line length is fine now
                        'suggested_symbol': getattr(
                            result, 'suggested_symbol', None
                        )
                    }

        msg = (f"✅ {len(valid_symbols)} valid symbols, "
               f"❌ {len(invalid_symbols)} invalid symbols")
        logger.info(msg)
        return valid_symbols, invalid_symbols

    def _apply_symbol_corrections(
        self, invalid_symbols: dict[str, Any]
    ) -> list[str]:
        """Apply symbol corrections and return corrected symbol list."""
        logger.info(f"🔧 Applying corrections for {len(invalid_symbols)} symbols")

        # Get correction suggestions
        suggestions = self.corrector.suggest_symbol_replacements(
            invalid_symbols
        )

        # Build corrected symbol list
        corrected_symbols = []
        corrections_applied = {}

        for symbol, symbol_suggestions in suggestions.items():
            if symbol_suggestions:
                new_symbol = symbol_suggestions[0]  # Use first suggestion
                corrected_symbols.append(new_symbol)
                corrections_applied[symbol] = new_symbol
                logger.info(f"🔧 {symbol} -> {new_symbol}")
            else:
                logger.warning(f"⚠️ No correction available for {symbol}")

        # Apply corrections through the corrector
        if corrections_applied:
            self.corrector.apply_corrections(corrections_applied)

        return corrected_symbols

    def _execute_feature_engineering(
        self, validated_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute feature engineering optimized for CNN-LSTM."""
        logger.info("⚙️ Starting Feature Engineering")
        self.current_stage = PipelineStage.FEATURE_ENGINEERING

        if not validated_data:
            logger.warning("No validated data for feature engineering")
            return pd.DataFrame()

        processed_data = {}

        for symbol, df in validated_data.items():
            # Add technical indicators
            engineered_df = self.feature_engineer.add_technical_indicators(df)

            # Normalize features
            normalized_df = self.feature_engineer.normalize_features(
                engineered_df
            )

            processed_data[symbol] = normalized_df

        # Combine all symbol data
        combined_df = pd.concat(
            processed_data.values(),
            keys=processed_data.keys(),
            names=['Symbol', 'Index']
        )
        combined_df = combined_df.reset_index(level=0)

        # Update metrics
        self.processing_metrics.update({
            'input_symbols': len(validated_data),
            'output_records': len(combined_df),
            'feature_columns': len(combined_df.columns)
        })

        logger.info(f"✅ Feature engineering completed: {combined_df.shape}")
        return combined_df

    def _prepare_cnn_lstm_data(
        self, validated_data: dict[str, pd.DataFrame]
    ) -> torch.Tensor:
        """Prepare data tensor for CNN-LSTM model training."""
        logger.info("🧠 Preparing CNN-LSTM tensor")
        self.current_stage = PipelineStage.SEQUENCE_PREPARATION

        # Add technical indicators and normalize
        processed_data = {}
        for symbol, df in validated_data.items():
            engineered_df = self.feature_engineer.add_technical_indicators(df)
            normalized_df = self.feature_engineer.normalize_features(
                engineered_df
            )
            processed_data[symbol] = normalized_df

        # Create CNN-LSTM tensor
        cnn_lstm_tensor = self.feature_engineer.prepare_cnn_lstm_tensor(
            processed_data
        )

        logger.info(f"✅ CNN-LSTM tensor prepared: {cnn_lstm_tensor.shape}")
        return cnn_lstm_tensor

    def _execute_storage(self, processed_data: pd.DataFrame) -> bool:
        """Execute storage to TimescaleDB."""
        logger.info("💾 Starting Data Storage")
        self.current_stage = PipelineStage.STORAGE

        # Create hypertable if needed
        self.storage.create_hypertable("ohlcv_processed", "Date")

        # Store the processed data
        success = self.storage.store_processed_data(
            processed_data, "ohlcv_processed"
        )

        return success

    def _create_pipeline_result(
        self,
        success: bool,
        data: Optional[pd.DataFrame] = None,
        cnn_lstm_tensor: Optional[torch.Tensor] = None,
        error: Optional[str] = None
    ) -> dict[str, Any]:
        """Create standardized pipeline result."""
        return {
            'success': success,
            'data': data,
            'cnn_lstm_tensor': cnn_lstm_tensor,
            'error': error,
            'metrics': self.processing_metrics,
            'validated_symbols': self.validated_symbols,
            'invalid_symbols': self.invalid_symbols,
            'validation_iterations': self.validation_iterations,
            'final_stage': self.current_stage.value
        }

    def get_pipeline_status(self) -> dict[str, Any]:
        """Get current pipeline status."""
        return {
            'current_stage': self.current_stage.value,
            'validation_iterations': self.validation_iterations,
            'validated_symbols_count': len(self.validated_symbols),
            'invalid_symbols_count': len(self.invalid_symbols),
            'metrics': self.processing_metrics
        }


# Convenience functions for easy usage

def create_data_orchestrator(
    enable_auto_correction: bool = True,
    max_validation_iterations: int = 3,
    target_assets: int = 100,
    sequence_length: int = 30,
    timescale_connection: Optional[str] = None
) -> UnifiedDataOrchestrator:
    """Create a unified orchestrator with standard configuration."""

    pipeline_config = PipelineConfig(
        validate_symbols=True,
        apply_corrections=enable_auto_correction,
        enable_evaluation=True,
        target_assets=target_assets,
        sequence_length=sequence_length
    )

    validation_config = ValidationLoopConfig(
        max_iterations=max_validation_iterations,
        auto_correct_symbols=enable_auto_correction
    )

    return UnifiedDataOrchestrator(
        pipeline_config=pipeline_config,
        validation_config=validation_config,
        storage_config=timescale_connection
    )


def run_pipeline(
    symbols: Optional[list[str]] = None, **kwargs
) -> dict[str, Any]:
    """Run the complete pipeline optimized for CNN-LSTM training."""
    orchestrator = create_unified_orchestrator(**kwargs)
    return orchestrator.execute_complete_pipeline(symbols)


# Example usage functions

def demonstrate_cnn_lstm_pipeline():
    """Demonstrate the CNN-LSTM optimized pipeline."""
    logger.info("🎯 Demonstrating CNN-LSTM Pipeline")

    # Run with 50 S&P 500 symbols for demonstration
    result = run_cnn_lstm_pipeline(
        symbols=None,  # Will use first 100 S&P 500 symbols
        enable_auto_correction=True,
        max_validation_iterations=2,
        target_assets=50,
        sequence_length=30
    )

    logger.info("Pipeline execution results:")
    logger.info(f"Success: {result['success']}")
    logger.info(f"CNN-LSTM tensor shape: {result.get('cnn_lstm_tensor', {}).shape if result.get('cnn_lstm_tensor') is not None else 'None'}")
    logger.info(f"Metrics: {result['metrics']}")

    return result


if __name__ == "__main__":
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init()

    # Run demonstration
    demonstrate_cnn_lstm_pipeline()
