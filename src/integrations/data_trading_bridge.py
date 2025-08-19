"""Data-Trading Bridge
=================================

Purpose
-------
Provide an adapter layer that converts unified data pipeline outputs (raw OHLCV
and basic engineered features) into the enriched format expected by the trading
environments (including technical indicators, lagged / rolling statistics, and
placeholder supervised learning predictions: ``mu_hat`` & ``sigma_hat``).

Why
----
This allows us to immediately integrate the new pipeline with existing / new
RL & SL environments without rewriting the original environment expectations.

Key Responsibilities
--------------------
1. Load parquet output from the data pipeline.
2. Add technical indicators & observation features (lags, rolling stats,
   regime, etc.).
3. Inject mock supervised learning predictions when real models are absent.
4. Emit trading-ready parquet plus a sidecar JSON metadata file describing the
    transformation (schema version, columns, window size, etc.).

Notes
-----
The mock prediction logic is intentionally simple and deterministic to support
testing; it should be replaced by real model inference when available.
"""

import sys
from pathlib import Path

import pandas as pd


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = [
    "DataTradingBridge",
    "WorkflowBridge",
    "test_bridge_functionality",
    "test_trading_environment_integration",
]


class DataTradingBridge:
    """Convert pipeline output into trading environment format with features.

    Adds: returns, moving averages, volatility measures, RSI, MACD, volume
    features, mock ``mu_hat`` / ``sigma_hat``, lag features, rolling
    statistics, momentum, and a volatility regime flag. Ensures numeric dtypes
    (float32) for
    all feature columns (except identifier columns like ``symbol`` / ``date``).
    """

    def __init__(self, cache_dir: str = "data/bridge_cache") -> None:
        """
        Initialize the bridge.

        Args:
            cache_dir: Directory to cache processed data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def convert_pipeline_output_to_trading_format(
        self,
        pipeline_data_path: str,
        output_path: str | None = None,
        add_mock_predictions: bool = True,
        window_size: int = 30,
    ) -> str:
        """
        Convert data pipeline output to trading environment format.

        Args:
            pipeline_data_path: Path to data pipeline output parquet file
            output_path: Output path for converted data (optional)
            add_mock_predictions: Whether to add mock SL predictions
            window_size: Window size for feature engineering

        Returns:
            Path to converted data file
        """

        # Load pipeline data
        df = pd.read_parquet(pipeline_data_path)

        # Add basic features
        df_enhanced = self._add_basic_features(df)

        # Add mock SL predictions if needed
        if add_mock_predictions:
            df_enhanced = self._add_mock_sl_predictions(df_enhanced)

        # Add engineered features for observation space
        df_enhanced = self._add_observation_features(df_enhanced, window_size)

        # Generate output path if not provided
        if output_path is None:
            input_name = Path(pipeline_data_path).stem
            output_path = (
                self.cache_dir / f"{input_name}_trading_format.parquet"
            )

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save converted data and metadata
        df_enhanced.to_parquet(output_path)
        meta = {
            "schema_version": "1.0.0",
            "source_file": str(pipeline_data_path),
            "rows": int(df_enhanced.shape[0]),
            "columns": list(df_enhanced.columns),
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "window_size": window_size,
            "mock_predictions": add_mock_predictions,
        }
        meta_path = output_path.with_suffix('.metadata.json')
        try:
            import json
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        return str(output_path)

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add core technical indicators and return original plus features."""
        df = df.copy()

        # Ensure we have required price columns
        if 'Close' not in df.columns:
            raise ValueError("Close price column is required")

        # Calculate returns
        df['returns'] = df['Close'].pct_change()

        # Simple moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()

        # Volatility
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()

        # RSI (simplified)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (simplified)
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Volume features if available
        if 'Volume' in df.columns:
            df['volume_ma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
        else:
            df['volume_ma'] = 0
            df['volume_ratio'] = 1

        return df

    def _add_mock_sl_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject deterministic mock SL predictions for testing purposes."""
        df = df.copy()

        # Mock mu_hat (expected return prediction)
        # Use a simple mean-reverting signal based on recent returns
        recent_returns = df['returns'].rolling(window=5).mean()
        df['mu_hat'] = -0.1 * recent_returns  # Mean-reverting signal

    # Mock sigma_hat (volatility prediction)
        # Use recent volatility with some persistence
        df['sigma_hat'] = df['volatility_10'].rolling(window=5).mean()

        # Fill NaN values
        df['mu_hat'] = df['mu_hat'].fillna(0.0)
        df['sigma_hat'] = df['sigma_hat'].fillna(0.02)  # Default 2% volatility

        return df

    def _add_observation_features(
        self, df: pd.DataFrame, window_size: int
    ) -> pd.DataFrame:
        """Add lagged, rolling, momentum & regime features for observations."""
        df = df.copy()

        # Core price features (normalized)
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                df[f'{col.lower()}_norm'] = (
                    df[col] / df['Close'].rolling(window=20).mean()
                )

        # Technical indicators (already added)

        # Lag features for temporal dependencies
        lag_features = ['returns', 'volatility_5', 'rsi', 'macd']
        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

        # Rolling statistics
        stat_features = ['returns', 'volatility_5']
        for feature in stat_features:
            if feature in df.columns:
                df[f'{feature}_mean_5'] = df[feature].rolling(window=5).mean()
                df[f'{feature}_std_5'] = df[feature].rolling(window=5).std()
                df[f'{feature}_min_5'] = df[feature].rolling(window=5).min()
                df[f'{feature}_max_5'] = df[feature].rolling(window=5).max()
        # Cross-sectional / momentum style feature (single asset variant)
        df['price_momentum'] = df['Close'] / df['Close'].shift(5) - 1
        df['volatility_regime'] = (
            df['volatility_10']
            > df['volatility_10'].rolling(window=20).mean()
        ).astype(int)

        # Forward-fill missing values and ensure numeric types
        df = df.fillna(method='ffill')
        df = df.fillna(0.0)  # Fill remaining NaN with 0

        # Convert all non-string columns to numeric
        for col in df.columns:
            if col not in ['symbol', 'date'] and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill any remaining NaN from conversion
        return df.fillna(0.0)



class WorkflowBridge:
    """Orchestrate pipeline → features → trading format conversion."""

    def __init__(self) -> None:
        """Initialize workflow bridge."""
        self.data_bridge = DataTradingBridge()

    def run_data_to_trading_pipeline(
        self,
        symbols: list = None,
        start_date: str = "2024-01-01",
        end_date: str = "2024-01-10",
        output_dir: str = "data/bridge_outputs"
    ) -> dict[str, str]:
        """
        Run complete pipeline from data ingestion to trading-ready format.

        Args:
            symbols: List of symbols to process
            start_date: Start date for data
            end_date: End date for data
            output_dir: Output directory for processed data

        Returns:
            Dictionary mapping symbols to output file paths
        """
        if symbols is None:
            symbols = ['AAPL']


        # Import data pipeline components lazily
        from data import (
            CleaningConfig,
            DataOrchestrator,
            DataPipelineConfig,
            DataRegistry,
            FeatureConfig,
            QualityConfig,
            StorageConfig,
            ValidationConfig,
            create_data_source_config,
        )

        results: dict[str, str] = {}

        for symbol in symbols:

            # 1. Run data pipeline for this symbol
            data_source = create_data_source_config(
                'yahoo_finance',
                name=f'yahoo_{symbol.lower()}',
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
            )

            config = DataPipelineConfig(
                pipeline_name=f'bridge_pipeline_{symbol.lower()}',
                data_sources=[data_source],
                validation_config=ValidationConfig(enabled=True),
                cleaning_config=CleaningConfig(enabled=True),
                feature_config=FeatureConfig(enabled=True),
                storage_config=StorageConfig(
                    format='parquet',
                    processed_data_dir=f"{output_dir}/processed",
                    raw_data_dir=f"{output_dir}/raw",
                ),
                quality_config=QualityConfig(enabled=False),
                output_dir=output_dir,
            )

            # Run data pipeline
            registry = DataRegistry()
            orchestrator = DataOrchestrator(config, registry)
            pipeline_results = orchestrator.run_full_pipeline()

            # Get the processed data path
            storage_results = pipeline_results.get('storage', {})
            symbol_results = storage_results.get(f'yahoo_{symbol.lower()}', {})
            processed_path = symbol_results.get('processed_path')

            if processed_path and Path(processed_path).exists():

                # 2. Convert to trading format
                trading_format_path = (
                    self.data_bridge.convert_pipeline_output_to_trading_format(
                        processed_path,
                        output_path=(
                            f"{output_dir}/trading_format/"
                            f"{symbol.lower()}_trading_data.parquet"
                        ),
                    )
                )

                results[symbol] = trading_format_path
            else:
                pass

        return results


def test_bridge_functionality():
    """Ad-hoc test for bridge functionality with pipeline (dev utility)."""

    # Find existing pipeline output
    pipeline_files = list(Path("data/pipelines").rglob("*processed*.parquet"))

    if not pipeline_files:

        # Run workflow bridge to create data
        workflow = WorkflowBridge()
        results = workflow.run_data_to_trading_pipeline(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-05'
        )

        if results:
            return list(results.values())[0]
    else:
        # Use existing pipeline output
        pipeline_file = pipeline_files[0]

        # Convert to trading format
        bridge = DataTradingBridge()
        return bridge.convert_pipeline_output_to_trading_format(
            str(pipeline_file)
        )
    return None


def test_trading_environment_integration(trading_data_path: str) -> bool | None:
    """Ad-hoc smoke test for environment integration."""

    try:
        # Import trading environment
        sys.path.insert(0, 'src')
        from trade_agent.agents.envs.trading_env import TradingEnvironment

        # Create environment with converted data
        env = TradingEnvironment(
            data_file=trading_data_path,
            initial_capital=100000.0,
            transaction_cost=0.001,
            window_size=30
        )


        # Test environment functionality
        obs, info = env.reset()

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the bridge functionality
    trading_file = test_bridge_functionality()

    if trading_file and Path(trading_file).exists():
        # Test trading environment integration
        success = test_trading_environment_integration(trading_file)

        if success:
            pass
        else:
            pass
    else:
        pass
