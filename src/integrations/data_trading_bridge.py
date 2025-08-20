"""Deprecated legacy integration bridge.

This module is retained as a thin compatibility shim so existing imports like
``from integrations.data_trading_bridge import WorkflowBridge`` continue to
work. All new code should use ``trade_agent.data.bridge`` or higher level
pipeline orchestration APIs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from trade_agent.data.bridge import convert_pipeline_output_to_trading_format
from trade_agent.data.config import (
    CleaningConfig,
    DataPipelineConfig,
    FeatureConfig,
    QualityConfig,
    StorageConfig,
    ValidationConfig,
    create_data_source_config,
)
from trade_agent.data.orchestrator import DataOrchestrator
from trade_agent.data.registry import DataRegistry


__all__ = ["DataTradingBridge", "WorkflowBridge"]


class DataTradingBridge:
    """Shim wrapper exposing the previous method name."""

    def __init__(self, cache_dir: str = "data/bridge_cache") -> None:  # noqa: D401
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        warnings.warn(
            (
                "DataTradingBridge is deprecated; use "
                "convert_pipeline_output_to_trading_format from "
                "trade_agent.data.bridge"
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    def convert_pipeline_output_to_trading_format(
        self,
        pipeline_data_path: str,
        output_path: str | None = None,
        add_mock_predictions: bool = True,  # kept for signature compatibility
        window_size: int = 30,  # unused but retained
    ) -> str:
        return convert_pipeline_output_to_trading_format(
            pipeline_data_path, output_path
        )


@dataclass
class WorkflowBridge:
    """High-level orchestration shim using the canonical pipeline modules."""

    output_dir: str = "data/bridge_outputs"

    def run_data_to_trading_pipeline(
        self,
        symbols: list[str] | None = None,
        start_date: str = "2024-01-01",
        end_date: str = "2024-01-10",
    ) -> dict[str, str]:
        symbols = symbols or ["AAPL"]
        results: dict[str, str] = {}

        for symbol in symbols:
            ds = create_data_source_config(
                "yahoo_finance",
                name=f"yahoo_{symbol.lower()}",
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
            )
            cfg = DataPipelineConfig(
                pipeline_name=f"bridge_pipeline_{symbol.lower()}",
                data_sources=[ds],
                validation_config=ValidationConfig(enabled=True),
                cleaning_config=CleaningConfig(enabled=True),
                feature_config=FeatureConfig(enabled=True),
                storage_config=StorageConfig(
                    format="parquet",
                    processed_data_dir=f"{self.output_dir}/processed",
                    raw_data_dir=f"{self.output_dir}/raw",
                ),
                quality_config=QualityConfig(enabled=False),
                output_dir=self.output_dir,
            )
            registry = DataRegistry()
            orchestrator = DataOrchestrator(cfg, registry)
            pipeline_results = orchestrator.run_full_pipeline()
            storage_results = pipeline_results.get("storage", {})
            symbol_results = storage_results.get(
                f"yahoo_{symbol.lower()}", {}
            )
            processed_path = symbol_results.get("processed_path")
            if processed_path and Path(processed_path).exists():
                out_file = (
                    f"{self.output_dir}/trading_format/"
                    f"{symbol.lower()}_trading_data.parquet"
                )
                results[symbol] = convert_pipeline_output_to_trading_format(
                    processed_path, out_file
                )
        return results


if __name__ == "__main__":  # pragma: no cover
    pass
