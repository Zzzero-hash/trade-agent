"""Workflow bridge components providing minimal data→env pipeline.

Implements ``WorkflowBridge`` required by integration tests. It synthesizes
feature/target data (or could hook into a real data pipeline). Returns paths
to parquet files in the expected trading environment format.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


__all__ = ["WorkflowBridge"]


@dataclass
class WorkflowBridge:
    seed: int = 42

    def run_data_to_trading_pipeline(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
        output_dir: str,
    ) -> dict[str, str]:
        """Generate synthetic OHLCV + target features per symbol.

        Returns mapping from symbol to parquet file path. Columns include:
        Close, mu_hat, sigma_hat (minimum required by environment). Extra
        columns (Open, High, Low, Volume) included for realism.
        """
        rng = np.random.default_rng(self.seed)
        out = {}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for sym in symbols:
            idx = pd.date_range(start_date, end_date, freq="D")
            n = len(idx)
            base = 100 + rng.normal(0, 0.5, size=n).cumsum()
            df = pd.DataFrame(
                {
                    "Open": base + rng.normal(0, 0.1, size=n),
                    "High": base + rng.uniform(0, 0.3, size=n),
                    "Low": base - rng.uniform(0, 0.3, size=n),
                    "Close": base,
                    "Volume": rng.integers(1_000, 5_000, size=n),
                    "mu_hat": rng.normal(0.0, 0.002, size=n),
                    "sigma_hat": rng.uniform(0.01, 0.05, size=n),
                },
                index=idx,
            )
            file = Path(output_dir) / f"{sym.lower()}_trading.parquet"
            df.to_parquet(file)
            out[sym] = str(file)
        return out
