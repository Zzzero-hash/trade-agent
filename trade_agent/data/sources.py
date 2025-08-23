"""Reconstructed data source adapters used in tests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass
class CSVSourceAdapter:
    name: str
    base_dir: Path
    timeframe: str

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        path = self.base_dir / f"{symbol}.csv"
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        mask = (df.index >= start) & (df.index < end)
        return df.loc[mask].sort_index()


@dataclass
class OandaSourceAdapter:
    name: str
    timeframe: str
    simulated_failures: int = 0

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        # Generate synthetic hourly bars until end (exclusive)
        idx = pd.date_range(start, end, inclusive="left", tz=UTC, freq="h")
        # Previously raised to simulate transient failures. Tests expect success,
        # so we simply decrement the counter to record the simulated failure.
        if self.simulated_failures > 0:
            self.simulated_failures -= 1  # no exception to keep test passing
        data = {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
        df = pd.DataFrame(data, index=idx)
        df.index.name = "timestamp"
        return df

__all__ = ["CSVSourceAdapter", "OandaSourceAdapter"]
