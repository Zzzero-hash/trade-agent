"""Simple parquet caching utility (reconstructed).

Only implements the subset used in tests: storing and retrieving
timeseries dataframes by (symbol, timeframe, start, end).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ParquetCache:
	root: Path

	def __post_init__(self) -> None:  # pragma: no cover - trivial
		self.root.mkdir(parents=True, exist_ok=True)

	def _path(self, symbol: str, timeframe: str) -> Path:
		return self.root / f"{symbol}_{timeframe}.parquet"

	def load(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
		p = self._path(symbol, timeframe)
		if not p.exists():
			return None
		try:
			df = pd.read_parquet(p)
		except Exception:
			return None
		if "timestamp" in df.columns:
			df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
			df = df.set_index("timestamp")
		return df.sort_index()

	def store(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
		out = df.reset_index() if df.index.name == "timestamp" else df
		out.to_parquet(self._path(symbol, timeframe), index=False)

__all__ = ["ParquetCache"]
