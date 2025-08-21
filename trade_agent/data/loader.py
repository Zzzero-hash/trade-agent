"""Simplified timeseries loader combining sources with cache."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .cache import ParquetCache
from .sources import CSVSourceAdapter, OandaSourceAdapter


@dataclass
class LoadConfig:
	timeframe: str


Source = CSVSourceAdapter | OandaSourceAdapter


def load_timeseries(
	symbol: str,
	start: datetime,
	end: datetime,
	sources: Iterable[Source],
	cache: ParquetCache | None = None,
	config: LoadConfig | None = None,
) -> pd.DataFrame:
	timeframe = (config.timeframe if config else "1H")
	if cache:
		cached = cache.load(symbol, timeframe)
		if cached is not None:
			mask = (cached.index >= start) & (cached.index < end)
			if mask.any():
				sub = cached.loc[mask]
				if len(sub) >= 1 and sub.index.min() <= start and sub.index.max() >= end - (end - start) / max(len(sub), 1):
					return sub
	last_exc: Exception | None = None
	for src in sources:
		try:
			df = src.fetch(symbol, start, end)
			if cache and not df.empty:
				cache.store(symbol, timeframe, df)
			return df
		except Exception as e:  # pragma: no cover - retry path
			last_exc = e
			continue
	if last_exc:
		raise last_exc
	return pd.DataFrame()

__all__ = ["LoadConfig", "load_timeseries"]
