"""Quick benchmark for data loading & cache effectiveness."""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from trade_agent.data.cache import ParquetCache
from trade_agent.data.loader import LoadConfig, load_timeseries
from trade_agent.data.sources import CSVSourceAdapter


def _ensure_csv(dir_: Path) -> None:
    path = dir_ / "BENCH.csv"
    if path.exists():
        return
    ts = pd.date_range(
        datetime.now(tz=UTC) - timedelta(days=5),
        periods=24 * 5,
        freq="H",
    )
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
    )
    df.to_csv(path, index=False)


def main() -> None:  # pragma: no cover
    base = Path("data/bench")
    base.mkdir(parents=True, exist_ok=True)
    _ensure_csv(base)
    adapter = CSVSourceAdapter(name="csv", base_dir=base, timeframe="1H")
    cache = ParquetCache(root=base / "cache")
    start = datetime.now(tz=UTC) - timedelta(days=5)
    end = start + timedelta(days=3)
    cfg = LoadConfig(timeframe="1H")
    t0 = time.perf_counter()
    load_timeseries("BENCH", start, end, [adapter], cache=cache, config=cfg)
    time.perf_counter() - t0
    t1 = time.perf_counter()
    load_timeseries("BENCH", start, end, [adapter], cache=cache, config=cfg)
    time.perf_counter() - t1


if __name__ == "__main__":  # pragma: no cover
    main()
