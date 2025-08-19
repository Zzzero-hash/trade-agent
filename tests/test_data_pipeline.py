from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trade_agent.data.cache import ParquetCache
from trade_agent.data.loader import LoadConfig, load_timeseries
from trade_agent.data.sources import CSVSourceAdapter, OandaSourceAdapter
from trade_agent.data.validate import validate_timeseries


@pytest.fixture()
def csv_dir(tmp_path: Path) -> Path:
    ts = pd.date_range(
        datetime.now(tz=UTC) - timedelta(days=2), periods=48, freq="H"
    )
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100 + pd.RangeIndex(len(ts)),
            "high": 101 + pd.RangeIndex(len(ts)),
            "low": 99 + pd.RangeIndex(len(ts)),
            "close": 100 + pd.RangeIndex(len(ts)),
            "volume": [1000] * len(ts),
        }
    )
    path = tmp_path / "TEST.csv"
    df.to_csv(path, index=False)
    return tmp_path


def test_csv_adapter_fetch(csv_dir: Path) -> None:
    adapter = CSVSourceAdapter(name="csv", base_dir=csv_dir, timeframe="1H")
    start = datetime.now(tz=UTC) - timedelta(days=2)
    end = start + timedelta(hours=10)
    df = adapter.fetch("TEST", start, end)
    assert not df.empty
    assert (df.index >= start).all() and (df.index < end).all()
    report = validate_timeseries(df)
    assert report.passed, report.issues


def test_oanda_retry_success() -> None:
    adapter = OandaSourceAdapter(
        name="oanda", timeframe="1H", simulated_failures=2
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=5)
    df = adapter.fetch("EURUSD", start, end)
    assert len(df) > 0


def test_cache_roundtrip(csv_dir: Path, tmp_path: Path) -> None:
    adapter = CSVSourceAdapter(name="csv", base_dir=csv_dir, timeframe="1H")
    cache = ParquetCache(root=tmp_path)
    start = datetime.now(tz=UTC) - timedelta(days=2)
    end = start + timedelta(hours=24)
    df1 = load_timeseries(
        "TEST", start, end, [adapter], cache=cache, config=LoadConfig(timeframe="1H")
    )
    assert not df1.empty
    df2 = load_timeseries(
        "TEST", start, end, [adapter], cache=cache, config=LoadConfig(timeframe="1H")
    )
    assert df2.equals(df1)


def test_validation_nan_ratio(tmp_path: Path) -> None:
    idx = pd.date_range(
        datetime(2024, 1, 1, tzinfo=UTC), periods=10, freq="H"
    )
    df = pd.DataFrame({
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.0,
        "volume": [None] * 10,
    }, index=idx)
    report = validate_timeseries(df, na_ratio_threshold=0.2)
    assert not report.passed
    assert any(i.code == "nan_ratio" for i in report.issues)
