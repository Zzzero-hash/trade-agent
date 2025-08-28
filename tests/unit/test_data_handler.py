from pathlib import Path

import pandas as pd

from trade_agent.engine.nodes.data_handler import ParquetStore


def test_parquet_store_roundtrip(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-08-27T00:00:00Z",
                 "2025-08-27T00:01:00Z",
                 "2025-08-27T00:02:00Z"],
                utc=True,
            ),
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [100.0, 110.0, 120.0],
        }
    )

    meta = store.write(df, "BTC-USD", "1m")
    assert meta.rows == 3

    out = store.read("BTC-USD", "1m")
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        out[["timestamp", "open", "high", "low", "close", "volume"]],
    )

    # Range filter
    sub = store.read("BTC-USD", "1m", end="2025-08-27T00:01:00Z")
    assert len(sub) == 2

    # Catalog list
    series = store.list_series()
    assert len(series) == 1
    assert series[0].symbol == "BTC-USD"
