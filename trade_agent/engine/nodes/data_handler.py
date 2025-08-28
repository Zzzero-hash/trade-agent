from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml


CATALOG_FILENAME = "catalog.yaml"
DATA_SUBDIR = "ohlcv"


@dataclass
class SeriesMeta:
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    rows: int
    last_updated: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "rows": self.rows,
            "last_updated": self.last_updated.isoformat(),
        }

    @staticmethod
    def from_dict(symbol: str, timeframe: str, d: dict[str, Any]) -> SeriesMeta:
        return SeriesMeta(
            symbol=symbol,
            timeframe=timeframe,
            start=pd.to_datetime(d["start"], utc=True).to_pydatetime(),
            end=pd.to_datetime(d["end"], utc=True).to_pydatetime(),
            rows=int(d["rows"]),
            last_updated=pd.to_datetime(d["last_updated"], utc=True).to_pydatetime(),
        )


class ParquetStore:
    """
    Layout (Hive partitions)
        <root>/ohlcv/
            symbol=BTC-USD/timeframe=1m/year=2025/month=08/part-*.parquet
    Catalog file:
        <root>/catalog.yaml
    root default to ./data (ignored by VCS).
    """
    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_root = self.root / DATA_SUBDIR
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / CATALOG_FILENAME
        self._catalog = self._load_catalog()

    # ---------- Catalog ---------- #

    def _load_catalog(self) -> dict[str, dict[str, dict[str, Any]]]:
        if not self.catalog_path.exists():
            return {}
        with self.catalog_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("ohlcv", {})

    def _save_catalog(self) -> None:
        self.catalog_path.write_text(
            yaml.safe_dump({"ohlcv": self._catalog}, sort_keys=True),
            encoding="utf-8",
        )

    def catalog_entry(self, symbol: str, timeframe: str) -> SeriesMeta | None:
        d = self._catalog.get(symbol, {}).get(timeframe)
        return SeriesMeta.from_dict(symbol, timeframe, d) if d else None

    def _update_catalog(self,
                        symbol: str,
                        timeframe: str,
                        df: pd.DataFrame) -> SeriesMeta | None:
        start = df["timestamp"].min().to_pydatetime()
        end = df["timestamp"].max().to_pydatetime()
        rows = len(df)
        existing = self.catalog_entry(symbol, timeframe)
        if existing:
            start = min(start, existing.start)
            end = max(end, existing.end)
            rows = existing.rows + rows
        meta = SeriesMeta(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            rows=rows,
            last_updated=datetime.now(UTC),
        )
        self._catalog.setdefault(symbol, {})[timeframe] = meta.to_dict()
        self._save_catalog()
        return meta

    # ---------- Write ---------- #

    def write(self,
              df: pd.DataFrame,
              symbol: str,
              timeframe: str
              ) -> SeriesMeta:
        if df.empty:
            raise ValueError("Empty DataFrame.")
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        wdf = df.copy()
        wdf["timestamp"] = pd.to_datetime(wdf["timestamp"], utc=True)  # tz-aware UTC
        wdf.sort_values("timestamp", inplace=True)

        # (Optional) de-dup within batch
        wdf = wdf.drop_duplicates(subset="timestamp", keep="last")

        wdf["symbol"] = symbol
        wdf["timeframe"] = timeframe
        wdf["year"] = wdf["timestamp"].dt.year
        wdf["month"] = wdf["timestamp"].dt.month

        # Define explicit schema so every batch has identical timestamp type
        schema = pa.schema([
            pa.field("timestamp", pa.timestamp("ns", tz="UTC")),
            pa.field("open", pa.float64()),
            pa.field("high", pa.float64()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
            pa.field("volume", pa.float64()),
            pa.field("symbol", pa.string()),
            pa.field("timeframe", pa.string()),
            pa.field("year", pa.int16()),
            pa.field("month", pa.int8()),
        ])

        table = pa.Table.from_pandas(wdf,
                                     preserve_index=False,
                                     schema=schema,
                                     safe=False
                                     )
        pq.write_to_dataset(
            table,
            root_path=str(self.data_root),
            partition_cols=["symbol", "timeframe", "year", "month"],
            existing_data_behavior="overwrite_or_ignore"
        )
        return self._update_catalog(symbol, timeframe, wdf)

    # ---------- Read ---------- #

    def read(
        self,
        symbol: str,
        timeframe: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if not self.data_root.exists():
            return pd.DataFrame()
        dataset = ds.dataset(self.data_root,
                             format="parquet",
                             partitioning="hive"
                             )
        dataset.schema.field("timestamp").type

        if not self.data_root.exists():
            return pd.DataFrame()

        dataset = ds.dataset(self.data_root, format="parquet", partitioning="hive")

        filt = (ds.field("symbol") == symbol) & (ds.field("timeframe") == timeframe)
        if start is not None:
            start_dt = pd.to_datetime(start, utc=True)
            filt &= ds.field("timestamp") >=start_dt.to_pydatetime()
        if end is not None:
            end_dt = pd.to_datetime(end, utc=True)
            filt &= ds.field("timestamp") <= end_dt.to_pydatetime()

        tbl = dataset.to_table(filter=filt,
                               columns=list(columns) if columns else None)
        if tbl.num_rows == 0:
            return pd.DataFrame(columns=columns or ["timestamp", "open", "high", "low", "close", "volume"])
        pdf = tbl.to_pandas()
        if "timestamp" in pdf.columns:
            pdf.sort_values("timestamp", inplace=True)
        return pdf.reset_index(drop=True)

    # --------- Introspection ---------

    def list_series(self) -> list[SeriesMeta]:
        out: list[SeriesMeta] = []
        for sym, tf_map in self._catalog.items():
            for tf, d in tf_map.items():
                out.append(SeriesMeta.from_dict(sym, tf, d))
        return out
