# ParquetStore

## Layout

Hive-style partitions:

```
data/ohlcv/symbol=SYMBOL/timeframe=TF/year=YYYY/month=MM/part-*.parquet
```

Catalog file: `data/catalog.yaml`.

## API

- write(df, symbol, timeframe) -> SeriesMeta
- read(symbol, timeframe, start=None, end=None, columns=None) -> DataFrame
- catalog_entry(symbol, timeframe) -> SeriesMeta | None
- list_series() -> list[SeriesMeta]

## DataFrame Requirements

Columns: timestamp, open, high, low, close, volume
Timestamp convertible to UTC; others numeric.

## Catalog

YAML structure:

```yaml
ohlcv:
  BTC-USD:
    1m:
      start: 2025-08-27T00:00:00+00:00
      end: 2025-08-27T00:10:00+00:00
      rows: 11
      last_updated: 2025-08-27T12:34:56.789012+00:00
```

## Patterns

Append new candles:

```python
existing_last = store.catalog_entry("BTC-USD", "1m").end
new_df = fetch_api_candles(start=existing_last)
store.write(new_df, "BTC-USD", "1m")
```

## Future Enhacements

- Day partition (add day column).
- Duplicate reconciliation across historical batches.
- Compression tuning (snappy/zstd).
- Metadata validation (expected contiguous minutes).
