trade_agent layout:
trade_agent/
**init**.py (existing)
engine/ (existing core execution area)
**init**.py
pipeline.py (existing: pipeline orchestration)
nodes/ (existing container for node types)
**init**.py
data_source.py (planned: market data fetch node)
sma.py (planned: SMA transform)
crossover.py (planned: SMA crossover signal)
(future) features/ (more indicator/feature nodes)
(future) execution.py (higher-level execute helpers)
backtest/ (existing namespace reserved)
**init**.py
runner.py (planned: SimpleLongOnlyBacktester + BacktestResult)
(future) metrics.py (planned: separated performance metrics)
telemetry/
**init**.py
core.py (planned: opt-in event recorder, env flag)
flags/
**init**.py
store.py (existing: feature flag store)
api/
**init**.py
app.py
routes/
**init**.py
health.py
pipelines.py
persistence/
**init**.py
repository.py
sqlite.py
utils/
**init**.py
logging.py
scripts/
verify_release.py (existing release gate script)
tests/
**init**.py
test_engine_pipeline.py (existing baseline)
test_backtest_runner.py (planned)
test_api_health.py (planned)
test_version_bump.py (existing)
test_flags.py (existing)
core/
**init**.py
test_engine_pipeline.py (planned)
test_metrics.py (planned)
test_telemetry_opt_in.py (planned)

## Data Storage (OHLCV ParquetStore)

The project persists OHLCV candles in partitioned Parquet (Hive) layout:

```
data/ohlcv/
    symbol=BTC-USD/timeframe=1m/year=2025/month=08/part-*.parquet
data/catalog.yaml
```

Catalog tracks per (symbol, timeframe):

- start/end (ISO UTC)
- cumulative row count
- last_updated

Basic usage:

```python
from trade_agent.engine.nodes.data_handler import ParquetStore
import pandas as pd

store = ParquetStore("data")
df = pd.DataFrame({
    "timestamp": pd.date_range("2025-08-27", periods=3, freq="T", tz="UTC"),
    "open":[1.0,2.0,3.0],
    "high":[1.1,2.1,3.1],
    "low":[0.9,1.9,2.9],
    "close":[1.05,2.05,3.05],
    "volume":[100.0,110.0,120.0],
})
store.write(df, "BTC-USD", "1m")
subset = store.read("BTC-USD", "1m", end="2025-08-27T00:01:00Z")
print(subset)
print(store.list_series())
```

Run the unit test:

```bash
pytest -k parquet_store -q
```

### Design Notes

- Timestamps stored as UTC (timestamp[ns, tz=UTC]).
- Partitioned by symbol, timeframe, year, month.
- Batch write de-duplicates timestamps within the batch (keep=last).
- Range filtering pushed down via pyarrow.dataset.
