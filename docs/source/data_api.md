# Data API Documentation

The Data API provides endpoints for managing financial time series data using Parquet storage with Hive partitioning.

## Endpoints

### GET /data/series

List all available data series in the catalog.

**Response:**

```json
[
  {
    "symbol": "BTC-USD",
    "timeframe": "1h",
    "start": "2025-08-27T00:00+00:00",
    "end": "2025-08-27T01:00:00+00:00",
    "rows": 2,
    "last_updated": "2025-09-05T18:46:26.407584+00:00"
  }
]
```

### POST /data/write

Write data to the ParquetStore with automatic deduplication.

**Request:**

```json
{
  "symbol": "BTC-USD",
  "timeframe": "1h",
  "data": [
    {
      "timestamp": "2025-08-27T00:00:00Z",
      "open": 50000.0,
      "high": 51000.0,
      "low": 49500.0,
      "close": 50500.0,
      "volume": 1000.0
    }
  ]
}
```

**Response:**

```json
{
  "symbol": "BTC-USD",
  "timeframe": "1h",
  "start": "2025-08-27T00:00:00+00:00",
  "end": "2025-08-27T01:00:00+00:00",
  "rows": 2,
  "last_updated": "2025-09-05T18:46:26.407584+00:00"
}
```

### POST /data/read

Read data from the ParquetStore with optional filtering.

**Request:**

```json
{
  "symbol": "BTC-USD",
  "timeframe": "1h",
  "start": "2025-08-27T00:00:00Z",
  "end": "2025-08-27T01:00:00Z",
  "columns": ["timestamp", "open", "close"]
}
```

**Response:**

```json
[
  {
    "timestamp": "2025-08-27T00:00:00Z",
    "open": 50000.0,
    "close": 50500.0
  }
]
```

### POST /data/upload

Upload and process CSV file data.

**Form Data:**

- file: CSV file with columns (timestamp, open, high, low, close, volume)
- symbol: Trading symbol (e.g., "BTC-USD")
- timeframe: Timeframe (e.g., "1h", "1d")

**Response:**

```json
{
  "symbol": "BTC-USD",
  "timeframe": "1h",
  "start": "2025-08-27T00:00:00+00:00",
  "end": "2025-08-27T01:00:00+00:00",
  "rows": 2,
  "last_updated": "2025-09-05T18:46:26.407584+00:00"
}
```

## Data Storage

Data is stored in Parquet format with Hive partitioning:

```
data/
├── catalog.yaml
└── ohlcv/
    └── symbol=BTC-USD/
        └── timeframe=1h/
            └── year=2025/
                └── month=08/
                    └── part-*.parquet
```

## Features

- **Automatic Deduplication**: Prevents duplicate data storage by checking existing data and merging appropriately
- **Timezone Handling**: Proper UTC timezone handling for all timestamps
- **Flexible Filtering**: Support for time range and column filtering in read operations
- **Data Validation**: Comprehensive validation for all inputs and file formats
- **Error Handling**: Detailed error messages for various failure scenarios

## Example Usage

```bash
# List available series
curl -X GET "http://localhost:8000/data/series"

# Write data
curl -X POST "http://localhost:8000/data/write" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC-USD", "timeframe": "1h", "data": [{"timestamp": "2025-08-27T00:00:00Z", "open": 50000.0, "high": 51000.0, "low": 49500.0, "close": 50500.0, "volume": 1000.0}]}'

# Read data
curl -X POST "http://localhost:8000/data/read" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC-USD", "timeframe": "1h"}'

# Upload CSV
curl -X POST "http://localhost:8000/data/upload" \
  -F "file=@sample_data.csv" \
  -F "symbol=BTC-USD" \
  -F "timeframe=1h"
```
