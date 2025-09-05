import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from trade_agent.engine.nodes.data_handler import ParquetStore
from trade_agent.plugins.builtins import save_yfinance_data_to_store


router = APIRouter(prefix="/data", tags=["data"])


class DataWriteRequest(BaseModel):
    symbol: str
    timeframe: str
    data: list[dict[str, Any]]  # List of data points as dictionaries


class DataReadRequest(BaseModel):
    symbol: str
    timeframe: str
    start: str | None = None
    end: str | None = None
    columns: list[str] | None = None


class SeriesInfo(BaseModel):
    symbol: str
    timeframe: str
    start: str
    end: str
    rows: int
    last_updated: str


class DataFetchRequest(BaseModel):
    symbol: str
    period: str = "1y"
    interval: str = "1d"
    timeframe: str | None = None  # Will default to interval if not provided


@router.get("/series", response_model=list[SeriesInfo])
async def list_series():
    """List all available data series in the catalog"""
    try:
        store = ParquetStore("data")
        series_list = store.list_series()

        # Convert SeriesMeta objects to SeriesInfo models
        result = []
        for series in series_list:
            series_info = SeriesInfo(
                symbol=series.symbol,
                timeframe=series.timeframe,
                start=series.start.isoformat(),
                end=series.end.isoformat(),
                rows=series.rows,
                last_updated=series.last_updated.isoformat()
            )
            result.append(series_info)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list series: {str(e)}")


@router.post("/write", response_model=SeriesInfo)
async def write_data(request: DataWriteRequest):
    """Write data to the ParquetStore with deduplication"""
    try:
        store = ParquetStore("data")

        # Validate input
        if not request.symbol or not request.timeframe:
            raise HTTPException(status_code=400, detail="Symbol and timeframe are required")

        if not request.data:
            raise HTTPException(status_code=400, detail="Data list cannot be empty")

        # Convert the data list to a pandas DataFrame
        new_df = pd.DataFrame(request.data)

        # Ensure timestamp column exists and is properly formatted
        if 'timestamp' not in new_df.columns:
            raise HTTPException(status_code=400, detail="Missing 'timestamp' column in data")

        # Convert timestamp to datetime with UTC timezone
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], utc=True)

        # Check if we already have data for this symbol/timeframe
        existing_meta = store.catalog_entry(request.symbol, request.timeframe)

        if existing_meta:
            # Read existing data to check for overlaps
            existing_df = store.read(request.symbol, request.timeframe)

            if not existing_df.empty:
                # Combine new and existing data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                # Remove duplicates, keeping the latest data
                combined_df = combined_df.drop_duplicates(subset='timestamp', keep='last')

                # Sort by timestamp
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

                # Write the combined data
                meta = store.write(combined_df, request.symbol, request.timeframe)
            else:
                # No existing data, write new data
                meta = store.write(new_df, request.symbol, request.timeframe)
        else:
            # No existing catalog entry, write new data
            meta = store.write(new_df, request.symbol, request.timeframe)

        # Convert SeriesMeta to SeriesInfo
        return SeriesInfo(
            symbol=meta.symbol,
            timeframe=meta.timeframe,
            start=meta.start.isoformat(),
            end=meta.end.isoformat(),
            rows=meta.rows,
            last_updated=meta.last_updated.isoformat()
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write data: {str(e)}")


@router.post("/read")
async def read_data(request: DataReadRequest):
    """Read data from the ParquetStore with optional filtering"""
    try:
        # Validate input
        if not request.symbol or not request.timeframe:
            raise HTTPException(status_code=400, detail="Symbol and timeframe are required")

        store = ParquetStore("data")

        # Read data from ParquetStore
        df = store.read(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start=request.start,
            end=request.end,
            columns=request.columns
        )

        # Convert DataFrame to list of dictionaries for JSON serialization
        if df.empty:
            return []

        # Convert timestamp to ISO format for JSON serialization
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Convert to list of dictionaries
        return df.to_dict(orient='records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read data: {str(e)}")


@router.post("/upload", response_model=SeriesInfo)
async def upload_csv(
    file: UploadFile = File(...),
    symbol: str = Form(...),
    timeframe: str = Form(...)
):
    """Upload and process CSV file data"""
    try:
        # Validate inputs
        if not symbol or not timeframe:
            raise HTTPException(status_code=400, detail="Symbol and timeframe are required")

        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Parse CSV content
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))

        # Validate required columns
        required_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )

        # Validate data is not empty
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file contains no data")

        # Convert timestamp to datetime with UTC timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Write data to ParquetStore with deduplication
        store = ParquetStore("data")

        # Check if we already have data for this symbol/timeframe
        existing_meta = store.catalog_entry(symbol, timeframe)

        if existing_meta:
            # Read existing data to check for overlaps
            existing_df = store.read(symbol, timeframe)

            if not existing_df.empty:
                # Combine new and existing data
                combined_df = pd.concat([existing_df, df], ignore_index=True)

                # Remove duplicates, keeping the latest data
                combined_df = combined_df.drop_duplicates(subset='timestamp', keep='last')

                # Sort by timestamp
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

                # Write the combined data
                meta = store.write(combined_df, symbol, timeframe)
            else:
                # No existing data, write new data
                meta = store.write(df, symbol, timeframe)
        else:
            # No existing catalog entry, write new data
            meta = store.write(df, symbol, timeframe)

        # Convert SeriesMeta to SeriesInfo
        return SeriesInfo(
            symbol=meta.symbol,
            timeframe=meta.timeframe,
            start=meta.start.isoformat(),
            end=meta.end.isoformat(),
            rows=meta.rows,
            last_updated=meta.last_updated.isoformat()
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")


@router.post("/fetch")
async def fetch_data(request: DataFetchRequest):
    """Smart data fetch endpoint that checks local storage first, then downloads from yfinance if needed"""
    try:
        # Validate input
        if not request.symbol:
            raise HTTPException(status_code=400, detail="Symbol is required")

        # Set timeframe to interval if not provided
        timeframe = request.timeframe or request.interval

        store = ParquetStore("data")

        # Check if we already have data for this symbol/timeframe
        existing_meta = store.catalog_entry(request.symbol, timeframe)

        if existing_meta:
            # We have existing data, read it
            df = store.read(request.symbol, timeframe)

            # Convert timestamp to ISO format for JSON serialization
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

            # Convert to list of dictionaries
            data = df.to_dict(orient='records')

            return {
                "data": data,
                "source": "local",
                "rows": len(data),
                "symbol": request.symbol,
                "timeframe": timeframe,
                "message": "Data loaded from local storage"
            }
        # No existing data, fetch from yfinance
        result = save_yfinance_data_to_store(
            symbol=request.symbol,
            period=request.period,
            interval=request.interval,
            store_path="data"
        )

        if not result.get("success", False):
            error_msg = result.get('error', 'Unknown error')
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch data from yfinance: {error_msg}"
            )

        # Read the newly saved data
        df = store.read(request.symbol, timeframe)

        # Convert timestamp to ISO format for JSON serialization
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Convert to list of dictionaries
        data = df.to_dict(orient='records')

        return {
            "data": data,
            "source": "yfinance",
            "rows": len(data),
            "symbol": request.symbol,
            "timeframe": timeframe,
            "message": "Data fetched from yfinance and saved to local storage"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")
