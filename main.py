"""
Enhanced OHLCV Data Collection for Trading Bot
This script fetches OHLCV data for S&P 500 symbols over the last two years
plus the current year, using daily intervals. It leverages Ray for parallel
data fetching and processes the data through a defined pipeline."""

import logging
import os
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import ray

from src.data import orchestrator
from src.data.ingestion import fetch_data


def main():
    """Main entry for the bot with enhanced OHLCV data collection."""
    # Initialize Ray with secure temporary directory configuration
    temp_base_dir = tempfile.gettempdir()
    ray.init(
        object_spilling_directory=os.path.join(temp_base_dir, "ray_spill"),
        _temp_dir=os.path.join(temp_base_dir, "ray_temp")
    )

    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Fetch S&P 500 symbols
    symbols = fetch_data()

    # Use recent data for trading (last 2 years + current)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    logger.info(f"Fetching OHLCV data from {start_date} to {end_date}")

    # Fetch comprehensive OHLCV data with daily intervals
    # Create remote function calls for each symbol
    df_refs = []
    for symbol in symbols:
        df_ref = fetch_data.remote(
            symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        df_refs.append(df_ref)

    # Get all results
    dfs = ray.get(df_refs)

    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Process the data
    processed_data = orchestrator.orchestrate_data_pipeline(df)
    logger.info("Pipeline completed successfully")
    logger.info(f"Processed data shape: {processed_data.shape}")
    logger.info(f"Sample processed data:\n{processed_data.head()}")

    # Cleanup Ray resources
    ray.shutdown()


if __name__ == "__main__":
    print("Select mode: ")
    print("1. Training Mode")
    print("2. Backtesting Mode")
    print("3. Live Trading Mode")
    mode = input("Enter mode (1, 2, or 3): ")
    if mode == "1":
        main()
    elif mode == "2":
        # TODO: Implement Back Testing mode
        pass
    elif mode == "3":
        # TODO: Implement live trading mode
        pass
