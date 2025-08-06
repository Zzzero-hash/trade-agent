import logging

import pandas as pd

from . import cleaning, processing

logger = logging.getLogger(__name__)


def orchestrate_data_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate the data pipeline from cleaning to processing."""
    try:
        logger.info("Starting data pipeline orchestration")

        # Step 1: Clean data (df is already fetched from main.py)
        logger.info("Starting data cleaning")
        cleaned_data = cleaning.clean_data(df)
        logger.info(f"Data cleaning completed. Shape: {cleaned_data.shape}")

        # Step 2: Process data
        logger.info("Starting data processing")
        processed_data = processing.process_data(cleaned_data)
        logger.info(
            f"Data processing completed. Shape: {processed_data.shape}"
        )

        return processed_data

    except Exception as e:
        logger.error(f"Data pipeline failed: {str(e)}")
        raise
