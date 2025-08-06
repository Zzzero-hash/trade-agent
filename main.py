import logging
from datetime import datetime, timedelta

from src.data import orchestrator
from src.data.ingestion import fetch_data, get_sp500_symbols


def main():
    """Main entry for the bot with enhanced OHLCV data collection."""
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Fetch S&P 500 symbols
    symbols = get_sp500_symbols()

    # Use recent data for trading (last 2 years + current)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    logger.info(f"Fetching OHLCV data from {start_date} to {end_date}")

    # Fetch comprehensive OHLCV data with daily intervals
    df = fetch_data(
        symbols,
        start_date=start_date,
        end_date=end_date,
        interval='1d'
    )

    # Process the data
    processed_data = orchestrator.orchestrate_data_pipeline(df)
    logger.info("Pipeline completed successfully")
    logger.info(f"Processed data shape: {processed_data.shape}")


if __name__ == "__main__":
    main()
