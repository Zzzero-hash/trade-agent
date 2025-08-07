import logging

import pandas as pd
import ray

from src.data import cleaning, processing
from src.data.evaluation import evaluate_pipeline_data

logger = logging.getLogger(__name__)


def get_validated_symbols() -> dict[str, pd.DataFrame]:
    """Get validated and corrected symbols with their data."""
    logger.info("Running comprehensive symbol validation and correction")

    # Run validation
    from src.data.symbol_corrector import run_symbol_validation
    validation_data = run_symbol_validation()

    # Get invalid symbols that need correction
    invalid_symbols = validation_data.get('invalid_symbols', {})

    if invalid_symbols:
        logger.info(
            f"Found {len(invalid_symbols)} symbols requiring attention"
        )

        # Get suggestions and apply corrections
        from src.data.symbol_corrector import SymbolCorrector
        corrector = SymbolCorrector()
        suggestions = corrector.suggest_symbol_replacements(invalid_symbols)

        # Apply corrections for symbols with valid suggestions
        corrections = {}
        for symbol, symbol_suggestions in suggestions.items():
            if symbol_suggestions:
                # Use first suggestion
                corrections[symbol] = symbol_suggestions[0]

        if corrections:
            logger.info(f"Applying {len(corrections)} symbol corrections")
            correction_results = corrector.apply_corrections(corrections)

            # Log successful corrections
            applied = correction_results.get('applied_corrections', [])
            for correction in applied:
                logger.info(
                    f"Successfully corrected: {correction['old_symbol']} -> "
                    f"{correction['new_symbol']}"
                )

            # Log failed corrections
            failed = correction_results.get('failed_corrections', [])
            for failure in failed:
                logger.warning(
                    f"Failed to correct: {failure['old_symbol']} -> "
                    f"{failure['new_symbol']} ({failure['error']})"
                )

        # Log symbols that will be skipped (no valid corrections)
        skipped_symbols = []
        for symbol, info in invalid_symbols.items():
            if symbol not in corrections:
                skipped_symbols.append(symbol)
                logger.warning(
                    f"Skipping invalid symbol {symbol} "
                    f"({info['asset_class']}): {info['status']}"
                )

        if skipped_symbols:
            logger.info(
                f"Total symbols skipped due to validation issues: "
                f"{len(skipped_symbols)}"
            )

    # Get validated data for all valid symbols
    from .ingestion import fetch_data_for_symbols
    return fetch_data_for_symbols()


def orchestrate_data_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate the data pipeline from cleaning to processing."""
    try:
        logger.info("Starting data pipeline orchestration")

        # Validate and correct symbols before processing
        validation_data = get_validated_symbols()

        # If we have validated data, use it instead of the input df
        if validation_data:
            logger.info(f"Using validated data for {len(validation_data)} symbols")
            # Convert validated data dict to single DataFrame if needed
            if isinstance(validation_data, dict):
                validated_dfs = []
                for symbol, symbol_df in validation_data.items():
                    if not symbol_df.empty:
                        symbol_df = symbol_df.copy()
                        symbol_df['Symbol'] = symbol
                        validated_dfs.append(symbol_df)

                if validated_dfs:
                    df = pd.concat(validated_dfs, ignore_index=True)
                    logger.info(f"Combined validated data shape: {df.shape}")
                else:
                    logger.warning("No valid data after symbol validation")
                    return pd.DataFrame()

        # Ensure DataFrame is not empty after symbol correction
        if df.empty:
            logger.warning("DataFrame is empty after symbol validation")
            return pd.DataFrame()

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


def validate_and_correct_symbols() -> dict[str, pd.DataFrame]:
    """Validate symbols and return data for validated symbols."""
    from src.data.symbol_corrector import run_symbol_validation
    validation_data = run_symbol_validation()

    # Apply corrections if needed
    if validation_data.get('invalid_symbols'):
        from src.data.symbol_corrector import SymbolCorrector
        corrector = SymbolCorrector()
        suggestions = corrector.suggest_symbol_replacements(validation_data['invalid_symbols'])
        corrections = {s: reps[0] for s, reps in suggestions.items() if reps}
        correction_results = corrector.apply_corrections(corrections)

        # Log correction results
        applied_count = len(correction_results.get('applied_corrections', []))
        logger.info(f"Applied {applied_count} corrections")

        failed_count = len(correction_results.get('failed_corrections', []))
        if failed_count > 0:
            logger.warning(f"Failed to apply {failed_count} corrections")

    # Fetch data for validated symbols
    from .ingestion import fetch_data_for_symbols
    return fetch_data_for_symbols()


def orchestrate_parallel_pipeline(
    data_dict: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Orchestrates a parallelized data processing pipeline.

    Args:
        data_dict: A dictionary where keys are symbols and values are
                   the corresponding DataFrames.

    Returns:
        A single DataFrame with processed and aligned data for all symbols.
    """
    """
    Orchestrates a parallelized data processing pipeline.

    Args:
        data_dict: A dictionary where keys are symbols and values are
                   the corresponding DataFrames.

    Returns:
        A single DataFrame with processed and aligned data for all symbols.
    """
    try:
        logger.info("Starting parallel data pipeline orchestration")

        # Step 1: Parallel data transformation
        logger.info("Starting parallel data transformation")
        transform_tasks = []
        for symbol, df in data_dict.items():
            task = processing.transform_yfinance_data_remote.remote(df)
            transform_tasks.append((symbol, task))

        # Step 2: Collect transformed data
        logger.info("Collecting transformed data")
        transformed_data = {}
        for symbol, task in transform_tasks:
            try:
                result = ray.get(task)
                if not result.empty:
                    transformed_data[symbol] = result
            except Exception as e:
                logger.error(f"Error transforming {symbol}: {str(e)}")

        # Step 3: Parallel feature extraction
        logger.info("Starting parallel feature extraction")
        feature_tasks = []
        for symbol, df in transformed_data.items():
            task = processing.extract_ts_features_remote.remote(df)
            feature_tasks.append((symbol, task))

        # Step 4: Collect features
        logger.info("Collecting extracted features")
        symbol_features = {}
        for symbol, task in feature_tasks:
            try:
                result = ray.get(task)
                if not result.empty:
                    symbol_features[symbol] = result
            except Exception as e:
                logger.error(
                    f"Error extracting features for {symbol}: {str(e)}"
                )

        # Step 5: Align and merge features
        logger.info("Aligning and merging features")
        try:
            final_df = processing.align_symbol_data(
                symbol_features
            )
            logger.info(
                f"Parallel data processing completed. Shape: {final_df.shape}"
            )
        except Exception as e:
            logger.error(
                f"Error aligning symbol data: {str(e)}"
            )
            raise

        # Step 6: Evaluate and return results
        logger.info("Finalizing results")
        try:
            # Here you can add any final processing steps if needed
            evaluated_data = evaluate_pipeline_data(final_df)
            logger.info(
                f"Data evaluation completed. Shape: {evaluated_data.shape}"
            )
            logger.info("Data pipeline orchestration completed successfully")
            return final_df
        except Exception as e:
            logger.error(f"Error finalizing results: {str(e)}")

    except Exception as e:
        logger.error(
            f"Parallel data pipeline failed: {str(e)}"
        )
        raise
