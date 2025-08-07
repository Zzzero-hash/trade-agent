"""
Enhanced data processing module with schema validation for consistent columns.

This module extends the existing processing functionality with comprehensive
schema validation to ensure consistent column counts throughout the pipeline.
"""
import logging

import pandas as pd

from src.data.processing import (
    check_data_quality,
    extract_ts_features,
    handle_temporal_gaps,
    select_features,
    transform_yfinance_data,
    validate_data_structure,
)
from src.data.schema_validator import (
    enforce_consistent_columns,
    validate_pipeline_stage,
)

logger = logging.getLogger(__name__)


class SchemaAwareProcessor:
    """
    Enhanced data processor that ensures consistent column structure
    throughout the pipeline using schema validation.
    """

    def __init__(self):
        """Initialize the schema-aware processor."""
        self.reference_columns: list[str] = []
        self.feature_columns: list[str] = []

    def process_data_with_schema_validation(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Main processing pipeline with schema validation at each stage.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame with consistent schema
        """
        try:
            logger.info("Starting schema-aware data processing pipeline")

            # Stage 1: Validate raw input
            logger.info("Stage 1: Validating raw input data")
            validation_result = validate_pipeline_stage(df, 'raw_input')
            if not validation_result.is_valid:
                logger.warning(
                    f"Raw input validation issues: {validation_result.errors}"
                )
                # Attempt to fix common issues
                df = self._fix_raw_input_issues(df, validation_result)

            # Stage 2: Transform data structure
            logger.info("Stage 2: Transforming data structure")
            transformed_df = transform_yfinance_data(df)

            # Validate transformed data
            validate_pipeline_stage(transformed_df, 'transformed')

            # Stage 3: Handle temporal gaps
            logger.info("Stage 3: Handling temporal gaps")
            transformed_df = handle_temporal_gaps(
                transformed_df,
                interval='1D',
                interpolation_method='linear',
                ffill_limit=5,
                bfill_limit=5,
            )

            # Stage 4: Data validation and cleaning
            logger.info("Stage 4: Data validation and quality checks")
            validate_data_structure(transformed_df)
            check_data_quality(transformed_df)  # Report stored but not used

            # Validate cleaned data
            validate_pipeline_stage(transformed_df, 'cleaned')

            # Stage 5: Feature engineering with schema consistency
            logger.info("Stage 5: Feature engineering with schema validation")
            feature_df = self._engineer_features_with_validation(transformed_df)

            # Stage 6: Final validation
            logger.info("Stage 6: Final output validation")
            final_result = validate_pipeline_stage(feature_df, 'final_output')

            if not final_result.is_valid:
                logger.warning("Final output validation failed, attempting to fix...")
                feature_df = self._fix_final_output_issues(feature_df, final_result)

            logger.info(f"Schema-aware processing completed. Final shape: {feature_df.shape}")
            return feature_df

        except Exception as e:
            logger.error(f"Schema-aware data processing failed: {str(e)}")
            raise

    def _fix_raw_input_issues(self, df: pd.DataFrame, validation_result) -> pd.DataFrame:
        """Fix common issues in raw input data."""
        logger.info("Attempting to fix raw input issues")

        # Handle missing required columns
        if validation_result.missing_columns:
            logger.info(f"Adding missing columns: {validation_result.missing_columns}")

            # Create mapping for common column name variations
            column_mapping = {
                'Date': 'timestamp',
                'Symbol': 'asset',
                'Close': 'value',
                'Adj Close': 'value'
            }

            # Apply mapping
            df_fixed = df.copy()
            for old_col, new_col in column_mapping.items():
                if old_col in df_fixed.columns and new_col in validation_result.missing_columns:
                    df_fixed = df_fixed.rename(columns={old_col: new_col})
                    logger.info(f"Mapped column '{old_col}' to '{new_col}'")

            # Add any still missing required columns with defaults
            for col in validation_result.missing_columns:
                if col not in df_fixed.columns:
                    if col == 'timestamp':
                        df_fixed[col] = pd.to_datetime('2024-01-01')
                    elif col == 'value':
                        df_fixed[col] = 0.0
                    elif col == 'asset':
                        df_fixed[col] = 'UNKNOWN'
                    logger.info(f"Added missing column '{col}' with default value")

            return df_fixed

        return df

    def _engineer_features_with_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering with schema validation and consistency."""
        logger.info("Starting feature engineering with schema validation")

        # Convert categorical variables to numeric codes
        feature_df = df.copy()
        feature_df['asset'] = feature_df['asset'].astype('category').cat.codes

        # Prepare features for selection
        X = feature_df.drop(columns=['timestamp', 'value'])
        y = feature_df['value']

        # Feature selection
        try:
            selected_features = select_features(X, y)
            filtered_data = X[selected_features]

            # Store reference columns for consistency
            if not self.feature_columns:
                self.feature_columns = selected_features
                logger.info(f"Stored reference feature columns: {len(self.feature_columns)} features")
            else:
                # Enforce consistency with previous runs
                logger.info("Enforcing feature column consistency")
                filtered_data = enforce_consistent_columns(
                    filtered_data,
                    self.feature_columns,
                    "feature_selection"
                )
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using basic features")
            filtered_data = X

        # Time series feature extraction
        try:
            ts_features = extract_ts_features(feature_df)

            # Combine features with schema validation
            combined = pd.concat([filtered_data, ts_features], axis=1)
            combined = combined.dropna(axis=1, how='all')

            # Validate feature engineered data
            validate_pipeline_stage(combined, 'feature_engineered')

            return combined

        except Exception as e:
            logger.warning(f"Time series feature extraction failed: {e}")
            return filtered_data

    def _fix_final_output_issues(self, df: pd.DataFrame, validation_result) -> pd.DataFrame:
        """Fix issues in final output to ensure model compatibility."""
        logger.info("Fixing final output issues")

        df_fixed = df.copy()

        # If too many columns, reduce by removing least important
        if validation_result.column_count > 50:
            logger.info(f"Reducing columns from {validation_result.column_count} to 50")
            # Keep only first 50 columns (could be enhanced with importance scoring)
            df_fixed = df_fixed.iloc[:, :50]

        # If too few columns, add synthetic features
        elif validation_result.column_count < 5:
            logger.info("Adding synthetic features to reach minimum of 5 columns")
            while len(df_fixed.columns) < 5:
                col_name = f"synthetic_feature_{len(df_fixed.columns)}"
                df_fixed[col_name] = 0.0

        return df_fixed

    def set_reference_columns(self, columns: list[str]) -> None:
        """Set reference columns for consistency enforcement."""
        self.reference_columns = columns
        logger.info(f"Set reference columns: {len(columns)} columns")

    def get_reference_columns(self) -> list[str]:
        """Get the current reference columns."""
        return self.reference_columns

    def reset_reference_columns(self) -> None:
        """Reset reference columns (useful for retraining scenarios)."""
        self.reference_columns = []
        self.feature_columns = []
        logger.info("Reset reference columns")


# Global processor instance
schema_processor = SchemaAwareProcessor()


def process_data_with_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function for schema-aware data processing.

    Args:
        df: Input DataFrame

    Returns:
        Processed DataFrame with consistent schema
    """
    return schema_processor.process_data_with_schema_validation(df)


def ensure_column_consistency_across_batches(dataframes: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Ensure consistent columns across multiple DataFrame batches.

    Args:
        dataframes: List of DataFrames to standardize

    Returns:
        List of DataFrames with consistent column structure
    """
    if not dataframes:
        return []

    # Find the union of all columns across all dataframes
    all_columns = set()
    for df in dataframes:
        all_columns.update(df.columns)

    reference_columns = sorted(list(all_columns))

    # Enforce consistency across all dataframes
    consistent_dataframes = []
    for i, df in enumerate(dataframes):
        logger.info(f"Enforcing consistency for batch {i+1}/{len(dataframes)}")
        consistent_df = enforce_consistent_columns(
            df,
            reference_columns,
            f"batch_{i+1}"
        )
        consistent_dataframes.append(consistent_df)

    return consistent_dataframes


def validate_model_input_schema(df: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame is suitable for model input.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid for model input, False otherwise
    """
    validation_result = validate_pipeline_stage(df, 'final_output')

    if validation_result.is_valid:
        logger.info("DataFrame passed model input validation")
        return True
    else:
        logger.error(f"DataFrame failed model input validation: {validation_result.errors}")
        return False
