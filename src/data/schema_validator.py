"""
Schema validation module for ensuring consistent column counts and structure
across the data pipeline.

This module provides a comprehensive framework for validating data schemas,
ensuring consistent column counts, and managing column transformations
throughout the data processing pipeline.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnSchema:
    """Defines the expected schema for a specific stage in the pipeline."""
    name: str
    required_columns: set[str]
    optional_columns: set[str] = field(default_factory=set)
    expected_dtypes: dict[str, str] = field(default_factory=dict)
    min_columns: Optional[int] = None
    max_columns: Optional[int] = None
    allow_extra_columns: bool = True

    def __post_init__(self):
        """Validate schema configuration after initialization."""
        if self.min_columns is None:
            self.min_columns = len(self.required_columns)
        if self.max_columns is None and not self.allow_extra_columns:
            self.max_columns = (
                len(self.required_columns) + len(self.optional_columns)
            )


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    stage: str
    actual_columns: set[str]
    expected_columns: set[str]
    missing_columns: set[str] = field(default_factory=set)
    extra_columns: set[str] = field(default_factory=set)
    dtype_mismatches: dict[str, tuple] = field(default_factory=dict)
    row_count: int = 0
    column_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SchemaValidator:
    """
    Validates DataFrame schemas throughout the data pipeline to ensure
    consistent column counts and structure.
    """

    def __init__(self):
        """Initialize the schema validator with predefined schemas."""
        self.schemas = self._define_pipeline_schemas()
        self.validation_history: list[ValidationResult] = []

    def _define_pipeline_schemas(self) -> dict[str, ColumnSchema]:
        """Define expected schemas for each stage of the pipeline."""
        return {
            'raw_input': ColumnSchema(
                name='raw_input',
                required_columns={'timestamp', 'value', 'asset'},
                optional_columns={
                    'Symbol', 'Date', 'Close', 'Open', 'High', 'Low', 'Volume'
                },
                expected_dtypes={
                    'timestamp': 'datetime64[ns]',
                    'value': 'float64',
                    'asset': 'object'
                },
                allow_extra_columns=True
            ),

            'cleaned': ColumnSchema(
                name='cleaned',
                required_columns={'timestamp', 'value', 'asset'},
                expected_dtypes={
                    'timestamp': 'datetime64[ns]',
                    'value': 'float64',
                    'asset': 'object'
                },
                min_columns=3,
                allow_extra_columns=True
            ),

            'transformed': ColumnSchema(
                name='transformed',
                required_columns={'timestamp', 'value', 'asset'},
                expected_dtypes={
                    'timestamp': 'datetime64[ns]',
                    'value': 'float64',
                    'asset': 'int64'  # After categorical encoding
                },
                min_columns=3,
                allow_extra_columns=True
            ),

            'feature_engineered': ColumnSchema(
                name='feature_engineered',
                required_columns=set(),  # Variable based on feature selection
                min_columns=5,  # Minimum number of features
                max_columns=100,  # Maximum to prevent feature explosion
                allow_extra_columns=False
            ),

            'final_output': ColumnSchema(
                name='final_output',
                required_columns=set(),  # Will be determined dynamically
                min_columns=5,
                max_columns=50,  # Reasonable limit for model input
                allow_extra_columns=False
            )
        }

    def validate_schema(
        self, df: pd.DataFrame, stage: str
    ) -> ValidationResult:
        """
        Validate DataFrame against expected schema for a given stage.

        Args:
            df: DataFrame to validate
            stage: Pipeline stage name

        Returns:
            ValidationResult with detailed validation information
        """
        if stage not in self.schemas:
            raise ValueError(f"Unknown pipeline stage: {stage}")

        schema = self.schemas[stage]
        actual_columns = set(df.columns)
        expected_columns = schema.required_columns.union(
            schema.optional_columns
        )

        # Initialize result
        result = ValidationResult(
            is_valid=True,
            stage=stage,
            actual_columns=actual_columns,
            expected_columns=expected_columns,
            row_count=len(df),
            column_count=len(df.columns)
        )

        # Check required columns
        missing_required = schema.required_columns - actual_columns
        if missing_required:
            result.missing_columns = missing_required
            result.is_valid = False
            result.errors.append(
                f"Missing required columns: {missing_required}"
            )

        # Check column count constraints
        if schema.min_columns and len(df.columns) < schema.min_columns:
            result.is_valid = False
            result.errors.append(
                f"Too few columns: {len(df.columns)} < {schema.min_columns}"
            )

        if schema.max_columns and len(df.columns) > schema.max_columns:
            result.is_valid = False
            result.errors.append(
                f"Too many columns: {len(df.columns)} > {schema.max_columns}"
            )

        # Check for unexpected columns
        if not schema.allow_extra_columns:
            extra_columns = actual_columns - expected_columns
            if extra_columns:
                result.extra_columns = extra_columns
                result.warnings.append(f"Unexpected columns: {extra_columns}")

        # Check data types
        for col, expected_dtype in schema.expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not self._dtype_compatible(actual_dtype, expected_dtype):
                    result.dtype_mismatches[col] = (
                        actual_dtype, expected_dtype
                    )
                    result.warnings.append(
                        f"Column '{col}' has dtype '{actual_dtype}', "
                        f"expected '{expected_dtype}'"
                    )

        # Store validation result
        self.validation_history.append(result)

        if not result.is_valid:
            logger.error(
                f"Schema validation failed for stage '{stage}': "
                f"{result.errors}"
            )
        elif result.warnings:
            logger.warning(
                f"Schema validation warnings for stage '{stage}': "
                f"{result.warnings}"
            )
        else:
            logger.info(f"Schema validation passed for stage '{stage}'")

        return result

    def _dtype_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual dtype is compatible with expected dtype."""
        # Handle common compatible types
        compatible_types = {
            'float64': ['float32', 'float64', 'int64', 'int32'],
            'int64': ['int32', 'int64'],
            'object': ['object', 'string'],
            'datetime64[ns]': ['datetime64[ns]', 'datetime64[ns, UTC]']
        }

        if expected in compatible_types:
            return actual in compatible_types[expected]

        return actual == expected

    def enforce_column_consistency(
        self,
        df: pd.DataFrame,
        reference_columns: list[str],
        stage: str = "unknown"
    ) -> pd.DataFrame:
        """
        Enforce consistent column structure by adding missing columns
        and reordering to match reference.

        Args:
            df: DataFrame to standardize
            reference_columns: List of expected column names in order
            stage: Pipeline stage for logging

        Returns:
            DataFrame with consistent column structure
        """
        logger.info(f"Enforcing column consistency for stage '{stage}'")

        df_standardized = df.copy()

        # Add missing columns with appropriate default values
        for col in reference_columns:
            if col not in df_standardized.columns:
                logger.warning(
                    f"Adding missing column '{col}' with default values"
                )
                default_val = self._get_default_value(
                    col, len(df_standardized)
                )
                df_standardized[col] = default_val

        # Remove extra columns not in reference (optional)
        extra_cols = set(df_standardized.columns) - set(reference_columns)
        if extra_cols:
            logger.info(f"Removing extra columns: {extra_cols}")
            df_standardized = df_standardized.drop(columns=extra_cols)

        # Reorder columns to match reference
        df_standardized = df_standardized[reference_columns]

        logger.info(
            f"Column consistency enforced: "
            f"{len(df_standardized.columns)} columns, "
            f"{len(df_standardized)} rows"
        )

        return df_standardized

    def _get_default_value(
        self, column_name: str, size: int
    ) -> Union[pd.Series, Any]:
        """Get appropriate default value for a missing column."""
        # Define default strategies based on column name patterns
        name_lower = column_name.lower()

        if 'timestamp' in name_lower or 'date' in name_lower:
            return pd.NaT
        elif any(x in name_lower for x in ['price', 'value', 'volume']):
            return 0.0
        elif any(x in name_lower for x in ['return', 'change', 'pct']):
            return 0.0
        elif 'asset' in name_lower or 'symbol' in name_lower:
            return 'UNKNOWN'
        else:
            # For feature columns, use NaN which can be handled by imputation
            return np.nan

    def create_column_mapping(
        self,
        source_df: pd.DataFrame,
        target_columns: list[str]
    ) -> dict[str, str]:
        """
        Create a mapping from source columns to target columns.
        Useful for handling column name variations across data sources.
        """
        source_cols = list(source_df.columns)
        mapping = {}

        # Common column name mappings
        common_mappings = {
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Symbol': 'asset',
            'Ticker': 'asset',
            'Close': 'value',
            'Price': 'value',
            'Adj Close': 'value'
        }

        # Apply common mappings first
        for source_col in source_cols:
            if source_col in common_mappings:
                target_col = common_mappings[source_col]
                if target_col in target_columns:
                    mapping[source_col] = target_col

        # For remaining columns, try fuzzy matching
        unmapped_source = set(source_cols) - set(mapping.keys())
        unmapped_target = set(target_columns) - set(mapping.values())

        for source_col in unmapped_source:
            best_match = self._find_best_column_match(
                source_col, unmapped_target
            )
            if best_match:
                mapping[source_col] = best_match
                unmapped_target.remove(best_match)

        return mapping

    def _find_best_column_match(
        self, source_col: str, target_cols: set[str]
    ) -> Optional[str]:
        """Find the best matching target column for a source column."""
        source_lower = source_col.lower()

        # Exact match
        for target in target_cols:
            if source_lower == target.lower():
                return target

        # Partial match
        for target in target_cols:
            if source_lower in target.lower() or target.lower() in source_lower:
                return target

        return None

    def get_validation_summary(self) -> dict[str, Any]:
        """Get a summary of all validation results."""
        if not self.validation_history:
            return {"message": "No validations performed yet"}

        summary = {
            "total_validations": len(self.validation_history),
            "successful_validations": sum(
                1 for r in self.validation_history if r.is_valid
            ),
            "failed_validations": sum(
                1 for r in self.validation_history if not r.is_valid
            ),
            "stages_validated": list(
                {r.stage for r in self.validation_history}
            ),
            "common_issues": self._analyze_common_issues()
        }

        return summary

    def _analyze_common_issues(self) -> dict[str, int]:
        """Analyze common validation issues across all validations."""
        issues = {}

        for result in self.validation_history:
            for error in result.errors:
                issues[error] = issues.get(error, 0) + 1
            for warning in result.warnings:
                issues[warning] = issues.get(warning, 0) + 1

        return dict(sorted(issues.items(), key=lambda x: x[1], reverse=True))

    def update_schema(self, stage: str, schema: ColumnSchema) -> None:
        """Update the schema for a specific stage."""
        self.schemas[stage] = schema
        logger.info(f"Updated schema for stage '{stage}'")

    def get_schema(self, stage: str) -> ColumnSchema:
        """Get the schema for a specific stage."""
        if stage not in self.schemas:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        return self.schemas[stage]


# Global validator instance for use across the pipeline
schema_validator = SchemaValidator()


def validate_pipeline_stage(df: pd.DataFrame, stage: str) -> ValidationResult:
    """
    Convenience function to validate a DataFrame at a specific pipeline stage.

    Args:
        df: DataFrame to validate
        stage: Pipeline stage name

    Returns:
        ValidationResult
    """
    return schema_validator.validate_schema(df, stage)


def enforce_consistent_columns(
    df: pd.DataFrame,
    reference_columns: list[str],
    stage: str = "unknown"
) -> pd.DataFrame:
    """
    Convenience function to enforce consistent column structure.

    Args:
        df: DataFrame to standardize
        reference_columns: List of expected column names in order
        stage: Pipeline stage for logging

    Returns:
        DataFrame with consistent column structure
    """
    return schema_validator.enforce_column_consistency(
        df, reference_columns, stage
    )
