import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by handling missing values.

    This function performs initial data cleaning by:
    1. Dropping rows where all values are NaN.
    2. Forward-filling any remaining NaN values.
    Args:
        df: Input DataFrame to clean.
    Returns:
        Cleaned DataFrame with rows containing all NaN values removed.
    """
    try:
        # Drop rows where all elements are missing
        cleaned_df = df.dropna(how='all')

        # Forward-fill remaining missing values
        # This is a preliminary step; more sophisticated gap handling
        # is done in the processing module.
        cleaned_df = cleaned_df.ffill()

        # Reset index to ensure it's clean
        cleaned_df = cleaned_df.reset_index(drop=True)

        if cleaned_df.empty:
            raise ValueError("DataFrame is empty after cleaning.")

        return cleaned_df
    except Exception as e:
        raise RuntimeError(f"Data cleaning failed: {e}") from e
