import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by dropping rows with all NaN values.
    Args:
        df: Input DataFrame to clean.
    Returns:
        Cleaned DataFrame with rows containing all NaN values removed.
    """
    try:
        cleaned_df = df.dropna(axis=0, how='all', inplace=False)
        if cleaned_df.empty:
            raise ValueError(
                "Cleaned DataFrame is empty after dropping NaN rows."
            )
        # Fill NaN values with forward fill method
        cleaned_filled_df = cleaned_df.ffill()
        if cleaned_filled_df.empty:
            raise ValueError("Cleaned and filled DataFrame is empty.")
        # Ensure the index is reset for consistency
        cleaned_filled_df.reset_index(drop=True, inplace=True)
    except Exception as e:
        raise RuntimeError(f"Data cleaning failed: {str(e)}") from e
    return cleaned_filled_df
