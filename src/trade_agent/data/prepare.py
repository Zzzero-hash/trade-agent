"""
Data preparation module for trading RL agent.

Provides functions for processing and standardizing downloaded data
for training and inference.
"""

from pathlib import Path

import pandas as pd
from rich.console import Console

from .data_standardizer import create_standardized_dataset
from .pipeline import DataPipeline

console = Console()


def prepare_data(
    input_path: Path | None = None,
    output_dir: Path = Path("data/processed"),
    config_path: Path | None = None,
    method: str = "robust",
    save_standardizer: bool = True,
    sentiment_data: pd.DataFrame | None = None,
) -> None:
    """
    Process and standardize downloaded data in one command.

    This function combines data processing and standardization into a single step,
    making it easier to prepare data for training and inference.

    Args:
        input_path: Path to input data (file or directory)
        output_dir: Directory to save processed data
        config_path: Path to configuration file
        force_rebuild: Whether to force rebuild of processed data
        parallel: Whether to use parallel processing
        method: Standardization method to use
        save_standardizer: Whether to save the standardizer for later use
        sentiment_data: Optional sentiment features DataFrame to integrate
    """
    # Set default input path if not provided
    if input_path is None:
        input_path = Path("data/raw")

    # Set default config file if not provided
    if config_path is None:
        config_path = Path("config.yaml")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Process the data
    DataPipeline()

    # Load and process data
    if input_path.is_file():
        # Single file
        df = pd.read_csv(input_path)
    elif input_path.is_dir():
        # Directory with multiple files
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {input_path}")

        dfs = []
        for csv_file in csv_files:
            try:
                # Check if file is empty or has no columns
                if csv_file.stat().st_size == 0:
                    print(f"Warning: Skipping empty file: {csv_file}")
                    continue

                df = pd.read_csv(csv_file)

                # Check if DataFrame is empty or has no columns
                if df.empty or len(df.columns) == 0:
                    print(f"Warning: Skipping file with no data: {csv_file}")
                    continue

                df["source_file"] = csv_file.name
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Error reading {csv_file}: {e}")
                continue

        if not dfs:
            raise ValueError(f"No valid CSV files found in {input_path}")

        df = pd.concat(dfs, ignore_index=True)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Step 2: Standardize the data
    # Create standardized dataset
    standardized_df, standardizer = create_standardized_dataset(
        df=df,
        save_path=str(output_dir / "data_standardizer.pkl") if save_standardizer else None
    )

    # Step 2.5: Integrate sentiment data if provided
    if sentiment_data is not None:
        console.print("[yellow]Integrating sentiment features into standardized data...[/yellow]")

        try:
            # Validate sentiment data
            if sentiment_data.empty:
                console.print("[yellow]Warning: Sentiment data is empty, skipping integration[/yellow]")
            elif "symbol" not in sentiment_data.columns:
                console.print("[yellow]Warning: Sentiment data missing 'symbol' column, skipping integration[/yellow]")
            elif "symbol" not in standardized_df.columns:
                console.print("[yellow]Warning: Standardized data missing 'symbol' column, skipping integration[/yellow]")
            else:
                # Merge sentiment data with standardized data
                standardized_df = standardized_df.merge(
                    sentiment_data,
                    on="symbol",
                    how="left",
                    suffixes=("", "_sentiment")
                )

                # Fill missing sentiment values with 0
                sentiment_columns = [col for col in sentiment_data.columns if col != "symbol"]
                for col in sentiment_columns:
                    if col in standardized_df.columns:
                        # Convert to numeric and fill NaN with 0
                        standardized_df[col] = pd.to_numeric(standardized_df[col], errors="coerce").fillna(0.0)

                console.print(f"[green]Integrated {len(sentiment_columns)} sentiment features[/green]")

                # Log summary of sentiment integration
                sentiment_summary = {}
                for col in sentiment_columns:
                    if col in standardized_df.columns:
                        non_zero_count = (standardized_df[col] != 0).sum()
                        total_count = len(standardized_df)
                        sentiment_summary[col] = f"{non_zero_count}/{total_count} non-zero values"

                if sentiment_summary:
                    console.print("[cyan]Sentiment feature summary:[/cyan]")
                    for feature, summary in sentiment_summary.items():
                        console.print(f"[cyan]  {feature}: {summary}[/cyan]")

        except Exception as e:
            console.print(f"[red]Error integrating sentiment data: {e}[/red]")
            console.print("[yellow]Warning: Sentiment integration failed, proceeding without sentiment features[/yellow]")

            # Add default sentiment columns if they don't exist
            default_sentiment_columns = ["sentiment_score", "sentiment_magnitude", "sentiment_sources", "sentiment_direction"]
            for col in default_sentiment_columns:
                if col not in standardized_df.columns:
                    standardized_df[col] = 0.0

    # Step 3: Save processed data
    # Save standardized data
    output_file = output_dir / "standardized_data.csv"
    standardized_df.to_csv(output_file, index=False)

    # Save feature summary
    feature_summary = {
        "total_features": standardizer.get_feature_count(),
        "feature_names": standardizer.get_feature_names(),
        "data_shape": standardized_df.shape,
        "standardization_method": method,
        "missing_value_strategies": standardizer.missing_value_strategies
    }

    import json
    summary_file = output_dir / "feature_summary.json"
    with open(summary_file, "w") as f:
        json.dump(feature_summary, f, indent=2)

    # Step 4: Cleanup raw data for organization
    raw_data_path = Path("data/raw")
    if raw_data_path.exists() and raw_data_path.is_dir():
        try:
            # Count files before deletion for reporting
            files_to_delete = list(raw_data_path.glob("*"))
            files_deleted_count = 0
            dirs_deleted_count = 0

            for item in files_to_delete:
                if item.is_file():
                    item.unlink()
                    files_deleted_count += 1
                elif item.is_dir():
                    import shutil
                    shutil.rmtree(item)
                    dirs_deleted_count += 1

            console.print(f"Cleaned up {files_deleted_count} files and {dirs_deleted_count} directories from {raw_data_path}")

        except Exception as e:
            # Silently continue if cleanup fails
            console.print(f"Could not clean up {raw_data_path}: {e}")
