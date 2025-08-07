"""
Simple Data Evaluation Model

This module provides a basic framework for evaluating the quality of
financial data and generating metrics and visualizations to represent
data quality as a percentage.
"""

import logging
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class DataEvaluator:
    """Simple data evaluation model for assessing data quality."""

    def __init__(self):
        """Initialize the data evaluator."""
        self.metrics = {}
        plt.style.use('seaborn-v0_8-darkgrid')

    def evaluate_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Evaluate the quality of the input DataFrame.

        Args:
            df: Input DataFrame with financial data

        Returns:
            Dictionary containing quality metrics and overall score
        """
        logger.info("Starting data quality evaluation")

        # Calculate basic metrics
        total_rows = len(df)
        total_columns = len(df.columns)

        # Missing value analysis
        missing_values = df.isnull().sum().sum()
        if total_rows * total_columns > 0:
            missing_percentage = (
                missing_values / (total_rows * total_columns)
            ) * 100
        else:
            missing_percentage = 0

        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        if total_rows > 0:
            duplicate_percentage = (duplicate_rows / total_rows) * 100
        else:
            duplicate_percentage = 0

        # Data type consistency
        numeric_columns = df.select_dtypes(include=['number']).shape[1]
        non_numeric_columns = total_columns - numeric_columns
        if total_columns > 0:
            numeric_percentage = (numeric_columns / total_columns) * 100
        else:
            numeric_percentage = 0

        # Date consistency (assuming a date column exists)
        date_consistency = self._check_date_consistency(df)

        # Calculate overall quality score (0-100%)
        quality_score = self._calculate_quality_score(
            missing_percentage,
            duplicate_percentage,
            date_consistency
        )

        # Store metrics
        self.metrics = {
            'total_rows': total_rows,
            'total_columns': total_columns,
            'missing_values': missing_values,
            'missing_percentage': missing_percentage,
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': duplicate_percentage,
            'numeric_columns': numeric_columns,
            'non_numeric_columns': non_numeric_columns,
            'numeric_percentage': numeric_percentage,
            'date_consistency': date_consistency,
            'quality_score': quality_score
        }

        logger.info(
            "Data quality evaluation completed. "
            f"Quality score: {quality_score:.2f}%"
        )
        return self.metrics

    def _check_date_consistency(self, df: pd.DataFrame) -> float:
        """
        Check date consistency in the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Date consistency score (0-100%)
        """
        # Look for common date column names
        date_columns = [
            col for col in df.columns
            if 'date' in col.lower() or 'time' in col.lower()
        ]

        if not date_columns:
            # If no date columns found, assume perfect score
            return 100.0

        # Check first date column for consistency
        date_col = df[date_columns[0]]

        # Check if it's actually a date column
        if not pd.api.types.is_datetime64_any_dtype(date_col):
            try:
                pd.to_datetime(date_col.iloc[0])
            except Exception:
                return 100.0  # Not a date column, so no penalty

        # Check for chronological order (if it's a time series)
        if len(date_col) > 1:
            try:
                date_series = pd.to_datetime(date_col)
                is_sorted = date_series.is_monotonic_increasing
                # Lower score if not sorted
                return 100.0 if is_sorted else 70.0
            except Exception:
                # Penalty for date parsing issues
                return 50.0

        return 100.0

    def _calculate_quality_score(
        self,
        missing_pct: float,
        duplicate_pct: float,
        date_consistency: float
    ) -> float:
        """
        Calculate overall quality score based on metrics.

        Args:
            missing_pct: Percentage of missing values
            duplicate_pct: Percentage of duplicate rows
            date_consistency: Date consistency score

        Returns:
            Overall quality score (0-100%)
        """
        # Weighted scoring system
        # Higher penalty for missing values
        missing_score = max(0, 100 - missing_pct * 2)
        # Higher penalty for duplicates
        duplicate_score = max(0, 100 - duplicate_pct * 5)
        date_score = date_consistency

        # Calculate weighted average (adjust weights as needed)
        quality_score = (
            missing_score * 0.5 +      # 50% weight for missing values
            duplicate_score * 0.3 +    # 30% weight for duplicates
            date_score * 0.2           # 20% weight for date consistency
        )

        return round(quality_score, 2)

    def generate_quality_report(self) -> str:
        """
        Generate a text report of the data quality metrics.

        Returns:
            Formatted string report
        """
        if not self.metrics:
            return (
                "No evaluation metrics available. "
                "Run evaluate_data_quality first."
            )

        report = f"""
DATA QUALITY REPORT
==================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL QUALITY SCORE: {self.metrics['quality_score']:.2f}%

METRICS:
--------
Total Rows: {self.metrics['total_rows']:,}
Total Columns: {self.metrics['total_columns']}

Missing Values: {self.metrics['missing_values']:,}
                ({self.metrics['missing_percentage']:.2f}%)
Duplicate Rows: {self.metrics['duplicate_rows']}
                ({self.metrics['duplicate_percentage']:.2f}%)

Numeric Columns: {self.metrics['numeric_columns']}
                 ({self.metrics['numeric_percentage']:.2f}%)
Date Consistency: {self.metrics['date_consistency']:.2f}%

RECOMMENDATIONS:
----------------
"""

        # Add recommendations based on scores
        if self.metrics['missing_percentage'] > 5:
            report += (
                "- High percentage of missing values detected. "
                "Consider data imputation or removal of incomplete "
                "records.\n"
            )
        else:
            report += "- Missing values are within acceptable range.\n"

        if self.metrics['duplicate_percentage'] > 1:
            report += (
                "- Significant duplicate records found. "
                "Consider deduplication.\n"
            )
        else:
            report += "- Duplicate records are minimal.\n"

        if self.metrics['date_consistency'] < 90:
            report += (
                "- Date consistency issues detected. "
                "Verify date column integrity.\n"
            )
        else:
            report += "- Date consistency is good.\n"

        return report.strip()

    def plot_quality_metrics(self, save_path: str = None) -> None:
        """
        Generate plots showing data quality metrics.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.metrics:
            logger.warning(
                "No metrics to plot. Run evaluate_data_quality first."
            )
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Data Quality Evaluation Metrics', fontsize=16)

        # 1. Quality Score Gauge
        ax1 = axes[0, 0]
        score = self.metrics['quality_score']
        ax1.barh(
            ['Quality Score'],
            [score],
            color='green' if score > 80 else 'orange' if score > 60 else 'red'
        )
        ax1.set_xlim(0, 100)
        ax1.set_xlabel('Score (%)')
        ax1.set_title('Overall Data Quality Score')
        ax1.text(
            score/2, 0, f'{score:.1f}%', ha='center', va='center',
            color='white' if score > 50 else 'black', fontweight='bold'
        )

        # 2. Missing vs Duplicate Data
        ax2 = axes[0, 1]
        categories = ['Missing', 'Duplicate']
        values = [
            self.metrics['missing_percentage'],
            self.metrics['duplicate_percentage']
        ]
        colors = ['skyblue', 'salmon']
        bars = ax2.bar(categories, values, color=colors)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Data Issues')
        ax2.set_ylim(0, max(10, max(values) * 1.2))

        # Add value labels
        for bar, value in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}%', ha='center', va='bottom'
            )

        # 3. Data Composition
        ax3 = axes[1, 0]
        labels = ['Numeric', 'Non-Numeric']
        sizes = [
            self.metrics['numeric_percentage'],
            100 - self.metrics['numeric_percentage']
        ]
        colors = ['lightblue', 'lightcoral']
        ax3.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90
        )
        ax3.set_title('Column Data Types')

        # 4. Quality Breakdown
        ax4 = axes[1, 1]
        metrics_names = ['Missing', 'Duplicate', 'Date Consistency']
        metrics_values = [
            # Inverse for visualization
            100 - self.metrics['missing_percentage'] * 2,
            # Inverse for visualization
            100 - self.metrics['duplicate_percentage'] * 5,
            self.metrics['date_consistency']
        ]
        bars = ax4.bar(
            metrics_names, metrics_values,
            color=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        ax4.set_ylabel('Score (0-100)')
        ax4.set_title('Quality Component Scores')
        ax4.set_ylim(0, 100)

        # Add value labels
        for bar, value in zip(bars, metrics_values):
            ax4.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom'
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Quality metrics plot saved to {save_path}")

        plt.show()


def evaluate_pipeline_data(
    df: pd.DataFrame,
    save_plots: bool = True
) -> dict[str, Any]:
    """
    Evaluate data quality at the end of the pipeline and generate reports.

    Args:
        df: Processed DataFrame from the pipeline
        save_plots: Whether to save plots to files

    Returns:
        Dictionary containing evaluation results
    """
    evaluator = DataEvaluator()
    metrics = evaluator.evaluate_data_quality(df)

    # Generate and print report
    report = evaluator.generate_quality_report()
    print(report)

    # Generate plots
    if save_plots:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f"data_quality_metrics_{timestamp}.png"
        evaluator.plot_quality_metrics(plot_path)
    else:
        evaluator.plot_quality_metrics()

    return metrics
