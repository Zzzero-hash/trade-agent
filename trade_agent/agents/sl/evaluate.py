"""
Evaluation procedures for supervised learning models.
"""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


class SLEvaluationMetrics:
    """Comprehensive evaluation metrics for supervised learning models."""

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Compute regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {}

        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

        # Mean Absolute Percentage Error (if no zero values)
        if np.all(y_true != 0):
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        else:
            metrics['mape'] = np.nan

        # Mean Absolute Scaled Error (MASE)
        metrics['mase'] = SLEvaluationMetrics._compute_mase(y_true, y_pred)

        return metrics

    @staticmethod
    def _compute_mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Absolute Scaled Error.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            float: MASE value
        """
        # Naive forecast (shifted by one period)
        y_naive = np.roll(y_true, 1)
        y_naive[0] = y_true[0]  # First value remains the same

        # Mean absolute error of predictions
        mae_pred = np.mean(np.abs(y_true - y_pred))

        # Mean absolute error of naive forecast
        mae_naive = np.mean(np.abs(y_true[1:] - y_naive[1:]))

        # MASE
        if mae_naive == 0:
            return np.nan
        return mae_pred / mae_naive

    @staticmethod
    def financial_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Compute financial-specific metrics.

        Args:
            y_true: True target values (returns)
            y_pred: Predicted values (returns)

        Returns:
            Dict[str, float]: Dictionary of financial metrics
        """
        metrics = {}

        # Directional accuracy
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        metrics['directional_accuracy'] = np.mean(true_direction == pred_direction)

        # Sharpe ratio of predictions (assuming risk-free rate = 0)
        pred_returns = y_pred
        if np.std(pred_returns) > 0:
            metrics['sharpe_ratio'] = np.mean(pred_returns) / np.std(pred_returns)
        else:
            metrics['sharpe_ratio'] = 0

        # Information Coefficient (IC)
        metrics['information_coefficient'] = np.corrcoef(y_true, y_pred)[0, 1]

        # Maximum Drawdown
        cumulative_returns = np.cumsum(pred_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        metrics['max_drawdown'] = np.min(drawdown)

        return metrics

    @staticmethod
    def distributional_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             quantiles: list[float] = None) -> dict[str, float]:
        """
        Compute distributional metrics for quantile predictions.

        Args:
            y_true: True target values
            y_pred: Predicted values
            quantiles: Quantiles to evaluate

        Returns:
            Dict[str, float]: Dictionary of distributional metrics
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        metrics = {}

        # Pinball loss for quantile predictions
        # Note: This assumes y_pred contains quantile predictions
        # For point predictions, this is just a special case
        for q in quantiles:
            pinball_loss = SLEvaluationMetrics._pinball_loss(y_true, y_pred, q)
            metrics[f'pinball_loss_q{q}'] = pinball_loss

        # Continuous Ranked Probability Score (CRPS)
        metrics['crps'] = SLEvaluationMetrics._compute_crps(y_true, y_pred)

        return metrics

    @staticmethod
    def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """
        Compute pinball loss for a specific quantile.

        Args:
            y_true: True target values
            y_pred: Predicted quantile values
            quantile: Quantile level (0 < quantile < 1)

        Returns:
            float: Pinball loss
        """
        errors = y_true - y_pred
        loss = np.maximum(quantile * errors, (quantile - 1) * errors)
        return np.mean(loss)

    @staticmethod
    def _compute_crps(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Continuous Ranked Probability Score.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            float: CRPS value
        """
        # Simplified CRPS for point predictions
        # In practice, this would require probabilistic predictions
        return np.mean(np.abs(y_true - y_pred))


class SLEvaluationPipeline:
    """Supervised learning evaluation pipeline."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize evaluation pipeline.

        Args:
            config (Dict[str, Any]): Configuration for the pipeline
        """
        self.config = config
        self.metrics_calculator = SLEvaluationMetrics()

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                y_pred_proba = None,
                y_pred_quantiles = None) -> dict[str, Any]:
        """
        Evaluate model predictions.

        Args:
            y_true: True target values
            y_pred: Point predictions
            y_pred_proba: Probability predictions (optional)
            y_pred_quantiles: Quantile predictions (optional)

        Returns:
            Dict[str, Any]: Evaluation results
        """
        results = {}

        # Basic regression metrics
        results['regression_metrics'] = self.metrics_calculator.regression_metrics(y_true, y_pred)

        # Financial metrics (if applicable)
        if self.config.get('compute_financial_metrics', True):
            results['financial_metrics'] = self.metrics_calculator.financial_metrics(y_true, y_pred)

        # Distributional metrics (if quantile predictions provided)
        if y_pred_quantiles is not None:
            # For simplicity, we'll use the median prediction
            median_pred = y_pred_quantiles.get(0.5, y_pred)
            results['distributional_metrics'] = self.metrics_calculator.distributional_metrics(
                y_true, median_pred
            )

        return results

    def generate_report(self, evaluation_results: dict[str, Any]) -> str:
        """
        Generate evaluation report.

        Args:
            evaluation_results: Results from evaluate method

        Returns:
            str: Formatted report
        """
        report = []
        report.append("=== Supervised Learning Model Evaluation Report ===\n")

        # Regression metrics
        if 'regression_metrics' in evaluation_results:
            report.append("Regression Metrics:")
            for metric, value in evaluation_results['regression_metrics'].items():
                report.append(f"  {metric.upper()}: {value:.6f}")
            report.append("")

        # Financial metrics
        if 'financial_metrics' in evaluation_results:
            report.append("Financial Metrics:")
            for metric, value in evaluation_results['financial_metrics'].items():
                if metric == 'directional_accuracy':
                    report.append(f"  {metric.replace('_', ' ').title()}: {value:.2%}")
                else:
                    report.append(f"  {metric.replace('_', ' ').title()}: {value:.6f}")
            report.append("")

        # Distributional metrics
        if 'distributional_metrics' in evaluation_results:
            report.append("Distributional Metrics:")
            for metric, value in evaluation_results['distributional_metrics'].items():
                report.append(f"  {metric.replace('_', ' ').title()}: {value:.6f}")
            report.append("")

        return "\n".join(report)


def evaluate_model_from_predictions(y_true: np.ndarray | pd.Series,
                                  y_pred: np.ndarray | pd.Series,
                                  config: dict[str, Any] = None) -> dict[str, Any]:
    """
    Evaluate a model using predictions.

    Args:
        y_true: True target values
        y_pred: Predicted values
        config: Configuration for evaluation

    Returns:
        Dict[str, Any]: Evaluation results
    """
    if config is None:
        config = {'compute_financial_metrics': True}

    # Convert to numpy arrays if pandas
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Create evaluation pipeline
    pipeline = SLEvaluationPipeline(config)

    # Evaluate
    results = pipeline.evaluate(y_true, y_pred)

    # Generate and print report
    pipeline.generate_report(results)

    return results
