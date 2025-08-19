#!/usr/bin/env python3
"""
Script to evaluate trained models according to specified criteria.
"""
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_agent.agents.sl.evaluate import SLEvaluationPipeline
from trade_agent.agents.sl.models.factory import SLModelFactory
from trade_agent.data.splits import purged_walk_forward_splits


def load_latest_model(model_type):
    """Load the latest trained model of the specified type."""
    model_files = glob.glob(f'models/sl_model_{model_type}_*.pkl')
    if not model_files:
        return None

    # Sort by timestamp to get the latest model
    def extract_timestamp(filename):
        # Extract timestamp from filename
        parts = filename.split('_')
        timestamp_str = parts[-1].split('.')[0]
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            # If parsing fails, return a default old timestamp
            return datetime(2000, 1, 1)

    model_files.sort(key=extract_timestamp)
    latest_model_path = model_files[-1]

    # Load the model
    try:
        model = SLModelFactory.create_model(model_type, {})
        return model.load_model(latest_model_path)
    except Exception:
        return None


def create_validation_split(df, val_ratio=0.2):
    """Create a validation split from the data."""
    # Use the purged walk-forward split to create a validation set
    splits = list(purged_walk_forward_splits(
        df,
        n_splits=1,
        train_ratio=0.0,
        val_ratio=val_ratio,
        test_ratio=1.0-val_ratio
    ))

    if splits:
        _, val_data, _ = splits[0]
        return val_data
    # Fallback: use last portion of data as validation
    val_size = int(len(df) * val_ratio)
    return df.iloc[-val_size:]


def calculate_transaction_costs(predictions, cost_per_trade=0.001):
    """
    Calculate transaction costs based on prediction changes.

    Args:
        predictions: Model predictions (returns)
        cost_per_trade: Cost per trade (default 0.1%)

    Returns:
        np.ndarray: Net returns after transaction costs
    """
    # Convert predictions to positions (1 for positive, -1 for negative)
    positions = np.sign(predictions)

    # Calculate position changes (trades)
    position_changes = np.abs(np.diff(positions, prepend=positions[0]))

    # Transaction costs are applied on position changes
    transaction_costs = position_changes * cost_per_trade

    # Net returns after transaction costs
    return predictions - transaction_costs



def evaluate_model_performance(model, X_val, y_val, model_name):
    """Evaluate model performance with comprehensive metrics."""

    # Make predictions
    try:
        predictions = model.predict(X_val)
    except Exception:
        return None

    # Align predictions and targets if needed (for sequence models)
    if len(predictions) != len(y_val):
        # This is likely a sequence-based model, align the targets
        sequence_length = len(y_val) - len(predictions)
        y_val_aligned = y_val[sequence_length:]
        predictions_aligned = predictions
    else:
        y_val_aligned = y_val
        predictions_aligned = predictions

    # Calculate residuals
    residuals = y_val_aligned - predictions_aligned
    residual_mean = np.mean(residuals)

    # Check if residual mean ≈ 0 (within 1% of standard deviation)
    residual_std = np.std(residuals)
    if residual_std > 0:
        normalized_residual_mean = abs(residual_mean) / residual_std
        if normalized_residual_mean < 0.01:  # Less than 1% of std
            residual_check = True
        else:
            residual_check = False
    else:
        residual_check = True  # Neutral check if no variation

    # Evaluate model metrics
    config = {'compute_financial_metrics': True}
    pipeline = SLEvaluationPipeline(config)
    results = pipeline.evaluate(y_val_aligned, predictions_aligned)

    # Print key metrics

    return {
        'model_name': model_name,
        'predictions': predictions_aligned,
        'targets': y_val_aligned,
        'residuals': residuals,
        'residual_mean': residual_mean,
        'residual_check': residual_check,
        'metrics': results
    }


def evaluate_naive_baseline(y_val):
    """Evaluate naive baseline (mean return)."""
    mean_return = np.mean(y_val)
    naive_predictions = np.full_like(y_val, mean_return)

    # Calculate metrics
    config = {'compute_financial_metrics': True}
    pipeline = SLEvaluationPipeline(config)
    results = pipeline.evaluate(y_val, naive_predictions)


    return {
        'predictions': naive_predictions,
        'metrics': results
    }


def evaluate_random_strategy(y_val):
    """Evaluate random strategy."""
    # Generate random predictions with same mean and std as actual returns
    np.random.seed(42)
    mean_return = np.mean(y_val)
    std_return = np.std(y_val)
    random_predictions = np.random.normal(mean_return, std_return, size=len(y_val))

    # Calculate metrics
    config = {'compute_financial_metrics': True}
    pipeline = SLEvaluationPipeline(config)
    results = pipeline.evaluate(y_val, random_predictions)


    return {
        'predictions': random_predictions,
        'metrics': results
    }


def compare_with_baselines(model_results, naive_results, random_results):
    """Compare model performance with baselines."""
    model_results['model_name']
    model_mse = model_results['metrics']['regression_metrics']['mse']
    model_directional_accuracy = model_results['metrics']['financial_metrics']['directional_accuracy']

    naive_mse = naive_results['metrics']['regression_metrics']['mse']
    naive_directional_accuracy = naive_results['metrics']['financial_metrics']['directional_accuracy']

    random_mse = random_results['metrics']['regression_metrics']['mse']
    random_directional_accuracy = random_results['metrics']['financial_metrics']['directional_accuracy']


    # Compare with naive baseline
    naive_check = model_mse < naive_mse

    # Compare directional accuracy with naive
    if model_directional_accuracy > naive_directional_accuracy:
        pass
    else:
        pass

    # Compare with random strategy
    random_check = model_mse < random_mse

    # Compare directional accuracy with random
    if model_directional_accuracy > random_directional_accuracy:
        pass
    else:
        pass

    return naive_check, random_check


def evaluate_with_transaction_costs(model_results, cost_per_trade=0.001):
    """Evaluate model performance after transaction costs."""
    model_results['model_name']
    predictions = model_results['predictions']
    targets = model_results['targets']

    # Calculate net returns after transaction costs
    net_returns = calculate_transaction_costs(predictions, cost_per_trade)

    # Evaluate metrics on net returns
    config = {'compute_financial_metrics': True}
    pipeline = SLEvaluationPipeline(config)
    results = pipeline.evaluate(targets, net_returns)


    # Calculate Sharpe ratio of net returns
    if np.std(net_returns) > 0:
        np.mean(net_returns) / np.std(net_returns)
    else:
        pass

    return results


def main() -> bool:
    """Main evaluation function."""
    df = pd.read_parquet('data/features.parquet')

    # Prepare features and targets
    target_columns = ['mu_hat', 'sigma_hat']
    feature_columns = [col for col in df.columns if col not in target_columns]

    df[feature_columns].values
    df['mu_hat'].values  # We're predicting mean returns


    # Create validation split
    val_data = create_validation_split(df)
    val_indices = val_data.index

    # Align features and targets with validation indices
    X_val = df.loc[val_indices, feature_columns].values
    y_val = df.loc[val_indices, 'mu_hat'].values


    # Define model types to evaluate
    # Only include models that we know can work with the current feature set
    baseline_models = ['ridge', 'garch']  # Skipping linear due to feature mismatch
    deep_models = ['mlp', 'cnn_lstm', 'transformer']

    # Load and evaluate models
    all_results = {}

    # Evaluate naive baseline
    naive_results = evaluate_naive_baseline(y_val)
    all_results['naive'] = naive_results

    # Evaluate random strategy
    random_results = evaluate_random_strategy(y_val)
    all_results['random'] = random_results

    # Evaluate baseline models
    for model_type in baseline_models:
        model = load_latest_model(model_type)
        if model is not None:
            results = evaluate_model_performance(model, X_val, y_val, model_type)
            if results is not None:
                all_results[model_type] = results

    # Evaluate deep models
    for model_type in deep_models:
        model = load_latest_model(model_type)
        if model is not None:
            results = evaluate_model_performance(model, X_val, y_val, model_type)
            if results is not None:
                all_results[model_type] = results

    # Compare models with baselines
    model_checks = {}
    evaluated_models = [m for m in baseline_models + deep_models if m in all_results]
    for model_type in evaluated_models:
        if model_type in all_results:
            naive_check, random_check = compare_with_baselines(
                all_results[model_type],
                all_results['naive'],
                all_results['random']
            )
            model_checks[model_type] = {
                'naive_check': naive_check,
                'random_check': random_check
            }

    # Evaluate performance after transaction costs
    for model_type in evaluated_models:
        if model_type in all_results:
            cost_results = evaluate_with_transaction_costs(all_results[model_type])
            all_results[model_type]['cost_adjusted_metrics'] = cost_results

    # Final assessment

    models_meeting_criteria = []
    models_not_meeting_criteria = []

    for model_type in evaluated_models:
        if model_type in all_results and model_type in model_checks:
            results = all_results[model_type]
            checks = model_checks[model_type]

            # Check all criteria
            residual_ok = results['residual_check']
            beats_naive = checks['naive_check']
            beats_random = checks['random_check']

            if residual_ok and beats_naive and beats_random:
                models_meeting_criteria.append(model_type)
            else:
                if not residual_ok:
                    pass
                if not beats_naive:
                    pass
                if not beats_random:
                    pass
                models_not_meeting_criteria.append(model_type)

    # If any models don't meet criteria, list likely causes and STOP
    return not models_not_meeting_criteria


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    else:
        pass
