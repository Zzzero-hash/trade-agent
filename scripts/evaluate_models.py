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

from src.data.splits import purged_walk_forward_splits
from src.sl.evaluate import SLEvaluationPipeline
from src.sl.models.factory import SLModelFactory


def load_latest_model(model_type):
    """Load the latest trained model of the specified type."""
    model_files = glob.glob(f'models/sl_model_{model_type}_*.pkl')
    if not model_files:
        print(f"No trained {model_type} model found.")
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
        model = model.load_model(latest_model_path)
        print(f"Loaded {model_type} model from {latest_model_path}")
        return model
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
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
    else:
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
    net_returns = predictions - transaction_costs

    return net_returns


def evaluate_model_performance(model, X_val, y_val, model_name):
    """Evaluate model performance with comprehensive metrics."""
    print(f"\n=== Evaluating {model_name} ===")

    # Make predictions
    try:
        predictions = model.predict(X_val)
    except Exception as e:
        print(f"Error making predictions with {model_name}: {e}")
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
    print(f"Residual mean: {residual_mean:.6f}")

    # Check if residual mean ≈ 0 (within 1% of standard deviation)
    residual_std = np.std(residuals)
    if residual_std > 0:
        normalized_residual_mean = abs(residual_mean) / residual_std
        if normalized_residual_mean < 0.01:  # Less than 1% of std
            print("✓ Residual mean is approximately 0")
            residual_check = True
        else:
            print("✗ Residual mean is not approximately 0")
            residual_check = False
    else:
        print("⚠ Residual standard deviation is 0")
        residual_check = True  # Neutral check if no variation

    # Evaluate model metrics
    config = {'compute_financial_metrics': True}
    pipeline = SLEvaluationPipeline(config)
    results = pipeline.evaluate(y_val_aligned, predictions_aligned)

    # Print key metrics
    print(f"MSE: {results['regression_metrics']['mse']:.6f}")
    print(f"MAE: {results['regression_metrics']['mae']:.6f}")
    print(f"R²: {results['regression_metrics']['r2']:.6f}")
    print(f"Directional Accuracy: {results['financial_metrics']['directional_accuracy']:.2%}")
    print(f"Information Coefficient: {results['financial_metrics']['information_coefficient']:.6f}")

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

    print(f"\n=== Naive Baseline (Mean Return = {mean_return:.6f}) ===")
    print(f"MSE: {results['regression_metrics']['mse']:.6f}")
    print(f"MAE: {results['regression_metrics']['mae']:.6f}")
    print(f"R²: {results['regression_metrics']['r2']:.6f}")

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

    print("\n=== Random Strategy ===")
    print(f"MSE: {results['regression_metrics']['mse']:.6f}")
    print(f"MAE: {results['regression_metrics']['mae']:.6f}")
    print(f"R²: {results['regression_metrics']['r2']:.6f}")
    print(f"Directional Accuracy: {results['financial_metrics']['directional_accuracy']:.2%}")

    return {
        'predictions': random_predictions,
        'metrics': results
    }


def compare_with_baselines(model_results, naive_results, random_results):
    """Compare model performance with baselines."""
    model_name = model_results['model_name']
    model_mse = model_results['metrics']['regression_metrics']['mse']
    model_directional_accuracy = model_results['metrics']['financial_metrics']['directional_accuracy']

    naive_mse = naive_results['metrics']['regression_metrics']['mse']
    naive_directional_accuracy = naive_results['metrics']['financial_metrics']['directional_accuracy']

    random_mse = random_results['metrics']['regression_metrics']['mse']
    random_directional_accuracy = random_results['metrics']['financial_metrics']['directional_accuracy']

    print(f"\n=== Comparison for {model_name} ===")

    # Compare with naive baseline
    if model_mse < naive_mse:
        print("✓ Model beats naive baseline (MSE)")
        naive_check = True
    else:
        print("✗ Model does not beat naive baseline (MSE)")
        naive_check = False

    # Compare directional accuracy with naive
    if model_directional_accuracy > naive_directional_accuracy:
        print("✓ Model beats naive baseline (Directional Accuracy)")
    else:
        print("✗ Model does not beat naive baseline (Directional Accuracy)")

    # Compare with random strategy
    if model_mse < random_mse:
        print("✓ Model beats random strategy (MSE)")
        random_check = True
    else:
        print("✗ Model does not beat random strategy (MSE)")
        random_check = False

    # Compare directional accuracy with random
    if model_directional_accuracy > random_directional_accuracy:
        print("✓ Model beats random strategy (Directional Accuracy)")
    else:
        print("✗ Model does not beat random strategy (Directional Accuracy)")

    return naive_check, random_check


def evaluate_with_transaction_costs(model_results, cost_per_trade=0.001):
    """Evaluate model performance after transaction costs."""
    model_name = model_results['model_name']
    predictions = model_results['predictions']
    targets = model_results['targets']

    # Calculate net returns after transaction costs
    net_returns = calculate_transaction_costs(predictions, cost_per_trade)

    # Evaluate metrics on net returns
    config = {'compute_financial_metrics': True}
    pipeline = SLEvaluationPipeline(config)
    results = pipeline.evaluate(targets, net_returns)

    print(f"\n=== {model_name} After Transaction Costs ({cost_per_trade:.1%}) ===")
    print(f"MSE: {results['regression_metrics']['mse']:.6f}")
    print(f"MAE: {results['regression_metrics']['mae']:.6f}")
    print(f"R²: {results['regression_metrics']['r2']:.6f}")
    print(f"Directional Accuracy: {results['financial_metrics']['directional_accuracy']:.2%}")
    print(f"Information Coefficient: {results['financial_metrics']['information_coefficient']:.6f}")

    # Calculate Sharpe ratio of net returns
    if np.std(net_returns) > 0:
        sharpe_ratio = np.mean(net_returns) / np.std(net_returns)
        print(f"Sharpe Ratio (Net Returns): {sharpe_ratio:.6f}")
    else:
        print("Sharpe Ratio (Net Returns): N/A (zero standard deviation)")

    return results


def main():
    """Main evaluation function."""
    print("Loading data...")
    df = pd.read_parquet('data/features.parquet')

    # Prepare features and targets
    target_columns = ['mu_hat', 'sigma_hat']
    feature_columns = [col for col in df.columns if col not in target_columns]

    X = df[feature_columns].values
    y = df['mu_hat'].values  # We're predicting mean returns

    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Create validation split
    print("Creating validation split...")
    val_data = create_validation_split(df)
    val_indices = val_data.index

    # Align features and targets with validation indices
    X_val = df.loc[val_indices, feature_columns].values
    y_val = df.loc[val_indices, 'mu_hat'].values

    print(f"Validation set size: {len(X_val)}")

    # Define model types to evaluate
    # Only include models that we know can work with the current feature set
    baseline_models = ['ridge', 'garch']  # Skipping linear due to feature mismatch
    deep_models = ['mlp', 'cnn_lstm', 'transformer']

    # Load and evaluate models
    all_results = {}

    # Evaluate naive baseline
    print("\n" + "="*50)
    print("EVALUATING NAIVE BASELINE")
    print("="*50)
    naive_results = evaluate_naive_baseline(y_val)
    all_results['naive'] = naive_results

    # Evaluate random strategy
    print("\n" + "="*50)
    print("EVALUATING RANDOM STRATEGY")
    print("="*50)
    random_results = evaluate_random_strategy(y_val)
    all_results['random'] = random_results

    # Evaluate baseline models
    print("\n" + "="*50)
    print("EVALUATING BASELINE MODELS")
    print("="*50)
    for model_type in baseline_models:
        model = load_latest_model(model_type)
        if model is not None:
            results = evaluate_model_performance(model, X_val, y_val, model_type)
            if results is not None:
                all_results[model_type] = results

    # Evaluate deep models
    print("\n" + "="*50)
    print("EVALUATING DEEP MODELS")
    print("="*50)
    for model_type in deep_models:
        model = load_latest_model(model_type)
        if model is not None:
            results = evaluate_model_performance(model, X_val, y_val, model_type)
            if results is not None:
                all_results[model_type] = results

    # Compare models with baselines
    print("\n" + "="*50)
    print("COMPARISON WITH BASELINES")
    print("="*50)
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
    print("\n" + "="*50)
    print("EVALUATION AFTER TRANSACTION COSTS")
    print("="*50)
    for model_type in evaluated_models:
        if model_type in all_results:
            cost_results = evaluate_with_transaction_costs(all_results[model_type])
            all_results[model_type]['cost_adjusted_metrics'] = cost_results

    # Final assessment
    print("\n" + "="*50)
    print("FINAL ASSESSMENT")
    print("="*50)

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
                print(f"✓ {model_type.upper()} meets all criteria")
                models_meeting_criteria.append(model_type)
            else:
                print(f"✗ {model_type.upper()} does not meet all criteria:")
                if not residual_ok:
                    print("  - Residual mean not ≈ 0")
                if not beats_naive:
                    print("  - Does not beat naive baseline")
                if not beats_random:
                    print("  - Does not beat random strategy")
                models_not_meeting_criteria.append(model_type)

    # If any models don't meet criteria, list likely causes and STOP
    if models_not_meeting_criteria:
        print("\n" + "="*50)
        print("LIKELY CAUSES FOR MODELS NOT MEETING CRITERIA")
        print("="*50)
        print("1. Feature engineering issues:")
        print("   - Features may not be predictive enough")
        print("   - Feature leakage might be present")
        print("   - Features may not be properly normalized")
        print("\n2. Forecast horizon issues:")
        print("   - 5-day horizon might be too short or too long")
        print("   - Market regime changes may affect predictability")
        print("\n3. Model limitations:")
        print("   - Models may be overfitting to training data")
        print("   - Insufficient model complexity for the patterns")
        print("\n4. Data quality issues:")
        print("   - Limited sample size in validation set")
        print("   - Non-stationarity in the data")
        print("\nRECOMMENDATION: STOP and investigate the above issues.")
        return False
    else:
        print("\nAll evaluated models meet the criteria!")
        return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    else:
        print("\nEvaluation completed successfully!")
