import json
import pickle

import optuna


def analyze_validation_performance(study_path):
    """
    Extracts validation performance from Optuna study for overfitting analysis.
    """
    print(f"Loading study from {study_path}")

    try:
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
    except Exception as e:
        print(f"Error loading study: {e}")
        return None

    print(f"Study loaded successfully with {len(study.trials)} trials")

    # Get completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Found {len(completed_trials)} completed trials")

    if not completed_trials:
        print("No completed trials found")
        return None

    # Sort by objective value (higher is better for Sharpe ratio)
    completed_trials.sort(key=lambda t: t.value, reverse=True)

    best_trial = completed_trials[0]
    print("\nBest trial validation performance:")
    print(f"  Trial number: {best_trial.number}")
    print(f"  Validation Sharpe ratio: {best_trial.value:.6f}")
    print(f"  Parameters: {best_trial.params}")

    # Get top 10 trials for analysis
    top_10 = completed_trials[:10]
    validation_sharpes = [t.value for t in top_10]

    print("\nTop 10 validation Sharpe ratios:")
    for i, trial in enumerate(top_10, 1):
        print(f"  {i:2d}. Trial {trial.number:3d}: {trial.value:.6f}")

    # Calculate statistics
    import numpy as np
    val_mean = np.mean(validation_sharpes)
    val_std = np.std(validation_sharpes)
    val_median = np.median(validation_sharpes)

    print("\nValidation performance statistics (top 10):")
    print(f"  Mean Sharpe: {val_mean:.6f}")
    print(f"  Std Sharpe:  {val_std:.6f}")
    print(f"  Median:      {val_median:.6f}")
    print(f"  Min:         {min(validation_sharpes):.6f}")
    print(f"  Max:         {max(validation_sharpes):.6f}")

    # Check for overfitting signs
    print("\nOverfitting analysis:")
    if val_std > 0.1:
        print(f"  WARNING: High variance in validation performance (std={val_std:.4f})")

    if best_trial.value > val_mean + 2 * val_std:
        print("  WARNING: Best trial is >2 std above mean - possible overfitting")

    return {
        'best_validation_sharpe': best_trial.value,
        'best_params': best_trial.params,
        'validation_mean': val_mean,
        'validation_std': val_std,
        'validation_median': val_median,
        'top_10_sharpes': validation_sharpes,
        'total_trials': len(study.trials),
        'completed_trials': len(completed_trials)
    }

if __name__ == '__main__':
    study_path = 'reports/reward_params_tuning_20250813_031628_study.pkl'
    results = analyze_validation_performance(study_path)

    # Save results
    if results:
        with open('reports/validation_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to reports/validation_analysis.json")
