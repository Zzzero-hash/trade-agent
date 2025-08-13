import json
import pickle

import numpy as np

# Load the study
study_path = 'reports/reward_params_tuning_20250813_031628_study.pkl'
print(f"Loading study from {study_path}")

try:
    with open(study_path, 'rb') as f:
        study = pickle.load(f)
except Exception as e:
    print(f"Error loading study: {e}")
    exit(1)

print(f"Study loaded successfully with {len(study.trials)} trials")

# Get completed trials
completed_trials = [t for t in study.trials if hasattr(t, 'state') and str(t.state) == 'TrialState.COMPLETE']
print(f"Found {len(completed_trials)} completed trials")

if not completed_trials:
    print("No completed trials found")
    exit(1)

# Sort by objective value (higher is better for Sharpe ratio)
completed_trials.sort(key=lambda t: t.value, reverse=True)

best_trial = completed_trials[0]
print("\nBest trial validation performance:")
print(f"  Trial number: {best_trial.number}")
print(f"  Validation Sharpe ratio: {best_trial.value:.6f}")
print(f"  Parameters: {best_trial.params}")

# Get top 10 trials for analysis
top_10 = completed_trials[:min(10, len(completed_trials))]
validation_sharpes = [t.value for t in top_10]

print(f"\nTop {len(top_10)} validation Sharpe ratios:")
for i, trial in enumerate(top_10, 1):
    print(f"  {i:2d}. Trial {trial.number:3d}: {trial.value:.6f}")

# Calculate statistics
val_mean = np.mean(validation_sharpes)
val_std = np.std(validation_sharpes)
val_median = np.median(validation_sharpes)

print(f"\nValidation performance statistics (top {len(top_10)}):")
print(f"  Mean Sharpe: {val_mean:.6f}")
print(f"  Std Sharpe:  {val_std:.6f}")
print(f"  Median:      {val_median:.6f}")
print(f"  Min:         {min(validation_sharpes):.6f}")
print(f"  Max:         {max(validation_sharpes):.6f}")

# Save results
results = {
    'best_validation_sharpe': float(best_trial.value),
    'best_params': best_trial.params,
    'validation_mean': float(val_mean),
    'validation_std': float(val_std),
    'validation_median': float(val_median),
    'top_10_sharpes': [float(x) for x in validation_sharpes],
    'total_trials': len(study.trials),
    'completed_trials': len(completed_trials)
}

with open('reports/validation_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to reports/validation_analysis.json")
print(f"\nKEY FINDING: Best validation Sharpe ratio was {best_trial.value:.6f}")
