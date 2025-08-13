import pickle

import optuna
import pandas as pd

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def analyze_study(study_path):
    """
    Analyzes an Optuna study from a .pkl file.

    Args:
        study_path (str): The path to the .pkl file containing the study.
    """
    print(f"Analyzing study from {study_path}\n")

    # Load the study
    try:
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    print("Study loaded successfully.")
    print(f"Number of trials: {len(study.trials)}")
    print("Best trial so far:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

    # Get all trials into a pandas DataFrame
    trials_df = study.trials_dataframe()

    # Sort by objective value in descending order (assuming higher is better)
    trials_df = trials_df.sort_values(by='value', ascending=False)

    print("\n--- Top 10 Trials ---")
    print(trials_df.head(10))

    print("\n--- Analysis of Top Trials ---")
    top_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    top_trials = sorted(top_trials, key=lambda t: t.value, reverse=True)

    if not top_trials:
        print("No completed trials found.")
        return

    best_value = top_trials[0].value
    print(f"Best objective value: {best_value:.6f}")

    # Check for robust optima
    similar_trials = [t for t in top_trials[:10] if abs(t.value - best_value) < 0.01 * abs(best_value)]
    if len(similar_trials) > 1:
        print(f"\nFound {len(similar_trials)} trials with very similar objective values (within 1% of the best). This suggests a robust optimization landscape.")
        for i, trial in enumerate(similar_trials):
            print(f"  Trial {trial.number}: Value={trial.value:.6f}, Params={trial.params}")
    else:
        print("\nThe best trial is significantly better than others. This might indicate a 'razor-thin' optimum.")
        if len(top_trials) > 1:
            second_best_value = top_trials[1].value
            print(f"The difference between the best and second-best trial is {best_value - second_best_value:.6f}")


    print("\n--- Parameter Sensibility Assessment ---")
    best_params = study.best_trial.params
    print("Assessing the sensibility of the best parameters:")
    # These checks are examples and should be tailored to the specific meaning of the hyperparameters
    if 'trade_penalty' in best_params and best_params['trade_penalty'] > 0.1:
        print("- The 'trade_penalty' is relatively high, suggesting the model is being discouraged from frequent trading.")
    if 'holding_penalty' in best_params and best_params['holding_penalty'] > 0.01:
        print("- The 'holding_penalty' is also significant, indicating a cost for holding positions.")
    if 'stop_loss_penalty' in best_params and best_params['stop_loss_penalty'] > 1.0:
        print("- The 'stop_loss_penalty' is very high, heavily penalizing the model for hitting stop-loss limits.")

    print("\n--- Best Hyperparameters ---")
    print(best_params)

if __name__ == '__main__':
    STUDY_PATH = 'reports/reward_params_tuning_20250813_031628_study.pkl'
    analyze_study(STUDY_PATH)
