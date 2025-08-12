#!/usr/bin/env python3
"""
Script to evaluate the SAC agent using both deterministic and stochastic policies
to compare exploration behavior.
"""

import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.trading_env import TradingEnvironment  # noqa: E402
from src.sl.models.base import set_all_seeds  # noqa: E402


def load_sac_model(model_path: str) -> SAC:
    """
    Load the SAC model from the specified path.

    Args:
        model_path: Path to the saved SAC model

    Returns:
        Loaded SAC model
    """
    print(f"Loading SAC model from {model_path}...")
    model = SAC.load(model_path)
    print("Model loaded successfully!")
    return model


def create_validation_environment(
    data_file: str = "data/features.parquet",
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001,
    window_size: int = 30,
    validation_split: float = 0.2,
    seed: int = 42
) -> TradingEnvironment:
    """
    Create validation environment with fixed seed for deterministic evaluation.

    Args:
        data_file: Path to features data file
        initial_capital: Starting capital for portfolio
        transaction_cost: Transaction cost per trade
        window_size: Feature window size
        validation_split: Proportion of data to use for validation
        seed: Random seed for deterministic evaluation

    Returns:
        Validation environment
    """
    # Load data
    df = pd.read_parquet(data_file)

    # Split data to get validation set (same as in training)
    split_index = int(len(df) * (1 - validation_split))
    val_data = df.iloc[split_index:]

    # Save temporary validation file
    val_file = "data/val_temp.parquet"
    val_data.to_parquet(val_file)

    # Create validation environment with fixed seed
    eval_env = TradingEnvironment(
        data_file=val_file,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        seed=seed + 1000,  # Different seed from training but
                     # fixed for evaluation
        window_size=window_size
    )

    return eval_env


def evaluate_policy(
    model: SAC,
    env: TradingEnvironment,
    deterministic: bool = True,
    n_episodes: int = 10
) -> dict[str, Any]:
    """
    Evaluate the SAC agent using either deterministic or stochastic policy.

    Args:
        model: Trained SAC model
        env: Evaluation environment
        deterministic: Whether to use deterministic policy
        n_episodes: Number of episodes to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    policy_type = "deterministic" if deterministic else "stochastic"
    print(f"Starting evaluation with {policy_type} policy for {n_episodes} " +
          "episode(s)...")

    episode_returns = []
    episode_rewards = []
    episode_entropies = []

    for episode in range(n_episodes):
        print(f"Running episode {episode + 1}/{n_episodes}...")

        # Reset environment
        obs, _ = env.reset()

        total_reward = 0.0
        step_count = 0
        entropies = []

        # Run episode until termination
        while True:
            # Get action (deterministic or stochastic)
            if deterministic:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # For stochastic action, we need to get the policy distribution
                import torch
                # Convert observation to tensor if needed
                if not isinstance(obs, torch.Tensor):
                    obs_tensor = torch.as_tensor(obs).float().to(model.device)
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                else:
                    obs_tensor = obs

                # Get the policy distribution and sample action
                with torch.no_grad():
                    distribution = model.policy.get_distribution(obs_tensor)
                    action = distribution.sample()
                    entropy = distribution.entropy().mean().item()
                    entropies.append(entropy)
                    # Convert action back to numpy
                    action = action.cpu().numpy().flatten()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Check termination
            if terminated or truncated:
                break

        # Calculate episode return (final equity - initial capital)
        episode_return = info['equity'] - env.initial_capital
        episode_rewards.append(total_reward)
        episode_returns.append(episode_return)

        # Calculate mean entropy for this episode if we collected any
        if entropies:
            episode_entropies.append(np.mean(entropies))

        print(f"  Episode {episode + 1}: Reward = {total_reward:.4f}, " +
              f"Return = ${episode_return:.2f}, Steps = {step_count}")

    # Clean up temporary file
    try:
        os.remove("data/val_temp.parquet")
    except FileNotFoundError:
        pass

    # Calculate metrics
    mean_reward = np.mean(episode_rewards)
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)

    metrics = {
        'policy_type': policy_type,
        'mean_reward': mean_reward,
        'mean_return': mean_return,
        'std_return': std_return,
        'episode_returns': episode_returns,
        'episode_rewards': episode_rewards,
        'n_episodes': n_episodes
    }

    # Add entropy metrics if we collected them
    if episode_entropies:
        metrics['mean_entropy'] = np.mean(episode_entropies)
        metrics['std_entropy'] = np.std(episode_entropies)
        metrics['episode_entropies'] = episode_entropies

    return metrics


def compare_policies(model: SAC, eval_env: TradingEnvironment) -> tuple[dict, dict]:
    """
    Compare deterministic and stochastic policies.

    Args:
        model: Trained SAC model
        eval_env: Evaluation environment

    Returns:
        Tuple of (deterministic_metrics, stochastic_metrics)
    """
    print("Comparing deterministic vs stochastic policies...")
    print("=" * 50)

    # Evaluate deterministic policy
    det_env = create_validation_environment()
    deterministic_metrics = evaluate_policy(model, det_env, deterministic=True, n_episodes=5)

    # Evaluate stochastic policy
    sto_env = create_validation_environment(seed=43)  # Different seed for variety
    stochastic_metrics = evaluate_policy(model, sto_env, deterministic=False, n_episodes=5)

    return deterministic_metrics, stochastic_metrics


def save_comparison(deterministic_metrics: dict, stochastic_metrics: dict,
                   output_file: str = "reports/sac_policy_comparison.json"):
    """
    Save policy comparison to a JSON file.

    Args:
        deterministic_metrics: Metrics from deterministic policy
        stochastic_metrics: Metrics from stochastic policy
        output_file: Path to output JSON file
    """
    # Combine metrics
    comparison = {
        'deterministic': deterministic_metrics,
        'stochastic': stochastic_metrics,
        'comparison': {
            'entropy_difference': (stochastic_metrics.get('mean_entropy', 0) -
                                 deterministic_metrics.get('mean_entropy', 0)),
            'return_difference': (stochastic_metrics['mean_return'] -
                                deterministic_metrics['mean_return']),
            'reward_difference': (stochastic_metrics['mean_reward'] -
                                deterministic_metrics['mean_reward'])
        }
    }

    # Convert numpy arrays to lists for JSON serialization
    serializable_comparison = {}
    for key, value in comparison.items():
        if isinstance(value, dict):
            serializable_comparison[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    serializable_comparison[key][sub_key] = sub_value.tolist()
                elif isinstance(sub_value, (np.integer, np.floating)):
                    serializable_comparison[key][sub_key] = sub_value.item()
                else:
                    serializable_comparison[key][sub_key] = sub_value
        else:
            serializable_comparison[key] = value

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(serializable_comparison, f, indent=2)

    print(f"Comparison saved to {output_file}")


def main():
    """Main evaluation function."""
    print("SAC Agent Policy Comparison Script")
    print("=" * 50)

    # Set seeds for reproducibility
    seed = 42
    set_all_seeds(seed)

    # Load configuration
    config_path = "configs/sac_config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found, using defaults.")
        config = {}

    # Extract configuration parameters
    config.get('training', {})
    sac_config = config.get('sac', {})
    seed = sac_config.get('seed', 42)

    # Model paths to try
    model_paths = [
        "models/rl/sac_final.zip",
        "models/rl/sac.zip",
        "models/rl/best_model.zip"
    ]

    model = None
    loaded_model_path = None

    # Try to load model
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = load_sac_model(model_path)
                loaded_model_path = model_path
                break
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")

    if model is None:
        print("No trained SAC model found. Please train a model first.")
        return 1

    print(f"Using model from: {loaded_model_path}")

    # Compare policies
    deterministic_metrics, stochastic_metrics = compare_policies(model,
                                                               create_validation_environment())

    # Print results
    print("\n" + "=" * 50)
    print("POLICY COMPARISON RESULTS")
    print("=" * 50)

    print("\nDeterministic Policy:")
    print(f"  Mean Reward: {deterministic_metrics['mean_reward']:.6f}")
    print(f"  Mean Return: ${deterministic_metrics['mean_return']:.2f}")
    print(f"  Return Standard Deviation: ${deterministic_metrics['std_return']:.2f}")

    print("\nStochastic Policy:")
    print(f"  Mean Reward: {stochastic_metrics['mean_reward']:.6f}")
    print(f"  Mean Return: ${stochastic_metrics['mean_return']:.2f}")
    print(f"  Return Standard Deviation: ${stochastic_metrics['std_return']:.2f}")
    if 'mean_entropy' in stochastic_metrics:
        print(f"  Mean Policy Entropy: {stochastic_metrics['mean_entropy']:.6f}")
        print(f"  Entropy Standard Deviation: {stochastic_metrics['std_entropy']:.6f}")

    # Check if policy is more exploratory
    if 'mean_entropy' in stochastic_metrics:
        entropy_diff = stochastic_metrics['mean_entropy']
        print("\nPolicy Analysis:")
        if entropy_diff > 0.5:
            print(f"  - Policy shows high exploration (entropy: {entropy_diff:.4f})")
        elif entropy_diff > 0.1:
            print(f"  - Policy shows moderate exploration (entropy: {entropy_diff:.4f})")
        else:
            print(f"  - Policy shows low exploration (entropy: {entropy_diff:.4f})")

    # Save comparison
    save_comparison(deterministic_metrics, stochastic_metrics,
                   "reports/sac_policy_comparison.json")

    print("\nComparison completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
