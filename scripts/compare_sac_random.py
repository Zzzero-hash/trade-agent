#!/usr/bin/env python3
"""
Script to compare SAC agent performance with random policy.
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


def evaluate_deterministic_policy(
    model: SAC,
    env: TradingEnvironment,
    n_episodes: int = 10
) -> dict[str, Any]:
    """
    Evaluate the SAC agent using a deterministic policy.

    Args:
        model: Trained SAC model
        env: Evaluation environment
        n_episodes: Number of episodes to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    print("Starting evaluation with deterministic policy for " +
          f"{n_episodes} episode(s)...")

    episode_returns = []
    episode_rewards = []

    for episode in range(n_episodes):
        print(f"Running episode {episode + 1}/{n_episodes}...")

        # Reset environment
        obs, _ = env.reset()

        total_reward = 0.0
        step_count = 0

        # Run episode until termination
        while True:
            # Get deterministic action
            action, _ = model.predict(obs, deterministic=True)

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
        'mean_reward': mean_reward,
        'mean_return': mean_return,
        'std_return': std_return,
        'episode_returns': episode_returns,
        'episode_rewards': episode_rewards,
        'n_episodes': n_episodes
    }

    return metrics


def evaluate_random_policy(
    env: TradingEnvironment,
    n_episodes: int = 10
) -> dict[str, Any]:
    """
    Evaluate a random policy.

    Args:
        env: Evaluation environment
        n_episodes: Number of episodes to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    print("Starting evaluation with random policy for " +
          f"{n_episodes} episode(s)...")

    episode_returns = []
    episode_rewards = []

    for episode in range(n_episodes):
        print(f"Running episode {episode + 1}/{n_episodes}...")

        # Reset environment
        obs, _ = env.reset()

        total_reward = 0.0
        step_count = 0

        # Run episode until termination
        while True:
            # Take random action
            action = env.action_space.sample()

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

        print(f"  Episode {episode + 1}: Reward = {total_reward:.4f}, " +
              f"Return = ${episode_return:.2f}, Steps = {step_count}")

    # Calculate metrics
    mean_reward = np.mean(episode_rewards)
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)

    metrics = {
        'mean_reward': mean_reward,
        'mean_return': mean_return,
        'std_return': std_return,
        'episode_returns': episode_returns,
        'episode_rewards': episode_rewards,
        'n_episodes': n_episodes
    }

    return metrics


def main():
    """Main comparison function."""
    print("SAC vs Random Policy Comparison")
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

    # Create validation environment
    eval_env = create_validation_environment(
        data_file="data/features.parquet",
        initial_capital=100000.0,
        transaction_cost=0.001,
        window_size=30,
        validation_split=0.2,
        seed=seed
    )

    print("Validation environment created with " +
          f"{len(eval_env.prices)} samples")

    # Evaluate SAC deterministic policy
    sac_metrics = evaluate_deterministic_policy(
        model=model,
        env=eval_env,
        n_episodes=10
    )

    # Create a new environment for random policy evaluation
    random_env = create_validation_environment(
        data_file="data/features.parquet",
        initial_capital=100000.0,
        transaction_cost=0.001,
        window_size=30,
        validation_split=0.2,
        seed=seed
    )

    # Evaluate random policy
    random_metrics = evaluate_random_policy(
        env=random_env,
        n_episodes=10
    )

    # Print comparison results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print("SAC Agent (Deterministic Policy):")
    print(f"  Mean Reward: {sac_metrics['mean_reward']:.6f}")
    print(f"  Mean Return: ${sac_metrics['mean_return']:.2f}")
    print(f"  Return Standard Deviation: ${sac_metrics['std_return']:.2f}")

    print("\nRandom Policy:")
    print(f"  Mean Reward: {random_metrics['mean_reward']:.6f}")
    print(f"  Mean Return: ${random_metrics['mean_return']:.2f}")
    print(f"  Return Standard Deviation: ${random_metrics['std_return']:.2f}")

    # Calculate improvement
    reward_improvement = (sac_metrics['mean_reward'] -
                         random_metrics['mean_reward'])
    return_improvement = (sac_metrics['mean_return'] -
                         random_metrics['mean_return'])

    print("\nImprovement (SAC - Random):")
    print(f"  Reward: {reward_improvement:.6f}")
    print(f"  Return: ${return_improvement:.2f}")

    # Check if SAC is better than random
    if return_improvement > 0:
        print("\n✓ SAC agent outperforms random policy!")
    else:
        print("\n✗ SAC agent does not outperform random policy!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
