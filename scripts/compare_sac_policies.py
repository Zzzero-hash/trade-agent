#!/usr/bin/env python3
"""
Script to compare SAC agent's deterministic and stochastic policies
by analyzing action variation.
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

import contextlib

from trade_agent.agents.envs.trading_env import TradingEnvironment  # noqa: E402
from trade_agent.agents.sl.models.base import set_all_seeds  # noqa: E402


def load_sac_model(model_path: str) -> SAC:
    """
    Load the SAC model from the specified path.

    Args:
        model_path: Path to the saved SAC model

    Returns:
        Loaded SAC model
    """
    return SAC.load(model_path)


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
    return TradingEnvironment(
        data_file=val_file,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        seed=seed + 1000,  # Different seed from training but
                     # fixed for evaluation
        window_size=window_size
    )



def collect_actions(
    model: SAC,
    env: TradingEnvironment,
    deterministic: bool = True,
    n_episodes: int = 1
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """
    Collect actions taken by the policy during evaluation.

    Args:
        model: Trained SAC model
        env: Evaluation environment
        deterministic: Whether to use deterministic policy
        n_episodes: Number of episodes to evaluate

    Returns:
        Tuple of (actions, rewards, returns)
    """

    all_actions = []
    all_rewards = []
    all_returns = []

    for _episode in range(n_episodes):

        # Reset environment
        obs, _ = env.reset()

        episode_actions = []
        episode_rewards = []
        total_reward = 0.0
        step_count = 0

        # Run episode until termination
        while True:
            # Get action (deterministic or stochastic)
            action, _ = model.predict(obs, deterministic=deterministic)

            # Store action
            episode_actions.append(action.copy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            total_reward += reward
            step_count += 1

            # Check termination
            if terminated or truncated:
                break

        # Calculate episode return (final equity - initial capital)
        episode_return = info['equity'] - env.initial_capital

        all_actions.extend(episode_actions)
        all_rewards.extend(episode_rewards)
        all_returns.append(episode_return)


    # Clean up temporary file
    with contextlib.suppress(FileNotFoundError):
        os.remove("data/val_temp.parquet")

    return all_actions, all_rewards, all_returns


def compare_policies(model: SAC) -> dict[str, Any]:
    """
    Compare deterministic and stochastic policies by analyzing action variation.

    Args:
        model: Trained SAC model

    Returns:
        Dictionary with comparison metrics
    """

    # Collect actions from deterministic policy
    det_env = create_validation_environment()
    det_actions, det_rewards, det_returns = collect_actions(
        model, det_env, deterministic=True, n_episodes=3)

    # Collect actions from stochastic policy
    sto_env = create_validation_environment(seed=43)  # Different seed
    sto_actions, sto_rewards, sto_returns = collect_actions(
        model, sto_env, deterministic=False, n_episodes=3)

    # Convert to numpy arrays for analysis
    det_actions = np.array(det_actions).flatten()
    sto_actions = np.array(sto_actions).flatten()
    det_rewards = np.array(det_rewards)
    sto_rewards = np.array(sto_rewards)
    det_returns = np.array(det_returns)
    sto_returns = np.array(sto_returns)

    # Calculate metrics
    return {
        'deterministic': {
            'mean_action': float(np.mean(det_actions)),
            'std_action': float(np.std(det_actions)),
            'mean_reward': float(np.mean(det_rewards)),
            'std_reward': float(np.std(det_rewards)),
            'mean_return': float(np.mean(det_returns)),
            'std_return': float(np.std(det_returns)),
            'action_consistency': float(1.0 - np.std(det_actions)),  # Higher means more consistent
        },
        'stochastic': {
            'mean_action': float(np.mean(sto_actions)),
            'std_action': float(np.std(sto_actions)),
            'mean_reward': float(np.mean(sto_rewards)),
            'std_reward': float(np.std(sto_rewards)),
            'mean_return': float(np.mean(sto_returns)),
            'std_return': float(np.std(sto_returns)),
            'action_consistency': float(1.0 - np.std(sto_actions)),  # Higher means more consistent
        },
        'comparison': {
            'action_variation_ratio': float(np.std(sto_actions) / (np.std(det_actions) + 1e-8)),
            'reward_variation_ratio': float(np.std(sto_rewards) / (np.std(det_rewards) + 1e-8)),
            'exploration_level': 'high' if np.std(sto_actions) > np.std(det_actions) * 1.2 else
                               'moderate' if np.std(sto_actions) > np.std(det_actions) * 1.05 else
                               'low'
        }
    }



def save_comparison(metrics: dict[str, Any],
                   output_file: str = "reports/sac_policy_comparison.json") -> None:
    """
    Save policy comparison to a JSON file.

    Args:
        metrics: Comparison metrics
        output_file: Path to output JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            serializable_metrics[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    serializable_metrics[key][sub_key] = sub_value.tolist()
                elif isinstance(sub_value, np.integer | np.floating):
                    serializable_metrics[key][sub_key] = sub_value.item()
                else:
                    serializable_metrics[key][sub_key] = sub_value
        else:
            serializable_metrics[key] = value

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)



def main() -> int:
    """Main comparison function."""

    # Set seeds for reproducibility
    seed = 42
    set_all_seeds(seed)

    # Load configuration
    config_path = "configs/sac_config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}

    # Extract configuration parameters
    sac_config = config.get('sac', {})
    seed = sac_config.get('seed', 42)

    # Model paths to try
    model_paths = [
        "models/rl/sac_final.zip",
        "models/rl/sac.zip",
        "models/rl/best_model.zip"
    ]

    model = None

    # Try to load model
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = load_sac_model(model_path)
                break
            except Exception:
                pass

    if model is None:
        return 1


    # Compare policies
    metrics = compare_policies(model)

    # Print results




    # Analysis
    if metrics['comparison']['exploration_level'] == 'high' or metrics['comparison']['exploration_level'] == 'moderate':
        pass
    else:
        pass

    # Performance check
    det_perf = metrics['deterministic']['mean_return']
    sto_perf = metrics['stochastic']['mean_return']
    perf_diff = sto_perf - det_perf

    if abs(perf_diff) < 50:  # Within $50, consider similar performance
        if metrics['comparison']['exploration_level'] != 'low':
            pass
    elif perf_diff < 0:
        pass
    else:
        pass

    # Save comparison
    save_comparison(metrics, "reports/sac_policy_comparison.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
