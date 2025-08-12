#!/usr/bin/env python3
"""
Script to evaluate a random policy in the trading environment.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.trading_env import TradingEnvironment


def test_random_policy():
    """Test a random policy in the trading environment."""
    print("Testing random policy...")

    # Create environment with fixed seed for deterministic results
    env = TradingEnvironment(seed=42)

    # Reset environment
    obs, _ = env.reset()

    # Run episode with random actions
    total_reward = 0.0
    steps = 0

    while True:
        # Take random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Check termination
        if terminated or truncated:
            break

    print(f"Random policy: {steps} steps, total reward: {total_reward:.6f}, final equity: ${info['equity']:.2f}")
    print(f"Return: ${info['equity'] - env.initial_capital:.2f}")

    return total_reward, info['equity']


if __name__ == "__main__":
    test_random_policy()
