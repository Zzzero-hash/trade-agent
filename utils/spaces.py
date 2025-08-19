from collections.abc import Iterable

import gym
import numpy as np


def build_observation_space(feature_dim: int) -> gym.spaces.Box:
    """
    Builds a continuous observation space.

    Args:
        feature_dim: The dimension of the feature space.

    Returns:
        A gym.spaces.Box representing the continuous observation space.
    """
    low = np.full(feature_dim, -np.inf, dtype=np.float32)
    high = np.full(feature_dim, np.inf, dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def build_action_space(action_dim: int) -> gym.spaces.Box:

    """
    Builds a continuous action space.

    Args:
        action_dim: The dimension of the action space.

    Returns:
        A gym.spaces.Box representing the continuous action space.
    """
    low = np.full(action_dim, -1.0, dtype=np.float32)
    high = np.full(action_dim, 1.0, dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def features_to_observation(features: Iterable[float]) -> np.ndarray:
    """Convert iterable of numeric features to float32 numpy array.

    This helper underpins a simple, consistent transformation pipeline for
    agents expecting numpy observations.
    """
    return np.asarray(list(features), dtype=np.float32)
