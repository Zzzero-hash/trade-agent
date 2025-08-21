"""Gymnasium space construction helpers (centralised).

Previously lived in top-level ``utils``; migrated under ``trade_agent.utils``.
"""
from __future__ import annotations

from collections.abc import Iterable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


__all__ = [
    "build_observation_space",
    "build_action_space",
    "features_to_observation",
]


def build_observation_space(feature_dim: int) -> gym.spaces.Box:
    """Build a continuous observation space."""
    low = np.full(feature_dim, -np.inf, dtype=np.float32)
    high = np.full(feature_dim, np.inf, dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def build_action_space(action_dim: int) -> gym.spaces.Box:
    """Build a continuous action space."""
    low = np.full(action_dim, -1.0, dtype=np.float32)
    high = np.full(action_dim, 1.0, dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def features_to_observation(features: Iterable[float]) -> NDArray[np.float32]:
    """Convert iterable of numeric features to float32 numpy array."""
    return np.asarray(list(features), dtype=np.float32)
