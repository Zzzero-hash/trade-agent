"""Determinism tests for unified set_seed utility.

Ensures that repeated seeding produces identical random draws and
initial environment observations.
"""
from __future__ import annotations

import numpy as np
import torch

from trade_agent.utils import set_seed


def test_set_seed_numpy_torch_reproducible() -> None:
    seed = 123
    set_seed(seed)
    a1 = np.random.rand(5)
    t1 = torch.rand(5)

    set_seed(seed)
    a2 = np.random.rand(5)
    t2 = torch.rand(5)

    assert np.allclose(a1, a2)
    assert torch.allclose(t1, t2)


def test_set_seed_environment_observation() -> None:
    # Lazy import to avoid gym dependency issues if not installed
    try:
        from trade_agent.envs.trading_env import TradingEnvironment
    except Exception:  # pragma: no cover - env optional in some installs
        return

    seed = 321
    # Minimal env instantiation (uses existing sample data file)
    data_file = "data/features.parquet"

    set_seed(seed)
    env1 = TradingEnvironment(data_file=data_file)
    obs1, _ = env1.reset()

    set_seed(seed)
    env2 = TradingEnvironment(data_file=data_file)
    obs2, _ = env2.reset()

    assert obs1.shape == obs2.shape
    # Allow exact match (float arrays seeded deterministically)
    assert np.allclose(obs1, obs2, equal_nan=True)
