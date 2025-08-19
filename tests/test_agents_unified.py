"""Unified agent interface tests using ToyTradingEnv.

Fast smoke tests ensure each agent can:
1. Instantiate and fit (minimal training) without error.
2. Produce an action within the action space bounds for a sample observation.
3. Save and load (round trip) restoring ability to act.
"""
from __future__ import annotations

# mypy: ignore-errors
# ruff: noqa: E402
import os
import tempfile
from typing import Any

import numpy as np

from trade_agent.agents import (  # type: ignore[import-not-found]
    HybridPolicyAgent,  # type: ignore[import-not-found]
    PPOAgent,  # type: ignore[import-not-found]
    SACAgent,  # type: ignore[import-not-found]
    ToyTradingEnv,  # type: ignore[import-not-found]
)


def make_env(config: dict[str, Any]):  # RLlib / SB3 style callable
    feature_dim = int(config.get("feature_dim", 6))
    seed = config.get("seed", 123)
    return ToyTradingEnv(feature_dim=feature_dim, seed=seed)


def _sample_obs(env: ToyTradingEnv) -> np.ndarray:
    obs, _ = env.reset()
    return obs


def test_sac_agent_round_trip() -> None:
    cfg = {"n_timesteps": 50, "seed": 7}
    agent = SACAgent(cfg, make_env)
    env = agent.env  # type: ignore[attr-defined]
    obs = _sample_obs(env)
    agent.fit()
    act = agent.predict(obs)
    assert act.shape[0] == env.action_space.shape[0]
    assert np.all(act <= env.action_space.high + 1e-6)
    assert np.all(act >= env.action_space.low - 1e-6)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sac_model")
        agent.save(path)
        agent2 = SACAgent(cfg, make_env)
        agent2.load(path)
        act2 = agent2.predict(obs)
        assert act2.shape == act.shape


def test_hybrid_agent_round_trip() -> None:
    cfg = {"hidden_dim": 16, "seed": 11}
    agent = HybridPolicyAgent(cfg, make_env)
    env = agent.env  # type: ignore[attr-defined]
    obs = _sample_obs(env)
    agent.fit()
    act = agent.predict(obs)
    assert act.shape[0] == env.action_space.shape[0]
    assert np.all(act <= env.action_space.high + 1e-6)
    assert np.all(act >= env.action_space.low - 1e-6)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hybrid.pt")
        agent.save(path)
        agent2 = HybridPolicyAgent(cfg, make_env)
        agent2.load(path)
        act2 = agent2.predict(obs)
        assert act2.shape == act.shape


def test_ppo_agent_round_trip() -> None:
    cfg = {"training_iterations": 1, "seed": 5}
    agent = PPOAgent(cfg, make_env)
    env = make_env(cfg)
    obs = _sample_obs(env)
    agent.fit()
    act = agent.predict(obs)
    assert act.shape[0] == env.action_space.shape[0]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ppo_stub.pt")
        agent.save(path)
        agent2 = PPOAgent(cfg, make_env)
        agent2.load(path)
        act2 = agent2.predict(obs)
        assert act2.shape == act.shape
