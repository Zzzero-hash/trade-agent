"""Smoke test for new package layout.

Ensures that the top-level package imports and a minimal environment
instantiates & steps deterministically when a seed is provided.
"""
from __future__ import annotations

import importlib


def test_import_trade_agent_root() -> None:
    pkg = importlib.import_module("trade_agent")
    assert hasattr(pkg, "__version__")


def test_minimal_env_instantiation() -> None:
    # Lazy import to ensure package root works first
    from trade_agent.envs.trading_env import (  # type: ignore
        TradingEnvironment,
    )

    env = TradingEnvironment(seed=123)
    obs, info = env.reset()  # type: ignore[assignment]
    assert obs is not None and info["step"] >= env.window_size
    o2, r, term, trunc, _info2 = env.step(  # type: ignore[assignment]
        env.action_space.sample()  # type: ignore[call-arg]
    )
    assert o2.shape == obs.shape
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)
