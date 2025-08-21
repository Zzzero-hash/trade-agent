"""Backward‑compatibility shim for moved trading environment.

The canonical import path is now ``trade_agent.envs.trading_env``. This
module re-exports ``TradingEnvironment`` to avoid breaking older code using
``trade_agent.agents.envs.trading_env``.
"""
from trade_agent.envs.trading_env import TradingEnvironment  # noqa: F401


__all__ = ["TradingEnvironment"]
