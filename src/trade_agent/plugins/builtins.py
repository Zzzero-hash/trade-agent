"""Built-in (always available) plugin implementations.

These register minimal reference implementations so discovery yields at least
one example even without external packages.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from trade_agent.agents.hybrid_policy import HybridPolicyAgent
from trade_agent.envs.trading_env import TradingEnvironment


# Data Collector example ----------------------------------------------------

def simple_memory_collector(**kwargs: Any) -> Iterable[Mapping[str, Any]]:
    """Yield a tiny in-memory data stream (placeholder)."""
    data: list[dict[str, Any]] = [
        {"symbol": "TEST", "price": 100.0},
        {"symbol": "TEST", "price": 100.5},
    ]
    yield from data


# RL Agent factory example --------------------------------------------------

def make_hybrid_agent(config: dict[str, Any]) -> Any:
    return HybridPolicyAgent(config, lambda cfg: TradingEnvironment())


# Broker adapter example ----------------------------------------------------
class InMemoryBroker:
    def __init__(self) -> None:
        self.orders: list[Mapping[str, Any]] = []

    def submit(self, order: Mapping[str, Any]) -> Mapping[str, Any]:
        self.orders.append(order)
        return {"status": "accepted", **order}

    def close(self) -> None:  # pragma: no cover
        self.orders.clear()


__all__ = [
    "simple_memory_collector",
    "make_hybrid_agent",
    "InMemoryBroker",
]
