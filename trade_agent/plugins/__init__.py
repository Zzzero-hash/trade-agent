"""Plugin system utilities.

Provides dynamic discovery for data collectors, RL agents, and broker adapters
via setuptools entry points.

Entry point groups:
- trade_agent.data_collectors
- trade_agent.rl_agents
- trade_agent.brokers

Each entry point should reference a callable that returns a class or an
instance implementing the expected interface. We keep contracts lightweight:

Data Collector contract (informal):
    collect(**kwargs) -> Iterable[Mapping | DataFrame]

RL Agent factory contract:
    create(config: dict) -> Agent

Broker adapter contract:
    submit(order: Order) -> Any

These are intentionally duck-typed to avoid hard dependencies.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import metadata
from typing import Any, Dict


__all__ = [
    "load_entry_points",
    "iter_data_collectors",
    "iter_rl_agent_factories",
    "iter_broker_adapters",
]

_EP_GROUPS = {
    "data": "trade_agent.data_collectors",
    "rl": "trade_agent.rl_agents",
    "broker": "trade_agent.brokers",
}


def load_entry_points(group: str) -> dict[str, Callable[..., Any]]:
    """Return mapping of entry point name -> loaded object for a group.

    Parameters
    ----------
    group: str
        One of 'data', 'rl', 'broker'.
    """
    if group not in _EP_GROUPS:
        raise ValueError(f"Unknown plugin group: {group}")
    loaded: dict[str, Callable[..., Any]] = {}
    try:
        eps = metadata.entry_points().select(group=_EP_GROUPS[group])
        for ep in eps:  # type: ignore[attr-defined]
            try:
                loaded[ep.name] = ep.load()
            except Exception:  # pragma: no cover
                continue
    except Exception:  # pragma: no cover
        pass

    # Fallback: if no entry points discovered, load built-ins directly
    if not loaded:
        from . import builtins  # noqa: WPS433 (intentional local import)

        if group == "data":
            loaded["memory"] = builtins.simple_memory_collector
        elif group == "rl":
            loaded["hybrid"] = builtins.make_hybrid_agent
        elif group == "broker":
            loaded["in_memory"] = (
                builtins.InMemoryBroker  # type: ignore[assignment]
            )
    return loaded


def _iter(group: str) -> Iterable[tuple[str, Callable[..., Any]]]:
    yield from load_entry_points(group).items()


def iter_data_collectors():
    return _iter("data")


def iter_rl_agent_factories():
    return _iter("rl")


def iter_broker_adapters():
    return _iter("broker")
