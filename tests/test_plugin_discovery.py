from collections.abc import Callable, Iterable
from typing import Any

from trade_agent.plugins import (
    iter_broker_adapters,
    iter_data_collectors,
    iter_rl_agent_factories,
)


def _as_dict(
    iterator: Iterable[tuple[str, Callable[..., Any]]]
) -> dict[str, Callable[..., Any]]:
    return {name: obj for name, obj in iterator}


def test_plugin_iterators_exist() -> None:
    # Should at least return an iterable (may be empty in clean install)
    collectors = _as_dict(iter_data_collectors())
    agents = _as_dict(iter_rl_agent_factories())
    brokers = _as_dict(iter_broker_adapters())
    # Built-in names should exist
    assert "memory" in collectors
    assert "hybrid" in agents
    assert "in_memory" in brokers
