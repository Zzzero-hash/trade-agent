"""Lightweight protocol/ABC definitions for plugin contracts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Protocol


class DataCollector(Protocol):  # pragma: no cover - structural typing only
    def collect(self, **kwargs: Any) -> Iterable[Mapping[str, Any]]: ...


class RLAgentFactory(Protocol):  # pragma: no cover
    def __call__(self, config: dict[str, Any]) -> Any: ...


class BrokerAdapter(ABC):  # pragma: no cover
    @abstractmethod
    def submit(self, order: Mapping[str, Any]) -> Any: ...

    def close(self) -> None:  # optional hook
        return None


__all__ = ["DataCollector", "RLAgentFactory", "BrokerAdapter"]
