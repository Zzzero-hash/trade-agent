"""Common agent interface used across RL and hybrid (SL + RL) agents.

Migrated canonical location under ``trade_agent.agents``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np


class Agent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def fit(self, data: Any) -> None:  # pragma: no cover - interface only
        """Train the agent (data optional for RL agents)."""

    @abstractmethod
    def predict(self, obs: Any) -> Any:  # pragma: no cover - interface only
        """Return an action for a single observation."""

    @abstractmethod
    def save(self, path: str) -> None:  # pragma: no cover - interface only
        """Persist model artefacts to path."""

    @abstractmethod
    def load(self, path: str) -> Any:  # pragma: no cover - interface only
        """Load model artefacts from path."""


class SupportsNumpyObservation(Protocol):  # pragma: no cover - structural type
    def reset(self) -> None:  # noqa: D401 - short protocol
        ...

    def step(self, action: np.ndarray) -> None:  # type: ignore[override]
        ...
