"""Common agent interface used across RL and hybrid (SL + RL) agents.

All concrete agents must implement:

fit(data): optional training step. For pure RL agents the environment
    usually encapsulates the data; the ``data`` argument is accepted for
    symmetry and may be ignored.
predict(obs): produce an action compatible with the environment action
    space given a single observation (not a batch). Implementations should
    raise ``RuntimeError`` if called before training / loading when a model
    is required.
save(path): persist model artefacts to ``path`` (file or directory depending
    on backend convention).
load(path): restore model artefacts from ``path``.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np


class Agent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def fit(self, data: Any) -> None:  # pragma: no cover - interface only
        """Train the agent.

        Parameters
        ----------
        data: Any
            Optional training data. RL agents may ignore this if their
            environment already wraps required data.
        """

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
    """Protocol for environments supplying numpy observations in tests."""

    def reset(self) -> None:
        ...

    def step(self, action: np.ndarray) -> None:  # type: ignore[override]
        ...
