"""Base classes and utilities for supervised learning models.

Minimal surface to satisfy tests in `tests/test_sl_models.py`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
import pickle
import random
from typing import Any, ClassVar, Dict

import numpy as np

try:  # Optional torch dependency for PyTorchSLModel
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    nn = object  # type: ignore


def set_all_seeds(seed: int) -> None:
    """Set PRNG seeds for Python, NumPy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None and hasattr(torch, 'manual_seed'):
        torch.manual_seed(seed)  # type: ignore
        if hasattr(torch.cuda, 'manual_seed_all'):
            torch.cuda.manual_seed_all(seed)  # type: ignore


@dataclass
class SLBaseModel(ABC):
    """Abstract base supervised learning model.

    Parameters
    ----------
    config: dict
        Configuration dictionary. Must include 'random_state' when determinism desired.
    """

    config: Dict[str, Any]
    is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover - abstract
        """Fit model to features X and target y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        """Predict using fitted model returning 1D array."""

    # Persistence helpers expected by tests (save/load using pickle)
    def save_model(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path: str) -> 'SLBaseModel':
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, SLBaseModel):  # pragma: no cover - safety
            raise TypeError("Loaded object is not an SLBaseModel")
        return obj


class PyTorchSLModel(SLBaseModel):  # pragma: no cover - thin wrapper
    """Base class for PyTorch-backed models (minimal for tests)."""

    net: Any

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # simplistic training loop
        if torch is None:
            raise ImportError("PyTorch not available")
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        input_dim = X_t.shape[1]
        self.net = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1))  # type: ignore
        optim = torch.optim.Adam(self.net.parameters(), lr=0.01)
        for _ in range(self.config.get('epochs', 2)):
            optim.zero_grad()
            out = self.net(X_t)
            loss = ((out.view(-1) - y_t.view(-1)) ** 2).mean()
            loss.backward()
            optim.step()
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_t = torch.tensor(X, dtype=torch.float32)
        return self.net(X_t).detach().view(-1).cpu().numpy()
