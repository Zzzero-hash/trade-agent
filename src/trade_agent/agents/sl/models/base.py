"""Core abstractions & utilities for supervised learning models.

This re-implements a subset of the original framework required for the
unit test suite: seeding helper, a pickle-based persistence layer, and
abstract base classes for traditional & PyTorch style models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
import pickle
import random
from typing import Any, ClassVar, Type, TypeVar

import numpy as np

try:  # optional torch
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

__all__ = ["set_all_seeds", "SLBaseModel", "PyTorchSLModel"]

T_SLModel = TypeVar("T_SLModel", bound="SLBaseModel")


def set_all_seeds(seed: int | None) -> None:
    """Set seeds across common libraries (numpy, random, torch)."""
    if seed is None:
        return
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    os.environ["PYTHONHASHSEED"] = str(seed_int)
    if torch is not None:  # pragma: no branch
        try:
            torch.manual_seed(seed_int)
            if (
                hasattr(torch, "cuda")
                and torch.cuda.is_available()  # type: ignore[attr-defined]
            ):
                torch.cuda.manual_seed_all(  # type: ignore[attr-defined]
                    seed_int
                )
        except Exception:  # pragma: no cover
            pass


class SLBaseModel(ABC):
    """Abstract supervised learning model interface.

    Subclasses must implement ``fit`` and ``predict``.
    A minimal pickle persistence API is provided.
    """

    model_type: ClassVar[str | None] = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:  # noqa: D401
        self.config = config or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SLBaseModel":  # noqa: D401
        """Train model in-place."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        """Return predictions for feature matrix ``X``."""

    # ---- Persistence ---- #
    def save_model(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls: Type[T_SLModel], path: str) -> T_SLModel:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):  # pragma: no cover - defensive
            raise TypeError(f"Loaded object is not {cls.__name__}")
        return obj


class PyTorchSLModel(SLBaseModel):  # pragma: no cover - thin shim
    """Base class for torch models (import compatibility only)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.device = "cpu"
        if torch is not None and getattr(torch, "cuda", None):  # type: ignore[attr-defined]
            self.device = (
                "cuda"
                if torch.cuda.is_available()  # type: ignore[attr-defined]
                else "cpu"
            )
