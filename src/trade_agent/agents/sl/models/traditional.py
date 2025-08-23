"""Traditional (statistical / linear) model stubs."""
from __future__ import annotations

from typing import Any

import numpy as np
from .base import SLBaseModel

try:  # optional sklearn
    from sklearn.linear_model import Ridge, LinearRegression  # type: ignore
except Exception:  # pragma: no cover
    Ridge = LinearRegression = None  # type: ignore


class RidgeModel(SLBaseModel):
    model_type = "ridge"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        alpha = self.config.get("alpha", 1.0)
        self._impl = Ridge(alpha=alpha) if Ridge else None  # type: ignore

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        if self._impl is None:  # pragma: no cover
            raise ImportError("sklearn not available for RidgeModel")
        self._impl.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._impl.predict(X) if self._impl else np.zeros(len(X))


class LinearModel(SLBaseModel):
    model_type = "linear"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._impl = LinearRegression() if LinearRegression else None  # type: ignore

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearModel":
        if self._impl is None:  # pragma: no cover
            raise ImportError("sklearn not available for LinearModel")
        self._impl.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._impl.predict(X) if self._impl else np.zeros(len(X))


class GARCHModel(SLBaseModel):
    """Very small placeholder forecasting variance via rolling std."""

    model_type = "garch"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GARCHModel":  # type: ignore[override]
        arr = np.asarray(X).ravel()
        self._rolling = np.abs(arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        arr = np.asarray(X).ravel()
        if not self.is_fitted:
            return np.zeros_like(arr)
        out = np.zeros_like(arr)
        alpha = 0.1
        prev = 0.0
        for i, v in enumerate(arr):
            prev = alpha * abs(v) + (1 - alpha) * prev
            out[i] = prev
        return out
