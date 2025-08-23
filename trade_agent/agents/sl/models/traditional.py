"""Traditional statistical / linear models."""
from __future__ import annotations

import numpy as np
from typing import Any, Dict
from .base import SLBaseModel

try:
    from sklearn.linear_model import LinearRegression, Ridge
except Exception:  # pragma: no cover
    LinearRegression = Ridge = None  # type: ignore


class LinearModel(SLBaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if LinearRegression is None:
            # Fallback simple normal equation solution
            X_ = np.c_[np.ones(len(X)), X]
            self._coef = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
            self._use_sklearn = False
        else:  # pragma: no cover - exercised if sklearn present
            self._model = LinearRegression()
            self._model.fit(X, y)
            self._use_sklearn = True
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        if getattr(self, '_use_sklearn', False):  # pragma: no cover
            return self._model.predict(X)
        X_ = np.c_[np.ones(len(X)), X]
        return X_ @ self._coef


class RidgeModel(SLBaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if Ridge is None:
            # Simple ridge via normal equation with alpha
            alpha = float(self.config.get('alpha', 1.0))
            X_ = np.c_[np.ones(len(X)), X]
            I = np.eye(X_.shape[1])
            self._coef = np.linalg.pinv(X_.T @ X_ + alpha * I) @ X_.T @ y
            self._use_sklearn = False
        else:  # pragma: no cover
            self._model = Ridge(alpha=self.config.get('alpha', 1.0))
            self._model.fit(X, y)
            self._use_sklearn = True
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        if getattr(self, '_use_sklearn', False):  # pragma: no cover
            return self._model.predict(X)
        X_ = np.c_[np.ones(len(X)), X]
        return X_ @ self._coef


class GARCHModel(SLBaseModel):
    """Very small placeholder: predicts rolling variance as proxy."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # X is a 1D array of returns in tests (passed as X[:,0])
        series = np.asarray(X)
        window = int(self.config.get('window', 10))
        # Precompute rolling variance
        vars_: list[float] = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            vars_.append(float(np.var(series[start:i+1]) or 0.0))
        self._variance = np.array(vars_)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        return self._variance[: len(X)]
