"""Deep learning style model placeholders (NumPy only)."""
from __future__ import annotations

import numpy as np
from .base import SLBaseModel


class MLPModel(SLBaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Store simple linear weights as placeholder
        X_ = np.c_[np.ones(len(X)), X]
        self._coef = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        X_ = np.c_[np.ones(len(X)), X]
        return (X_ @ self._coef).astype(float)


class CNNLSTMModel(SLBaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # For simplicity compute rolling mean with window=sequence_length
        self.seq_len = int(self.config.get('sequence_length', 10))
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        seq = self.seq_len
        out = []
        for i in range(len(X) - seq + 1):
            out.append(float(np.mean(X[i : i + seq], dtype=np.float64)))
        return np.array(out, dtype=float)


class TransformerModel(CNNLSTMModel):  # same placeholder behavior
    pass
