"""Simplified deep learning style models (numpy placeholders)."""
from __future__ import annotations

from typing import Any

import numpy as np
from .base import PyTorchSLModel


class MLPModel(PyTorchSLModel):
    model_type = "mlp"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPModel":  # type: ignore[override]
        X_aug = np.c_[X, np.ones(len(X))]
        try:
            w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        except Exception:  # pragma: no cover
            w = np.zeros(X_aug.shape[1])
        self._w = w
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        if not self.is_fitted:
            return np.zeros(len(X))
        X_aug = np.c_[X, np.ones(len(X))]
        return X_aug @ self._w


class _SequenceBase(PyTorchSLModel):
    sequence_length: int

    def _sliding(self, X: np.ndarray) -> np.ndarray:
        seq = self.sequence_length
        out = []
        for i in range(len(X) - seq + 1):
            window = X[i : i + seq]
            out.append(window.mean(axis=0))
        return np.asarray(out)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SequenceBase":  # type: ignore[override]
        self.sequence_length = int(self.config.get("sequence_length", 10))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        if not self.is_fitted:
            return np.zeros(max(len(X) - self.sequence_length + 1, 0))
        aggregated = self._sliding(X)
        return aggregated.mean(axis=1)


class CNNLSTMModel(_SequenceBase):
    model_type = "cnn_lstm"


class TransformerModel(_SequenceBase):
    model_type = "transformer"
