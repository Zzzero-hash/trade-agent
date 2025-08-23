"""Ensemble model placeholders."""
from __future__ import annotations

import numpy as np
from .base import SLBaseModel


class EnsembleModel(SLBaseModel):
    def __init__(self, config):  # type: ignore[no-untyped-def]
        super().__init__(config)
        self._models: list[SLBaseModel] = []

    def add_model(self, model: SLBaseModel) -> None:
        self._models.append(model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for m in self._models:
            m.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        preds = [m.predict(X) for m in self._models]
        if not preds:
            return np.zeros(len(X))
        return np.mean(preds, axis=0)


class StackingModel(EnsembleModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Same behavior for placeholder
        super().fit(X, y)
