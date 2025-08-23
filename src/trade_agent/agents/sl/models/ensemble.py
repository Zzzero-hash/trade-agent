"""Ensemble model placeholders."""
from __future__ import annotations

from typing import List

import numpy as np
from .base import SLBaseModel


class EnsembleModel(SLBaseModel):
    model_type = "ensemble"

    def __init__(self, config: dict | None = None) -> None:  # type: ignore[override]
        super().__init__(config)
        self._models: List[SLBaseModel] = []

    def add_model(self, model: SLBaseModel) -> None:
        self._models.append(model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":  # type: ignore[override]
        for m in self._models:
            m.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        if not self._models:
            return np.zeros(len(X))
        preds = [m.predict(X) for m in self._models]
        return sum(preds) / len(preds)


class StackingModel(EnsembleModel):
    model_type = "stacking"

    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return super().predict(X)
