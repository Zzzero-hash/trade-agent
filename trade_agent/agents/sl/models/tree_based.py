"""Tree-based model placeholders."""
from __future__ import annotations

import numpy as np
from .base import SLBaseModel

try:  # pragma: no cover - optional deps
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover
    RandomForestRegressor = None  # type: ignore


class RandomForestModel(SLBaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if RandomForestRegressor is None:
            # Fallback: store mean per feature subset (trivial)
            self._mean = float(np.mean(y))
            self._use_sklearn = False
        else:  # pragma: no cover
            self._model = RandomForestRegressor(  # type: ignore
                n_estimators=self.config.get('n_estimators', 10),
                random_state=self.config.get('random_state', 42),
            )
            self._model.fit(X, y)
            self._use_sklearn = True
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        if getattr(self, '_use_sklearn', False):  # pragma: no cover
            return self._model.predict(X)
        return np.full(len(X), self._mean)


# LightGBM / XGBoost placeholders (skip if libs missing)
class LightGBMModel(SLBaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover - likely missing lib
        self._mean = float(np.mean(y))
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        return np.full(len(X), self._mean)


class XGBoostModel(SLBaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover - likely missing lib
        self._mean = float(np.mean(y))
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        return np.full(len(X), self._mean)
