"""Tree & boosting model stubs."""
from __future__ import annotations

from typing import Any

import numpy as np
from .base import SLBaseModel

try:  # optional deps
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
except Exception:  # pragma: no cover
    RandomForestRegressor = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore


class RandomForestModel(SLBaseModel):
    model_type = "random_forest"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        n_estimators = self.config.get("n_estimators", 10)
        self._impl = (
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=self.config.get("random_state", 42),
            )
            if RandomForestRegressor
            else None
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        if self._impl is None:  # pragma: no cover
            raise ImportError("sklearn not available for RandomForestModel")
        self._impl.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._impl.predict(X) if self._impl else np.zeros(len(X))


class LightGBMModel(SLBaseModel):
    model_type = "lightgbm"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":  # type: ignore[override]
        if lgb is None:  # pragma: no cover
            raise ImportError("lightgbm not installed")
        self._train_set = lgb.Dataset(X, label=y)  # type: ignore[attr-defined]
        params = {"objective": "regression", "verbose": -1}
        self._model = lgb.train(params, self._train_set, num_boost_round=5)  # type: ignore[attr-defined]
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        if getattr(self, "_model", None) is None:
            return np.zeros(len(X))
        return self._model.predict(X)


class XGBoostModel(SLBaseModel):
    model_type = "xgboost"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":  # type: ignore[override]
        if xgb is None:  # pragma: no cover
            raise ImportError("xgboost not installed")
        dtrain = xgb.DMatrix(X, label=y)  # type: ignore[attr-defined]
        params = {"objective": "reg:squarederror"}
        self._model = xgb.train(params, dtrain, num_boost_round=5)  # type: ignore[attr-defined]
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        if getattr(self, "_model", None) is None:
            return np.zeros(len(X))
        dmat = xgb.DMatrix(X)  # type: ignore[attr-defined]
        return self._model.predict(dmat)
