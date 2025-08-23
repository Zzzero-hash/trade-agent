"""Factory for creating SL model instances by string key."""
from __future__ import annotations

from typing import Any, Dict, Type

from .base import SLBaseModel
from .traditional import RidgeModel, LinearModel, GARCHModel
from .tree_based import RandomForestModel, LightGBMModel, XGBoostModel
from .deep_learning import MLPModel, CNNLSTMModel, TransformerModel
from .ensemble import EnsembleModel, StackingModel

_REGISTRY: Dict[str, Type[SLBaseModel]] = {
    "ridge": RidgeModel,
    "linear": LinearModel,
    "garch": GARCHModel,
    "random_forest": RandomForestModel,
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "mlp": MLPModel,
    "cnn_lstm": CNNLSTMModel,
    "transformer": TransformerModel,
    "ensemble": EnsembleModel,
    "stacking": StackingModel,
}


class SLModelFactory:
    """Factory with helper utilities."""

    @staticmethod
    def create_model(model_type: str, config: dict[str, Any] | None = None) -> SLBaseModel:
        key = model_type.lower()
        if key not in _REGISTRY:  # pragma: no cover
            raise ValueError(f"Unknown model_type '{model_type}'")
        return _REGISTRY[key](config)

    @staticmethod
    def get_available_models() -> list[str]:
        return sorted(_REGISTRY.keys())

    @staticmethod
    def is_model_available(model_type: str) -> bool:
        return model_type.lower() in _REGISTRY
