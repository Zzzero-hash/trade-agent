"""Factory for supervised learning models."""
from __future__ import annotations

from typing import Dict, Type
from .base import SLBaseModel
from .traditional import RidgeModel, LinearModel, GARCHModel
from .tree_based import RandomForestModel, LightGBMModel, XGBoostModel
from .deep_learning import MLPModel, CNNLSTMModel, TransformerModel

_MODEL_REGISTRY: Dict[str, Type[SLBaseModel]] = {
    'ridge': RidgeModel,
    'linear': LinearModel,
    'garch': GARCHModel,
    'random_forest': RandomForestModel,
    'lightgbm': LightGBMModel,
    'xgboost': XGBoostModel,
    'mlp': MLPModel,
    'cnn_lstm': CNNLSTMModel,
    'transformer': TransformerModel,
}


class SLModelFactory:
    @staticmethod
    def create_model(model_type: str, config: dict) -> SLBaseModel:
        key = model_type.lower()
        if key not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        return _MODEL_REGISTRY[key](config)

    @staticmethod
    def get_available_models() -> list[str]:
        return sorted(_MODEL_REGISTRY.keys())

    @staticmethod
    def is_model_available(model_type: str) -> bool:
        return model_type.lower() in _MODEL_REGISTRY
