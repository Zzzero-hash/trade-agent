"""
Factory for creating supervised learning models.
"""
from typing import Any

# Import model classes
try:
    # Traditional models
    # Deep learning models
    from .deep_learning import CNNLSTMModel, MLPModel, TransformerModel

    # Ensemble models
    from .ensemble import EnsembleModel, StackingModel, WeightedEnsembleModel
    from .traditional import GARCHModel, LinearModel, RidgeModel

    # Tree-based models
    from .tree_based import LightGBMModel, RandomForestModel, XGBoostModel
except ImportError:
    # Fallback imports for development environment
    from src.sl.models.deep_learning import CNNLSTMModel, MLPModel, TransformerModel
    from src.sl.models.ensemble import (
        EnsembleModel,
        StackingModel,
        WeightedEnsembleModel,
    )
    from src.sl.models.traditional import GARCHModel, LinearModel, RidgeModel
    from src.sl.models.tree_based import LightGBMModel, RandomForestModel, XGBoostModel


class SLModelFactory:
    """Factory class for creating supervised learning models."""

    # Registry of available models
    _model_registry = {
        # Traditional models
        'ridge': RidgeModel,
        'linear': LinearModel,
        'garch': GARCHModel,

        # Tree-based models
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'random_forest': RandomForestModel,

        # Deep learning models
        'cnn_lstm': CNNLSTMModel,
        'transformer': TransformerModel,
        'mlp': MLPModel,

        # Ensemble models
        'ensemble': EnsembleModel,
        'stacking': StackingModel,
        'weighted_ensemble': WeightedEnsembleModel,
    }

    @classmethod
    def create_model(cls, model_type: str, config: dict[str, Any]):
        """
        Create a supervised learning model instance.

        Args:
            model_type (str): Type of model to create
            config (Dict[str, Any]): Configuration for the model

        Returns:
            SLBaseModel: Instance of the requested model

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")

        model_class = cls._model_registry[model_type]
        return model_class(config)

    @classmethod
    def register_model(cls, model_type: str, model_class):
        """
        Register a new model type.

        Args:
            model_type (str): Type identifier for the model
            model_class: Model class to register
        """
        cls._model_registry[model_type] = model_class

    @classmethod
    def get_available_models(cls):
        """
        Get list of available model types.

        Returns:
            List[str]: List of available model types
        """
        return list(cls._model_registry.keys())

    @classmethod
    def is_model_available(cls, model_type: str) -> bool:
        """
        Check if a model type is available.

        Args:
            model_type (str): Type of model to check

        Returns:
            bool: True if model is available, False otherwise
        """
        return model_type in cls._model_registry
