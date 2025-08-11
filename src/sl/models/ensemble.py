"""
Ensemble methods and stacking for supervised learning models.
"""
from typing import Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import from base module
try:
    from .base import SLBaseModel, set_all_seeds
except ImportError:
    # Fallback for development environment
    from src.sl.models.base import SLBaseModel, set_all_seeds


class EnsembleModel(SLBaseModel):
    """Ensemble model that combines multiple base models."""

    def __init__(self, config: dict):
        """
        Initialize the Ensemble model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)

        # Ensemble configuration
        self.base_models_config = config.get('base_models', [])
        self.base_models = []
        self.weights = config.get('weights', None)  # If None, use equal weights
        self.ensemble_method = config.get('ensemble_method', 'weighted_average')

        # Initialize base models
        self._initialize_base_models()

    def _initialize_base_models(self):
        """Initialize base models from configuration."""
        # This would typically use a factory pattern, but we'll create them directly
        # for simplicity in this implementation
        pass  # Base models will be set during fitting

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the ensemble model on training data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        # Set seeds for deterministic processing
        set_all_seeds(self.random_state)

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Fit all base models
        predictions = []
        for model in self.base_models:
            model.fit(X, y)
            pred = model.predict(X)
            predictions.append(pred)

        # Store training predictions for potential use in stacking
        self.train_predictions_ = np.column_stack(predictions)

        # If weights are not provided, compute them based on performance
        if self.weights is None:
            self._compute_weights(y)

        self.is_fitted = True

    def _compute_weights(self, y_true: np.ndarray):
        """
        Compute weights for ensemble based on model performance.

        Args:
            y_true: True target values
        """
        if not hasattr(self, 'train_predictions_'):
            raise ValueError("Model must be fitted before computing weights")

        # Compute MSE for each model
        mses = []
        for i in range(self.train_predictions_.shape[1]):
            mse = mean_squared_error(y_true, self.train_predictions_[:, i])
            mses.append(mse)

        # Convert MSEs to weights (inverse of MSE, normalized)
        if np.all(np.array(mses) == 0):
            # All models perfect, use equal weights
            self.weights = np.ones(len(mses)) / len(mses)
        else:
            # Inverse of MSE, normalized
            inverse_mses = 1.0 / (np.array(mses) + 1e-8)  # Add small epsilon to avoid division by zero
            self.weights = inverse_mses / np.sum(inverse_mses)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the ensemble model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get predictions from all base models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)

        # Combine predictions based on ensemble method
        if self.ensemble_method == 'weighted_average':
            return self._weighted_average(predictions)
        elif self.ensemble_method == 'simple_average':
            return self._simple_average(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _weighted_average(self, predictions: list[np.ndarray]) -> np.ndarray:
        """
        Compute weighted average of predictions.

        Args:
            predictions: List of prediction arrays

        Returns:
            np.ndarray: Weighted average predictions
        """
        if self.weights is None:
            # Use equal weights if not computed
            self.weights = np.ones(len(predictions)) / len(predictions)

        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += self.weights[i] * pred

        return weighted_pred

    def _simple_average(self, predictions: list[np.ndarray]) -> np.ndarray:
        """
        Compute simple average of predictions.

        Args:
            predictions: List of prediction arrays

        Returns:
            np.ndarray: Simple average predictions
        """
        return np.mean(predictions, axis=0)

    def add_model(self, model: SLBaseModel):
        """
        Add a base model to the ensemble.

        Args:
            model: Base model to add
        """
        self.base_models.append(model)


class StackingModel(SLBaseModel):
    """Stacking ensemble model with a meta-learner."""

    def __init__(self, config: dict):
        """
        Initialize the Stacking model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)

        # Stacking configuration
        self.base_models_config = config.get('base_models', [])
        self.base_models = []
        self.meta_learner_config = config.get('meta_learner', {'type': 'linear'})
        self.meta_learner = None
        self.cv_folds = config.get('cv_folds', 5)

        # Initialize meta-learner
        self._initialize_meta_learner()

    def _initialize_meta_learner(self):
        """Initialize the meta-learner."""
        meta_type = self.meta_learner_config.get('type', 'linear')
        if meta_type == 'linear':
            self.meta_learner = LinearRegression()
        else:
            raise ValueError(f"Unsupported meta-learner type: {meta_type}")

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the stacking model on training data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        # Set seeds for deterministic processing
        set_all_seeds(self.random_state)

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Fit base models and generate meta-features
        meta_features = self._generate_meta_features(X, y)

        # Fit meta-learner
        self.meta_learner.fit(meta_features, y)

        # Fit base models on full dataset
        for model in self.base_models:
            model.fit(X, y)

        self.is_fitted = True

    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate meta-features using cross-validation.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            np.ndarray: Meta-features for training meta-learner
        """
        from sklearn.model_selection import KFold

        # Initialize meta-features array
        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        # Create cross-validation folds
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Generate out-of-fold predictions for each base model
        for i, model in enumerate(self.base_models):
            oof_predictions = np.zeros(X.shape[0])

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, _y_val = y[train_idx], y[val_idx]

                # Fit model on training fold
                model.fit(X_train, y_train)

                # Predict on validation fold
                oof_predictions[val_idx] = model.predict(X_val)

            meta_features[:, i] = oof_predictions

        return meta_features

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the stacking model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get predictions from all base models as meta-features
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            meta_features[:, i] = model.predict(X)

        # Predict with meta-learner
        return self.meta_learner.predict(meta_features)

    def add_model(self, model: SLBaseModel):
        """
        Add a base model to the stacking ensemble.

        Args:
            model: Base model to add
        """
        self.base_models.append(model)


class WeightedEnsembleModel(SLBaseModel):
    """Weighted ensemble model with dynamic weight adjustment."""

    def __init__(self, config: dict):
        """
        Initialize the Weighted Ensemble model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)

        # Weighted ensemble configuration
        self.base_models = []
        self.weights = []
        self.weight_update_method = config.get('weight_update_method', 'performance')
        self.learning_rate = config.get('learning_rate', 0.1)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the weighted ensemble model on training data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        # Set seeds for deterministic processing
        set_all_seeds(self.random_state)

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Initialize equal weights if not set
        if len(self.weights) == 0:
            self.weights = [1.0 / len(self.base_models)] * len(self.base_models)

        # Fit all base models
        for model in self.base_models:
            model.fit(X, y)

        self.is_fitted = True

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the weighted ensemble model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get weighted predictions from all base models
        weighted_pred = np.zeros(X.shape[0])
        for i, model in enumerate(self.base_models):
            pred = model.predict(X)
            weighted_pred += self.weights[i] * pred

        return weighted_pred

    def update_weights(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Update model weights based on recent performance.

        Args:
            y_true: True target values
            y_pred: Predicted values
        """
        # This is a simplified weight update - in practice, you might want to
        # update weights based on individual model performance
        pass

    def add_model(self, model: SLBaseModel, weight: float = None):
        """
        Add a base model to the weighted ensemble.

        Args:
            model: Base model to add
            weight: Initial weight for the model (if None, will be computed)
        """
        self.base_models.append(model)
        if weight is not None:
            self.weights.append(weight)
        else:
            # Initialize with equal weight
            self.weights.append(1.0 / len(self.base_models))
