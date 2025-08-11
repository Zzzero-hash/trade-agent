"""
Base classes for supervised learning models.
"""
import os
from abc import ABC, abstractmethod
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class SLBaseModel(ABC):
    """Base class for all supervised learning models."""

    def __init__(self, config: dict):
        """
        Initialize the base model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        self.config = config
        self.model = None
        self.is_fitted = False
        self.random_state = config.get('random_state', 42)
        self.model_name = config.get('model_name', self.__class__.__name__)

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the model on training data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Point predictions.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        pass

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Probability predictions (if applicable).

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Probability predictions
        """
        raise NotImplementedError("predict_proba not implemented for this model")

    def predict_quantiles(self, X: Union[np.ndarray, pd.DataFrame],
                         quantiles: list[float]) -> np.ndarray:
        """
        Quantile predictions (if applicable).

        Args:
            X: Feature matrix
            quantiles: List of quantiles to predict

        Returns:
            np.ndarray: Quantile predictions
        """
        raise NotImplementedError("predict_quantiles not implemented for this model")

    def save_model(self, path: str):
        """
        Save model to disk.

        Args:
            path (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save model
        joblib.dump(self, path)

    @classmethod
    def load_model(cls, path: str):
        """
        Load model from disk.

        Args:
            path (str): Path to load the model from

        Returns:
            SLBaseModel: Loaded model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        return joblib.load(path)


class PyTorchSLModel(SLBaseModel):
    """Base class for PyTorch-based supervised learning models."""

    def __init__(self, config: dict):
        """
        Initialize the PyTorch model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() and
                                  config.get('use_cuda', True) else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)

    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame],
                     y: Optional[Union[np.ndarray, pd.Series]] = None):
        """
        Convert data to PyTorch tensors.

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Returns:
            torch.Tensor or tuple: Prepared data
        """
        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            return X_tensor, y_tensor
        return X_tensor

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the PyTorch model.

        Args:
            X: Feature matrix
            y: Target vector
        """
        if self.model is None:
            raise ValueError("Model not initialized. Please implement model creation in subclass.")

        X_tensor, y_tensor = self._prepare_data(X, y)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

        self.is_fitted = True

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the PyTorch model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        self.model.eval()
        with torch.no_grad():
            X_tensor = self._prepare_data(X)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

    def save_model(self, path: str):
        """
        Save PyTorch model to disk.

        Args:
            path (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save model state dict and config
        model_data = {
            'state_dict': self.model.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        torch.save(model_data, path)

    @classmethod
    def load_model(cls, path: str):
        """
        Load PyTorch model from disk.

        Args:
            path (str): Path to load the model from

        Returns:
            PyTorchSLModel: Loaded model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load model data
        model_data = torch.load(path)
        config = model_data['config']

        # Create model instance
        model = cls(config)
        model.model.load_state_dict(model_data['state_dict'])
        model.is_fitted = True

        return model


def set_all_seeds(seed: int = 42):
    """
    Set seeds for all random number generators to ensure deterministic processing.

    Args:
        seed (int): Random seed to use
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
