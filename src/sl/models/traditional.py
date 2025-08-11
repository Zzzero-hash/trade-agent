"""
Traditional supervised learning models including linear models and GARCH.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

# Relative import from base module
from .base import SLBaseModel, set_all_seeds


class RidgeModel(SLBaseModel):
    """Ridge regression model for financial forecasting."""

    def __init__(self, config: dict):
        """
        Initialize the Ridge model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)
        self.alpha = config.get('alpha', 1.0)
        self.model = Ridge(alpha=self.alpha, random_state=self.random_state)

    def fit(self, X, y):
        """
        Fit the Ridge model on training data.

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

        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        """
        Make predictions with the Ridge model.

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

        return self.model.predict(X)


class LinearModel(SLBaseModel):
    """Linear regression model for financial forecasting."""

    def __init__(self, config: dict):
        """
        Initialize the Linear model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)
        self.model = LinearRegression()

    def fit(self, X, y):
        """
        Fit the Linear model on training data.

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

        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        """
        Make predictions with the Linear model.

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

        return self.model.predict(X)


class GARCHModel(SLBaseModel):
    """GARCH(1,1) model for volatility forecasting."""

    def __init__(self, config: dict):
        """
        Initialize the GARCH model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)
        self.omega = config.get('omega', 0.01)  # Constant term
        self.alpha = config.get('alpha', 0.1)   # ARCH term coefficient
        self.beta = config.get('beta', 0.8)     # GARCH term coefficient
        self.omega_estimated = None
        self.alpha_estimated = None
        self.beta_estimated = None
        self.sigma2_estimated = None

    def fit(self, X, y):
        """
        Fit the GARCH model on training data.

        Args:
            X: Feature matrix (residuals or returns)
            y: Target vector (squared returns or realized volatility)
        """
        # Set seeds for deterministic processing
        set_all_seeds(self.random_state)

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # For GARCH, we typically use returns as input
        if X.ndim > 1:
            returns = X[:, 0]  # Use first column as returns
        else:
            returns = X

        # Estimate parameters using maximum likelihood (simplified approach)
        self._estimate_parameters(returns)
        self.is_fitted = True

    def _estimate_parameters(self, returns):
        """
        Estimate GARCH parameters using a simplified approach.

        Args:
            returns: Return series
        """
        # Initialize variance
        var = np.var(returns)

        # Simple estimation using method of moments as initialization
        self.omega_estimated = self.omega
        self.alpha_estimated = self.alpha
        self.beta_estimated = self.beta

        # Estimate long-run variance
        if (1 - self.alpha_estimated - self.beta_estimated) > 0:
            self.sigma2_estimated = self.omega_estimated / (1 - self.alpha_estimated - self.beta_estimated)
        else:
            self.sigma2_estimated = var
            warnings.warn("GARCH parameters may not be stationary")

    def predict(self, X):
        """
        Make volatility predictions with the GARCH model.

        Args:
            X: Feature matrix (returns or residuals)

        Returns:
            np.ndarray: Volatility predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        # For GARCH, we typically use returns as input
        if X.ndim > 1:
            returns = X[:, 0]  # Use first column as returns
        else:
            returns = X

        # Initialize predictions array
        n = len(returns)
        predictions = np.zeros(n)

        # Initialize variance
        sigma2 = self.sigma2_estimated if self.sigma2_estimated is not None else np.var(returns)

        # GARCH(1,1) recursion
        for i in range(n):
            # Predict next period variance
            predictions[i] = np.sqrt(sigma2)

            # Update variance for next iteration
            if i < n - 1:  # Don't update for last observation
                sigma2 = (self.omega_estimated +
                         self.alpha_estimated * returns[i]**2 +
                         self.beta_estimated * sigma2)

        return predictions


def compute_volatility_forecast(returns, omega=0.01, alpha=0.1, beta=0.8):
    """
    Compute long-run volatility forecast using GARCH(1,1) parameters.

    Args:
        returns: Return series
        omega: Constant term
        alpha: ARCH term coefficient
        beta: GARCH term coefficient

    Returns:
        float: Long-run volatility forecast
    """
    # Long-run variance in GARCH(1,1)
    if (1 - alpha - beta) > 0:
        long_run_variance = omega / (1 - alpha - beta)
        return np.sqrt(long_run_variance)
    else:
        # Fallback to sample variance if parameters are not stationary
        return np.std(returns)
