"""
Tree-based supervised learning models including XGBoost, LightGBM, and Random Forest.
"""
import warnings

import pandas as pd

# Try to import tree-based models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with 'pip install xgboost'")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with 'pip install lightgbm'")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Install with 'pip install scikit-learn'")

# Import from base module
try:
    from .base import SLBaseModel, set_all_seeds
except ImportError:
    # Fallback for development environment
    from src.sl.models.base import SLBaseModel, set_all_seeds


class XGBoostModel(SLBaseModel):
    """XGBoost regression model for financial forecasting."""

    def __init__(self, config: dict):
        """
        Initialize the XGBoost model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with 'pip install xgboost'")

        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.subsample = config.get('subsample', 1.0)
        self.colsample_bytree = config.get('colsample_bytree', 1.0)
        self.reg_alpha = config.get('reg_alpha', 0)
        self.reg_lambda = config.get('reg_lambda', 1)
        self.model = None

    def fit(self, X, y):
        """
        Fit the XGBoost model on training data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with 'pip install xgboost'")

        # Set seeds for deterministic processing
        set_all_seeds(self.random_state)

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Create XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': -1
        }

        # Fit the model
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        """
        Make predictions with the XGBoost model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with 'pip install xgboost'")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)


class LightGBMModel(SLBaseModel):
    """LightGBM regression model for financial forecasting."""

    def __init__(self, config: dict):
        """
        Initialize the LightGBM model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with 'pip install lightgbm'")

        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', -1)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.num_leaves = config.get('num_leaves', 31)
        self.subsample = config.get('subsample', 1.0)
        self.colsample_bytree = config.get('colsample_bytree', 1.0)
        self.reg_alpha = config.get('reg_alpha', 0)
        self.reg_lambda = config.get('reg_lambda', 0)
        self.model = None

    def fit(self, X, y):
        """
        Fit the LightGBM model on training data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with 'pip install lightgbm'")

        # Set seeds for deterministic processing
        set_all_seeds(self.random_state)

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Create LightGBM parameters
        params = {
            'objective': 'regression',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }

        # Fit the model
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        """
        Make predictions with the LightGBM model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with 'pip install lightgbm'")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)


class RandomForestModel(SLBaseModel):
    """Random Forest regression model for financial forecasting."""

    def __init__(self, config: dict):
        """
        Initialize the Random Forest model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with 'pip install scikit-learn'")

        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', None)
        self.min_samples_split = config.get('min_samples_split', 2)
        self.min_samples_leaf = config.get('min_samples_leaf', 1)
        self.max_features = config.get('max_features', 'sqrt')
        self.model = None

    def fit(self, X, y):
        """
        Fit the Random Forest model on training data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with 'pip install scikit-learn'")

        # Set seeds for deterministic processing
        set_all_seeds(self.random_state)

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Create Random Forest parameters
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': -1
        }

        # Fit the model
        self.model = RandomForestRegressor(**params)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        """
        Make predictions with the Random Forest model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with 'pip install scikit-learn'")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)
