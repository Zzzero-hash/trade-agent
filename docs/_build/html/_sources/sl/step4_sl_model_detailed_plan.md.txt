# Step 4: Supervised Learning Model Implementation - Detailed Plan

## Objective

Design and plan the supervised learning component of the ensemble pipeline that will serve as the first stage in the SL → PPO+SAC architecture.

## 1. SL Model Architecture Design

### 1.1 Model Types for Different Prediction Targets

#### 1.1.1 Expected Returns (E[r]) Forecasting

- **Primary Models**:
  - XGBoost Regressor (for non-linear relationships and feature interactions)
  - LightGBM Regressor (for efficiency with large datasets)
  - Ridge Regression (for linear baseline and regularization)
  - LSTM Neural Network (for sequential patterns)

- **Ensemble Approach**:
  - Weighted average based on cross-validation performance
  - Stacking with a meta-learner (Linear Regression or small MLP)

#### 1.1.2 Volatility (σ) Forecasting

- **Primary Models**:
  - GARCH(1,1) model baseline (traditional financial approach)
  - Random Forest Regressor (for non-linear volatility patterns)
  - LSTM Neural Network (for sequential volatility modeling)
  - SVR (Support Vector Regression) with RBF kernel

#### 1.1.3 Probability Distributions Forecasting

- **Primary Models**:
  - Quantile Regression Forests (for distributional forecasts)
  - Neural Network with quantile loss (for deep quantile regression)
  - Mixture Density Networks (for full distribution modeling)
  - Bayesian Neural Networks (for uncertainty quantification)

### 1.2 Model Architecture Components

#### 1.2.1 Base Model Interface

```python
class SLBaseModel:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.random_state = config.get('random_state', 42)

    def fit(self, X, y):
        """Fit the model on training data"""
        pass

    def predict(self, X):
        """Point predictions"""
        pass

    def predict_proba(self, X):
        """Probability predictions (if applicable)"""
        pass

    def predict_quantiles(self, X, quantiles):
        """Quantile predictions (if applicable)"""
        pass

    def save_model(self, path):
        """Save model to disk"""
        pass

    def load_model(self, path):
        """Load model from disk"""
        pass
```

#### 1.2.2 Model Factory

```python
class SLModelFactory:
    @staticmethod
    def create_model(model_type: str, config: dict):
        if model_type == "xgboost":
            return XGBoostModel(config)
        elif model_type == "lightgbm":
            return LightGBMModel(config)
        elif model_type == "ridge":
            return RidgeModel(config)
        elif model_type == "lstm":
            return LSTMModel(config)
        elif model_type == "rf":
            return RandomForestModel(config)
        # ... other model types
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

### 1.3 PyTorch Framework Integration

#### 1.3.1 Deep Learning Models with PyTorch

- LSTM-based models for sequential data
- Transformer models for attention-based forecasting
- MLP models for non-sequential features

#### 1.3.2 PyTorch Model Base Class

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class PyTorchSLModel(SLBaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None

    def _prepare_data(self, X, y=None):
        """Convert data to PyTorch tensors"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            return X_tensor, y_tensor
        return X_tensor

    def fit(self, X, y):
        """Fit the PyTorch model"""
        X_tensor, y_tensor = self._prepare_data(X, y)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config.get('batch_size', 32), shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.config.get('epochs', 100)):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

        self.is_fitted = True

    def predict(self, X):
        """Make predictions with the PyTorch model"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._prepare_data(X)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
```

## 2. Training Pipeline with Cross-Validation

### 2.1 Temporal Cross-Validation Strategy

To prevent data leakage in time-series forecasting, we implement temporal cross-validation:

```{mermaid}
graph LR
    A[Training Set 1] --> B[Validation Set 1]
    B --> C[Training Set 2] --> D[Validation Set 2]
    D --> E[Training Set 3] --> F[Validation Set 3]
    F --> G[Training Set 4] --> H[Validation Set 4]
    H --> I[Training Set 5] --> J[Validation Set 5]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e1f5fe
    style D fill:#f3e5f5
    style E fill:#e1f5fe
    style F fill:#f3e5f5
    style G fill:#e1f5fe
    style H fill:#f3e5f5
    style I fill:#e1f5fe
    style J fill:#f3e5f5
```

### 2.2 Cross-Validation Implementation

```python
from sklearn.model_selection import TimeSeriesSplit

class TemporalCV:
    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap
        self.cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and validation sets"""
        return self.cv.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.cv.get_n_splits(X, y, groups)
```

### 2.3 Hyperparameter Tuning with Optuna

```python
import optuna

class HyperparameterTuner:
    def __init__(self, model_class, cv_strategy, scoring_metric):
        self.model_class = model_class
        self.cv_strategy = cv_strategy
        self.scoring_metric = scoring_metric

    def objective(self, trial):
        # Suggest hyperparameters
        params = self._suggest_params(trial)

        # Cross-validation
        scores = []
        for train_idx, val_idx in self.cv_strategy.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self.model_class(params)
            model.fit(X_train, y_train)
            score = self._evaluate(model, X_val, y_val)
            scores.append(score)

        return np.mean(scores)

    def tune(self, X, y, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params
```

## 3. Model Evaluation Metrics and Validation Procedures

### 3.1 Performance Metrics

#### 3.1.1 Regression Metrics

- **Mean Squared Error (MSE)**: `MSE = 1/n * Σ(y_i - ŷ_i)²`
- **Mean Absolute Error (MAE)**: `MAE = 1/n * Σ|y_i - ŷ_i|`
- **Root Mean Squared Error (RMSE)**: `RMSE = √(MSE)`
- **Mean Absolute Percentage Error (MAPE)**: `MAPE = 100/n * Σ|(y_i - ŷ_i)/y_i|`
- **R-squared (R²)**: `R² = 1 - (Σ(y_i - ŷ_i)²)/(Σ(y_i - ȳ)²)`

#### 3.1.2 Financial Metrics

- **Directional Accuracy**: Percentage of correctly predicted return directions
- **Sharpe Ratio of Predictions**: Risk-adjusted return of strategy based on predictions
- **Information Coefficient (IC)**: Correlation between predictions and actual returns
- **Maximum Drawdown**: Largest peak-to-trough decline in predicted portfolio value

#### 3.1.3 Distributional Metrics

- **Pinball Loss**: For quantile predictions
- **Continuous Ranked Probability Score (CRPS)**: For probabilistic forecasts
- **Log-Likelihood**: For density forecasts

### 3.2 Validation Procedures

#### 3.2.1 Out-of-Sample Testing

- Walk-forward analysis with expanding and rolling windows
- Backtesting on historical data with proper temporal splits

#### 3.2.2 Stability Checks

- Parameter stability over time
- Model performance degradation monitoring
- Concept drift detection using statistical tests

## 4. Model Persistence and Versioning Strategy

### 4.1 Model Serialization

- **Scikit-learn models**: Using joblib for efficient serialization
- **PyTorch models**: Using torch.save() with state_dict for weights
- **XGBoost/LightGBM models**: Using built-in save_model() methods

### 4.2 Versioning Strategy

```python
import hashlib
import json

class ModelVersioning:
    def __init__(self, model_dir="models/"):
        self.model_dir = model_dir

    def create_version(self, model, config, metrics):
        """Create a new model version with metadata"""
        # Generate unique identifier based on config and training data hash
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{timestamp}_{config_hash}"

        # Save model and metadata
        model_path = f"{self.model_dir}/sl_model_{version}.pkl"
        metadata_path = f"{self.model_dir}/sl_model_{version}_metadata.json"

        # Save model
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            "version": version,
            "timestamp": timestamp,
            "config": config,
            "metrics": metrics,
            "model_path": model_path
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return version
```

### 4.3 Model Registry

```python
class ModelRegistry:
    def __init__(self, registry_path="models/registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def register_model(self, model_name, version, metrics, is_production=False):
        """Register a model version in the registry"""
        entry = {
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "is_production": is_production
        }

        self.registry[model_name] = entry
        self._save_registry()

        if is_production:
            self._set_production_model(model_name, version)

    def get_production_model(self, model_name):
        """Get the production version of a model"""
        if model_name in self.registry:
            return self.registry[model_name]
        return None
```

## 5. File Paths for All SL-Related Components

```
src/sl/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── base.py                 # Base model classes
│   ├── traditional.py          # Linear models, GARCH, etc.
│   ├── tree_based.py           # XGBoost, LightGBM, Random Forest
│   ├── deep_learning.py        # PyTorch-based models (LSTM, Transformer)
│   ├── ensemble.py             # Ensemble methods and stacking
│   └── factory.py              # Model factory for instantiation
├── training/
│   ├── __init__.py
│   ├── cross_validation.py     # Temporal cross-validation implementation
│   ├── hyperparameter_tuning.py # Optuna-based hyperparameter tuning
│   └── model_selection.py      # Model selection procedures
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # All evaluation metrics
│   ├── backtesting.py          # Backtesting framework
│   └── uncertainty.py          # Uncertainty quantification methods
├── persistence/
│   ├── __init__.py
│   ├── versioning.py           # Model versioning system
│   └── registry.py             # Model registry
└── pipelines/
    ├── __init__.py
    └── forecasting_pipeline.py # End-to-end forecasting pipeline
```

## 6. Integration with Feature Engineering Pipeline

### 6.1 Data Flow Integration

The SL model integrates with the feature engineering pipeline as follows:

```{mermaid}
graph TD
    A[Raw Market Data] --> B[Data Cleaning]
    B --> C[Feature Engineering]
    C --> D[SL Model Training]
    D --> E[Model Evaluation]
    E --> F[Model Selection]
    F --> G[Production Model]
    G --> H[RL Agent Input]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#f1f8e9
    style G fill:#e0f2f1
    style H fill:#fafafa
```

### 6.2 Interface with Feature Pipeline

```python
class SLFeatureAdapter:
    def __init__(self, feature_pipeline):
        self.feature_pipeline = feature_pipeline

    def prepare_training_data(self, raw_data, target_config):
        """Prepare data for SL model training"""
        # Apply feature engineering
        features = self.feature_pipeline.transform(raw_data)

        # Construct targets based on configuration
        targets = self._construct_targets(raw_data, target_config)

        # Apply temporal alignment to prevent data leakage
        aligned_features, aligned_targets = self._temporal_align(features, targets)

        return aligned_features, aligned_targets

    def prepare_prediction_data(self, raw_data):
        """Prepare data for SL model prediction"""
        # Apply feature engineering using fitted parameters
        features = self.feature_pipeline.transform(raw_data)
        return features
```

### 6.3 Deterministic Processing with Fixed Seeds

All components ensure deterministic processing:

```python
def set_all_seeds(seed=42):
    """Set seeds for all random number generators"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## 7. Dependencies

- Step 3 (Feature engineering components)
- PyTorch >= 2.0.0
- scikit-learn >= 1.3.0
- XGBoost >= 1.6.0
- LightGBM >= 3.3.0
- Optuna >= 3.0.0
- joblib >= 1.2.0
- pandas >= 2.0.0
- numpy >= 1.24.0

## 8. Estimated Runtime

4 hours for initial implementation and testing
