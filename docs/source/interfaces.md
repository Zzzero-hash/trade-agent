# Component Interfaces

This document defines the interfaces between components in the trading system architecture. These interfaces ensure loose coupling and clear data flow between modules.

## Data Flow Interfaces

### Market Data → Features Engineering

```python
# Input
{
    "raw_data": pd.DataFrame,  # Raw market data with OHLCV and other fields
    "symbols": List[str],      # List of asset symbols
    "timestamp": datetime      # Current timestamp
}

# Output
{
    "features": np.ndarray,    # Engineered features array
    "feature_names": List[str] # Names of features
}
```

### Features Engineering → SL Forecasters

```python
# Input
{
    "features": np.ndarray,    # Engineered features
    "feature_names": List[str] # Names of features
}

# Output
{
    "predictions": {
        "expected_returns": np.ndarray,  # E[r] predictions
        "volatility": np.ndarray,        # σ predictions
        "probabilities": np.ndarray      # Probability distributions
    }
}
```

### SL Forecasters → Trading Environment

```python
# Input
{
    "features": np.ndarray,              # Current features
    "sl_predictions": {
        "expected_returns": np.ndarray,  # SL E[r] predictions
        "volatility": np.ndarray,        # SL σ predictions
        "probabilities": np.ndarray      # SL probability distributions
    }
}

# Output (Observation)
{
    "features": np.ndarray,              # Current features
    "predictions": np.ndarray            # SL predictions
}
```

### Trading Environment → RL Agents

```python
# Input (Observation)
{
    "features": np.ndarray,              # Current features
    "predictions": np.ndarray            # SL predictions
}

# Output (Action)
np.ndarray  # Target positions for assets [-1, 1]
```

### RL Agents → Ensemble Combiner

```python
# Input
{
    "ppo_action": np.ndarray,  # PPO agent action
    "sac_action": np.ndarray   # SAC agent action
}

# Output
np.ndarray  # Ensemble combined action
```

### Ensemble Combiner → Execution System

```python
# Input
{
    "action": np.ndarray,      # Ensemble action
    "timestamp": datetime      # Current timestamp
}

# Output
{
    "orders": List[Dict],      # List of order objects
    "execution_time": datetime # Expected execution time
}
```

## Component Interface Definitions

### Market Data Interface

```python
class MarketDataProvider:
    def get_historical_data(self, symbols, start_date, end_date):
        """Retrieve historical market data"""
        pass

    def get_realtime_data(self, symbols):
        """Get real-time market data"""
        pass

    def get_asset_universe(self):
        """Get current asset universe"""
        pass

class MarketDataStorage:
    def store_data(self, data, source):
        """Store market data from source"""
        pass

    def retrieve_data(self, criteria):
        """Retrieve data based on criteria"""
        pass
```

### Features Engineering Interface

```python
class FeatureExtractor:
    def fit(self, data):
        """Fit feature extractor on data"""
        pass

    def transform(self, data):
        """Transform data to features"""
        pass

    def fit_transform(self, data):
        """Fit and transform data"""
        pass

class FeaturePipeline:
    def add_extractor(self, extractor):
        """Add feature extractor to pipeline"""
        pass

    def execute(self, data):
        """Execute feature pipeline"""
        pass
```

### SL Forecasters Interface

```python
class SLForecaster:
    def fit(self, X, y):
        """Fit forecaster on features and targets"""
        pass

    def predict(self, X):
        """Generate predictions for features"""
        pass

    def predict_proba(self, X):
        """Generate probability predictions"""
        pass

class ForecastingPipeline:
    def add_model(self, model, target):
        """Add model for specific target"""
        pass

    def train(self, data):
        """Train all models in pipeline"""
        pass

    def forecast(self, data):
        """Generate forecasts for data"""
        pass
```

### Trading Environment Interface

```python
class TradingEnvironment:
    def reset(self):
        """Reset environment to initial state"""
        pass

    def step(self, action):
        """Execute action and return (observation, reward, done, info)"""
        pass

    def render(self):
        """Render current state"""
        pass

class RewardFunction:
    def calculate(self, state, action, next_state):
        """Calculate reward for transition"""
        pass
```

### RL Agents Interface

```python
class RLAgent:
    def train(self, env, steps):
        """Train agent in environment"""
        pass

    def predict(self, observation):
        """Predict action for observation"""
        pass

    def save(self, path):
        """Save agent to path"""
        pass

    def load(self, path):
        """Load agent from path"""
        pass
```

### Ensemble Combiner Interface

```python
class EnsembleCombiner:
    def combine(self, actions):
        """Combine multiple actions into single action"""
        pass

    def update_weights(self, performance):
        """Update combination weights based on performance"""
        pass

    def apply_risk_controls(self, action):
        """Apply risk controls to action"""
        pass
```

### Execution System Interface

```python
class ExecutionSystem:
    def submit_orders(self, orders):
        """Submit orders for execution"""
        pass

    def cancel_order(self, order_id):
        """Cancel order by ID"""
        pass

    def get_positions(self):
        """Get current positions"""
        pass

    def get_account_info(self):
        """Get account information"""
        pass
```

## Data Contracts

### Market Data Contract

```python
# Required fields for market data
{
    "symbol": str,           # Asset identifier
    "timestamp": datetime,   # Data timestamp
    "open": float,           # Opening price
    "high": float,           # Highest price
    "low": float,            # Lowest price
    "close": float,          # Closing price
    "volume": float,         # Trading volume
    "adj_close": float       # Adjusted closing price
}
```

### Features Data Contract

```python
# Required structure for features
{
    "symbol": str,           # Asset identifier
    "timestamp": datetime,   # Feature timestamp
    "features": np.ndarray,  # Feature values (n_features,)
    "feature_names": List[str]  # Feature names
}
```

### Predictions Data Contract

```python
# Required structure for predictions
{
    "symbol": str,           # Asset identifier
    "timestamp": datetime,   # Prediction timestamp
    "expected_return": float,  # E[r] prediction
    "volatility": float,     # σ prediction
    "probabilities": np.ndarray  # Probability distribution
}
```

### Orders Data Contract

```python
# Required structure for orders
{
    "symbol": str,           # Asset identifier
    "quantity": float,       # Order quantity
    "order_type": str,       # "market", "limit", etc.
    "side": str,             # "buy", "sell"
    "price": float,          # Limit price (if applicable)
    "tif": str               # Time in force
}
```

## Configuration Interfaces

### Component Configuration

```python
class ComponentConfig:
    def __init__(self, config_dict):
        """Initialize with configuration dictionary"""
        pass

    def get(self, key, default=None):
        """Get configuration value"""
        pass

    def validate(self):
        """Validate configuration"""
        pass
```

### System Configuration

```python
class SystemConfig:
    def __init__(self, config_file):
        """Load configuration from file"""
        pass

    def get_component_config(self, component):
        """Get configuration for specific component"""
        pass

    def validate_integrity(self):
        """Validate configuration integrity across components"""
        pass
```

## Error Handling Interfaces

### Component Error Interface

```python
class ComponentError(Exception):
    def __init__(self, component, message, details=None):
        """Initialize component error"""
        self.component = component
        self.message = message
        self.details = details

    def __str__(self):
        return f"{self.component}: {self.message}"
```

### Data Quality Error Interface

```python
class DataQualityError(ComponentError):
    def __init__(self, component, message, data_issues=None):
        """Initialize data quality error"""
        super().__init__(component, message)
        self.data_issues = data_issues
```

## Performance Monitoring Interfaces

### Component Performance Interface

```python
class PerformanceMonitor:
    def start_timer(self, operation):
        """Start timing operation"""
        pass

    def end_timer(self, operation):
        """End timing operation"""
        pass

    def record_metric(self, metric, value):
        """Record performance metric"""
        pass
```

### System Health Interface

```python
class SystemHealth:
    def check_component(self, component):
        """Check component health"""
        pass

    def report_status(self):
        """Report overall system status"""
        pass
```

## Interface Versioning

All interfaces follow semantic versioning:

- MAJOR version for incompatible API changes
- MINOR version for backward-compatible functionality additions
- PATCH version for backward-compatible bug fixes

Interface versions are specified in component configurations and validated at runtime to ensure compatibility.
