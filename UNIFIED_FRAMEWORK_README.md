# Unified Experimentation Framework

This document describes the enhanced modular design implemented to address the structural issues identified in the training pipeline audit.

## Overview

The unified framework consolidates fragmented hyperparameter optimization, standardizes cross-validation strategies, and provides comprehensive experiment tracking. It maintains backward compatibility while enabling sophisticated experimentation workflows.

## Key Components

### 1. Experiment Configuration (`src/experiments/config.py`)

Type-safe configuration system with validation:

```python
from src.experiments import ExperimentConfig, create_model_config

config = ExperimentConfig(
    experiment_name="my_experiment",
    data_config=DataConfig(data_path="data/sample_data.parquet"),
    model_configs=[create_model_config("ridge"), create_model_config("mlp")],
    optimization_config=OptimizationConfig(enabled=True, n_trials=100)
)
```

### 2. Unified Hyperparameter Tuner (`src/optimize/unified_tuner.py`)

Optuna-based optimization for both SL and RL models:

```python
from src.optimize import UnifiedHyperparameterTuner

tuner = UnifiedHyperparameterTuner("my_study")
results = tuner.optimize_sl_model("ridge", X, y, metric="sharpe")
```

### 3. Training Orchestrator (`src/experiments/orchestrator.py`)

End-to-end pipeline execution:

```python
from src.experiments import TrainingOrchestrator

orchestrator = TrainingOrchestrator(config)
results = orchestrator.run_full_pipeline()
```

### 4. Experiment Registry (`src/experiments/registry.py`)

Centralized experiment tracking:

```python
from src.experiments import ExperimentRegistry

registry = ExperimentRegistry()
experiment_id = registry.register_experiment(config)
registry.log_results(experiment_id, "ridge", {"val_sharpe": 0.85})
```

## Usage Examples

### Command Line Interface

```bash
# Simple experiment
python scripts/run_experiment.py --name my_test --model ridge --data data/sample_data.parquet

# Multi-model with optimization
python scripts/run_experiment.py --name optimization_test \
  --models ridge mlp cnn_lstm --optimize --n-trials 50

# Ensemble experiment
python scripts/run_experiment.py --name ensemble_test \
  --models ridge mlp --ensemble --ensemble-method weighted_average

# List experiments
python scripts/run_experiment.py --list-experiments

# Show experiment details
python scripts/run_experiment.py --show-experiment EXPERIMENT_ID
```

### Enhanced Hydra Integration

```bash
# Using new framework through Hydra
python scripts/train_sl_hydra.py \
  experiment.name=hydra_test \
  model=ridge \
  optimization.enabled=true \
  optimization.n_trials=20

# Ensemble with Hydra
python scripts/train_sl_hydra.py \
  experiment.name=ensemble_hydra \
  model=mlp \
  ensemble.enabled=true \
  ensemble.method=weighted_average
```

### Programmatic Usage

```python
from src.experiments import ExperimentConfig, TrainingOrchestrator

# Create configuration
config = ExperimentConfig.create_default("my_experiment")

# Run experiment
orchestrator = TrainingOrchestrator(config)
results = orchestrator.run_full_pipeline()

# Get experiment summary
summary = orchestrator.get_experiment_summary()
print(f"Best model: {summary['best_model']}")
```

## Configuration Files

### YAML Configuration

```yaml
# experiment_config.yaml
experiment:
  name: "my_experiment"

train:
  data_path: "data/sample_data.parquet"
  target: "mu_hat"

optimization:
  enabled: true
  n_trials: 100
  metric: "sharpe"

ensemble:
  enabled: true
  method: "weighted_average"
```

Load and run:

```python
config = ExperimentConfig.load("experiment_config.yaml")
orchestrator = TrainingOrchestrator(config)
results = orchestrator.run_full_pipeline()
```

## Advanced Features

### Purged Time-Series Cross-Validation

```python
from src.optimize import PurgedTimeSeriesCV

cv_strategy = PurgedTimeSeriesCV(
    n_splits=5,
    embargo_days=1,  # Gap between train/val
    purge_days=0     # Additional purging
)
```

### Experiment Comparison

```python
# Get best configuration across all experiments
registry = ExperimentRegistry()
best_config = registry.get_best_config(metric="val_sharpe")

# Export results for analysis
registry.export_results("experiment_results.csv")
```

### Model Artifacts Management

```python
# Models and artifacts are automatically tracked
summary = registry.get_experiment_summary(experiment_id)
artifacts = summary['artifacts']  # Paths to saved models
```

## Migration from Legacy System

### Backward Compatibility

The framework maintains compatibility with existing scripts:

- `scripts/train_sl_hydra.py` automatically detects and uses the new framework
- Legacy configurations continue to work
- Existing model factories and training pipelines are integrated

### Gradual Migration

1. **Start with CLI**: Use `scripts/run_experiment.py` for new experiments
2. **Convert configs**: Migrate existing JSON configs to new YAML format
3. **Enhance workflows**: Add optimization and ensemble capabilities
4. **Full integration**: Replace legacy scripts with unified framework

## Benefits

### Before Refactor

- Fragmented hyperparameter optimization (2 separate implementations)
- Inconsistent cross-validation strategies
- No centralized experiment tracking
- Manual configuration management
- Limited reproducibility

### After Refactor

- ✅ Unified Optuna-based optimization for all model types
- ✅ Standardized purged time-series cross-validation
- ✅ SQLite-based experiment registry with full tracking
- ✅ Type-safe configuration with validation
- ✅ Complete reproducibility with artifact management
- ✅ Enhanced CLI and programmatic interfaces
- ✅ Ensemble framework integration
- ✅ Backward compatibility with legacy system

## Examples

See `examples/unified_framework_demo.py` for comprehensive usage examples demonstrating all framework capabilities.

## Dependencies

The framework requires:

- `optuna` for hyperparameter optimization
- `omegaconf` for configuration management
- `pandas` for data handling
- `pydantic` for configuration validation (optional)
- `sqlite3` for experiment tracking (built-in)

Install with:

```bash
pip install optuna omegaconf pandas pydantic
```
