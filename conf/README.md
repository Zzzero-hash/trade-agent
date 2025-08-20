# Configuration Documentation

This document describes the configuration structure and migration status for the Trade Agent project.

## Directory Structure

```
conf/
├── config.yaml              # Main Hydra configuration
├── enhanced_config.yaml     # Enhanced configuration with overrides
├── hydra/                   # Hydra framework settings
│   └── sweeper/
│       └── optuna.yaml      # Optuna hyperparameter optimization settings
├── model/                   # Model configurations
│   ├── _base.yaml          # Base model template (NEW)
│   ├── mlp.yaml            # MLP model configuration
│   ├── linear.yaml         # Linear model configuration
│   ├── garch.yaml          # GARCH model configuration
│   ├── transformer.yaml    # Transformer model configuration
│   ├── cnn_lstm.yaml       # CNN-LSTM model configuration
│   └── ridge.yaml          # Ridge regression configuration
├── rl/                     # Reinforcement Learning configurations
│   ├── config.yaml         # RL base configuration
│   ├── rl_optuna.yaml      # RL hyperparameter optimization
│   ├── algo/               # Algorithm-specific configurations
│   │   ├── sac.yaml        # SAC algorithm parameters
│   │   ├── ppo.yaml        # PPO algorithm parameters
│   │   └── sac_optuna.yaml # SAC optimization parameters
│   ├── features/           # Feature extraction configurations
│   │   └── mlp.yaml        # MLP feature extractor
│   └── training/           # Training configurations
│       └── default.yaml    # Default training parameters
├── pipeline/               # Pipeline configurations (CONSOLIDATED)
│   ├── rl_pipeline.yaml    # RL training pipeline
│   ├── sl_pipeline.yaml    # Supervised learning pipeline
│   ├── sample_pipeline.yaml # Sample pipeline example
│   └── simple_test.yaml    # Simple test pipeline
└── search_space/           # Hyperparameter search spaces
    ├── mlp_basic.yaml      # Basic MLP search space
    ├── mlp_extended.yaml   # Extended MLP search space
    ├── garch_core.yaml     # GARCH search space
    ├── transformer_core.yaml # Transformer search space
    ├── sac_core.yaml       # SAC algorithm search space
    ├── ppo_core.yaml       # PPO algorithm search space
    ├── ridge_alpha.yaml    # Ridge alpha search space
    ├── cnn_lstm_seq.yaml   # CNN-LSTM sequence search space
    └── README.md           # Search space documentation
```

## Migration Status

### ✅ Completed Migrations

1. **JSON Configs Removal** (High Impact, Zero Risk)
   - ✅ Removed `configs/sac_config.json` (duplicate of `conf/rl/algo/sac.yaml`)
   - ✅ Removed `configs/mlp_config.json` (duplicate of `conf/model/mlp.yaml`)
   - ✅ Removed `configs/linear_config.json` (duplicate of `conf/model/linear.yaml`)
   - ✅ Removed `configs/garch_config.json` (duplicate of `conf/model/garch.yaml`)
   - ✅ Removed `configs/transformer_config.json` (duplicate of `conf/model/transformer.yaml`)

2. **Pipeline Directory Consolidation**
   - ✅ Moved all files from `conf/pipelines/` to `conf/pipeline/`
   - ✅ Removed empty `conf/pipelines/` directory
   - ✅ Updated all internal references

3. **Base Model Template**
   - ✅ Created `conf/model/_base.yaml` with shared structure
   - ✅ Template includes common fields: `model_config`, `cv_config`, `tuning_config`
   - ✅ All model configs updated to inherit from base template

4. **Search Space Standardization**
   - ✅ Applied naming pattern: `{component}_{purpose}_{variant}.yaml`
   - ✅ Examples:
     - `mlp_basic.yaml` → Basic MLP hyperparameters
     - `mlp_extended.yaml` → Extended MLP hyperparameters
     - `garch_core.yaml` → Core GARCH parameters
     - `transformer_core.yaml` → Core Transformer parameters

## Hydra Configuration Hierarchy

The project uses Hydra for configuration management with the following hierarchy:

1. **Base Configuration** (`config.yaml`)
   - Contains default values and structure
   - Can be overridden by more specific configs

2. **Model-Specific Configs** (`model/`)
   - Inherit from `_base.yaml`
   - Define model-specific parameters
   - Can be selected via `--config-name=model/{model_name}`

3. **Algorithm Configs** (`rl/algo/`)
   - Define RL algorithm hyperparameters
   - Used for training specific algorithms

4. **Pipeline Configs** (`pipeline/`)
   - Define end-to-end training pipelines
   - Combine models, algorithms, and training procedures

## Usage Examples

### Basic Model Training

```bash
python train_model.py model=mlp
```

### RL Training with Specific Algorithm

```bash
python train_rl.py rl/algo=sac training=default
```

### Hyperparameter Optimization

```bash
python optimize_model.py hydra/sweeper=optuna search_space=mlp_basic
```

### Custom Pipeline

```bash
python run_pipeline.py pipeline=rl_pipeline
```

## Configuration Patterns

### Model Configuration Pattern

```yaml
# In conf/model/_base.yaml
defaults:
  - _self_

model_type: ${model_type}
random_state: ${random_state}

model_config:
  # Model-specific parameters

cv_config:
  n_splits: 3
  gap: 10

tuning_config:
  enable_tuning: false
  scoring_metric: neg_mean_squared_error
  param_grid: {}
  n_trials: 10
```

### Search Space Pattern

```yaml
# In conf/search_space/{component}_{purpose}_{variant}.yaml
model_config:
  hidden_sizes:
    - [64, 32]
    - [128, 64, 32]
  learning_rate: [0.001, 0.01]
  dropout: [0.1, 0.2, 0.3]
```

## Best Practices

1. **Use YAML for New Configurations**
   - YAML configs are more maintainable than JSON
   - Support for comments and references
   - Better integration with Hydra

2. **Follow Naming Conventions**
   - Model configs: `model/{model_name}.yaml`
   - Search spaces: `{component}_{purpose}_{variant}.yaml`
   - Pipeline configs: `pipeline/{pipeline_name}.yaml`

3. **Inherit from Base Templates**
   - Use `_base.yaml` for common structure
   - Override only what's necessary
   - Maintain consistency across configs

4. **Document Configuration Changes**
   - Update this README when adding new configs
   - Document parameter meanings and valid ranges
   - Include usage examples

5. **Test Configurations**
   - Verify configs load without errors
   - Test with small datasets before production
   - Validate parameter ranges and types

## Deprecated Features

- **JSON Configurations**: All JSON configs have been removed and replaced with YAML equivalents
- **Old Pipeline Path**: `conf/pipelines/` has been consolidated into `conf/pipeline/`
- **Unstructured Configs**: All configs now follow standardized patterns

## Future Improvements

1. Add configuration validation schemas
2. Implement configuration version control
3. Add more comprehensive documentation for each config file
4. Create automated tests for configuration loading
5. Add configuration examples for common use cases
