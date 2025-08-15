# Legacy System Sunset Plan ⚠️

![Hydra Migration](https://hydra.cc/img/logo.svg)

> **Warning**
> All legacy JSON configuration files and standalone training scripts will be permanently removed on **2025-12-31**.
> Migrate to Hydra-based configuration before this date to avoid service disruptions.

## Deprecated Components → Hydra Equivalents

| Legacy Component                | Hydra Replacement                | Migration Benefit                       |
| ------------------------------- | -------------------------------- | --------------------------------------- |
| `configs/*_config.json`         | `conf/model/*.yaml`              | Structured hierarchical configuration   |
| `scripts/train_single_model.py` | `scripts/train_sl_hydra.py`      | Built-in parameter overrides            |
| `configs/ppo_config.json`       | `conf/agent/ppo.yaml`            | Unified experiment management           |
| `configs/sac_config.json`       | `conf/agent/sac.yaml`            | Automated output directory creation     |
| Manual hyperparameter tuning    | `conf/hydra/sweeper/optuna.yaml` | Distributed hyperparameter optimization |

## Migration Guide for JSON Config Users

### 1. Install Hydra Dependencies

```bash
pip install hydra-core hydra-optuna-sweeper
```

### 2. Convert Config Formats

```python
# conversion_script.py
from hydra import initialize, compose
import json

with initialize(version_base="1.3", config_path="../conf"):
    cfg = compose(config_name="config")
    print(json.dumps(cfg, indent=2))  # Compare with legacy JSON
```

### 3. Update Training Commands

```diff
- python scripts/train_single_model.py --config configs/cnn_lstm_config.json
+ python scripts/train_sl_hydra.py model=cnn_lstm
```

### 4. Validate Configurations

```bash
python scripts/train_sl_hydra.py --cfg job model=cnn_lstm
```

## Phase Removal Timeline

| Phase                  | Date Range              | Impact                              |
| ---------------------- | ----------------------- | ----------------------------------- |
| **Deprecation Notice** | 2025-09-01 - 2025-10-31 | Warning logs on legacy config usage |
| **Feature Freeze**     | 2025-11-01 - 2025-11-30 | No new features for legacy systems  |
| **Hard Removal**       | 2025-12-01              | Complete removal of deprecated code |

## Hydra Documentation Links

- [Configuration Basics](conf/hydra/README.md)
- [Optuna Sweeper Setup](conf/hydra/sweeper/optuna.yaml)
- [Experiment Tracking](./outputs/README.md)

> **Note**
> For urgent migration support, reference our [Hydra adoption checklist](DVC_SETUP.md#hydra-integration)
