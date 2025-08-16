"""
Unified experiment configuration system for modular experimentation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_path: str
    target_column: str = "mu_hat"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    model_type: str
    model_params: dict[str, Any] = field(default_factory=dict)
    training_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation strategy."""
    strategy: str = "purged_time_series"  # or "temporal"
    n_splits: int = 5
    embargo_days: int = 1
    purge_days: int = 0
    gap: int = 0


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    enabled: bool = True
    n_trials: int = 100
    study_name: str = "default_study"
    storage: Optional[str] = None
    metric: str = "sharpe"
    sampler: str = "tpe"
    pruner: str = "median"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    enabled: bool = False
    method: str = "weighted_average"  # or "gating", "stacking"
    weights: Optional[dict[str, float]] = None
    gating_features: list[str] = field(default_factory=list)


@dataclass
class RewardConfig:
    """Configurable reward components for RL environments."""
    pnl_weight: float = 1.0
    risk_penalty_weight: float = 0.0
    turnover_penalty_weight: float = 0.0
    position_limit: float = 2.0
    transaction_cost_weight: float = 1.0
    risk_window: int = 20


@dataclass
class ExperimentConfig:
    """Unified experiment configuration."""
    experiment_name: str
    data_config: DataConfig
    model_configs: list[ModelConfig]
    cv_config: CrossValidationConfig
    optimization_config: OptimizationConfig
    ensemble_config: Optional[EnsembleConfig] = None
    random_state: int = 42
    output_dir: str = "experiments"
    save_models: bool = True
    use_enhanced_env: bool = False  # Use EnhancedTradingEnvironment for RL
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    def to_hydra_config(self) -> DictConfig:
        """Convert to Hydra configuration format."""
        config_dict = {
            'experiment_name': self.experiment_name,
            'random_state': self.random_state,
            'output_dir': self.output_dir,
            'save_model': self.save_models,
            'train': {
                'data_path': self.data_config.data_path,
                'target': self.data_config.target_column
            },
            'cv': {
                'n_splits': self.cv_config.n_splits,
                'gap': self.cv_config.gap
            },
            'optimization': {
                'enabled': self.optimization_config.enabled,
                'n_trials': self.optimization_config.n_trials,
                'metric': self.optimization_config.metric
            }
        }

        # Add model configurations
        if self.model_configs:
            # For single model, use the first one
            model_config = self.model_configs[0]
            config_dict['model_type'] = model_config.model_type
            config_dict['model_config'] = model_config.model_params

        return OmegaConf.create(config_dict)

    def validate(self) -> None:
        """Validate configuration consistency."""
        # Check data split ratios
        total_ratio = (
            self.data_config.train_ratio +
            self.data_config.val_ratio +
            self.data_config.test_ratio
        )
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Data split ratios must sum to 1.0, got {total_ratio}"
            )

        # Check model configurations
        if not self.model_configs:
            raise ValueError("At least one model configuration is required")

        # Validate model types
        supported_models = [
            'ridge', 'linear', 'mlp', 'cnn_lstm', 'transformer', 'garch',
            'ppo', 'sac'
        ]
        for model_config in self.model_configs:
            if model_config.model_type not in supported_models:
                raise ValueError(
                    f"Unsupported model type: {model_config.model_type}"
                )

        # Check CV configuration
        if self.cv_config.n_splits < 2:
            raise ValueError("Number of CV splits must be at least 2")

        # Check optimization configuration
        if self.optimization_config.n_trials < 1:
            raise ValueError(
                "Number of optimization trials must be at least 1"
            )

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            'experiment_name': self.experiment_name,
            'data_config': {
                'data_path': self.data_config.data_path,
                'target_column': self.data_config.target_column,
                'train_ratio': self.data_config.train_ratio,
                'val_ratio': self.data_config.val_ratio,
                'test_ratio': self.data_config.test_ratio
            },
            'model_configs': [
                {
                    'model_type': mc.model_type,
                    'model_params': mc.model_params,
                    'training_params': mc.training_params
                }
                for mc in self.model_configs
            ],
            'cv_config': {
                'strategy': self.cv_config.strategy,
                'n_splits': self.cv_config.n_splits,
                'embargo_days': self.cv_config.embargo_days,
                'purge_days': self.cv_config.purge_days,
                'gap': self.cv_config.gap
            },
            'optimization_config': {
                'enabled': self.optimization_config.enabled,
                'n_trials': self.optimization_config.n_trials,
                'study_name': self.optimization_config.study_name,
                'storage': self.optimization_config.storage,
                'metric': self.optimization_config.metric,
                'sampler': self.optimization_config.sampler,
                'pruner': self.optimization_config.pruner
            },
            'random_state': self.random_state,
            'output_dir': self.output_dir,
            'save_models': self.save_models
        }

        if self.ensemble_config:
            config_dict['ensemble_config'] = {
                'enabled': self.ensemble_config.enabled,
                'method': self.ensemble_config.method,
                'weights': self.ensemble_config.weights,
                'gating_features': self.ensemble_config.gating_features
            }

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        filepath = Path(filepath)
        with open(filepath) as f:
            config_dict = yaml.safe_load(f)

        # Parse data config
        data_config = DataConfig(**config_dict['data_config'])

        # Parse model configs
        model_configs = [
            ModelConfig(**mc) for mc in config_dict['model_configs']
        ]

        # Parse CV config
        cv_config = CrossValidationConfig(**config_dict['cv_config'])

        # Parse optimization config
        optimization_config = OptimizationConfig(
            **config_dict['optimization_config']
        )

        # Parse ensemble config if present
        ensemble_config = None
        if 'ensemble_config' in config_dict:
            ensemble_config = EnsembleConfig(**config_dict['ensemble_config'])

        return cls(
            experiment_name=config_dict['experiment_name'],
            data_config=data_config,
            model_configs=model_configs,
            cv_config=cv_config,
            optimization_config=optimization_config,
            ensemble_config=ensemble_config,
            random_state=config_dict.get('random_state', 42),
            output_dir=config_dict.get('output_dir', 'experiments'),
            save_models=config_dict.get('save_models', True)
        )

    @classmethod
    def create_default(
        cls, experiment_name: str = "default_experiment"
    ) -> 'ExperimentConfig':
        """Create a default experiment configuration."""
        return cls(
            experiment_name=experiment_name,
            data_config=DataConfig(
                data_path="data/sample_data.parquet",
                target_column="mu_hat"
            ),
            model_configs=[
                ModelConfig(
                    model_type="ridge",
                    model_params={"alpha": 1.0, "random_state": 42}
                )
            ],
            cv_config=CrossValidationConfig(),
            optimization_config=OptimizationConfig()
        )


def create_model_config(model_type: str, **kwargs) -> ModelConfig:
    """Helper function to create model configurations."""
    default_params = {
        'ridge': {'alpha': 1.0, 'random_state': 42},
        'linear': {'random_state': 42},
        'mlp': {
            'hidden_layer_sizes': (100, 50),
            'learning_rate': 'adaptive',
            'random_state': 42,
            'max_iter': 500
        },
        'cnn_lstm': {
            'cnn_channels': [32, 64],
            'lstm_hidden_size': 64,
            'dropout': 0.2,
            'sequence_length': 30,
            'random_state': 42
        }
    }

    model_params = default_params.get(model_type, {})
    model_params.update(kwargs)

    return ModelConfig(
        model_type=model_type,
        model_params=model_params
    )
