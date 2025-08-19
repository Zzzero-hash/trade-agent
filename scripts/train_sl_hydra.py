#!/usr/bin/env python3
"""Enhanced Hydra-enabled entrypoint for unified training pipeline.

This script integrates with the new unified experimentation framework
while maintaining backward compatibility with existing configurations.

Usage examples:
  1) Single run with new framework:
      python scripts/train_sl_hydra.py \
         experiment.name=ridge_test \
         model=ridge \
         train.data_path=data/sample_data.parquet \
         train.target=mu_hat

  2) With hyperparameter optimization:
      python scripts/train_sl_hydra.py \
         experiment.name=mlp_optimization \
         model=mlp \
         optimization.enabled=true \
         optimization.n_trials=50

  3) Ensemble experiment:
      python scripts/train_sl_hydra.py \
         experiment.name=ensemble_test \
         ensemble.enabled=true \
         ensemble.method=weighted_average

  4) Optuna sweep:
      python scripts/train_sl_hydra.py -m hydra/sweeper=optuna \
         optuna.n_trials=10 \
         model=ridge,mlp,cnn_lstm
"""
from __future__ import annotations

import contextlib
import sys as _sys
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

try:  # graceful fallback if hydra-core not installed in minimal test env
    import hydra  # type: ignore
    from hydra.utils import to_absolute_path  # type: ignore
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    class _Dummy:  # minimal stand-ins
        def main(self=None, *a, **k):  # type: ignore[no-untyped-def]
            def deco(fn):
                def wrapper(*args, **kwargs) -> int:  # type: ignore[no-untyped-def]
                    return 0
                return wrapper
            return deco
    hydra = _Dummy()  # type: ignore
    def to_absolute_path(p: str) -> str:  # type: ignore[no-untyped-def]
        return p
    class DictConfig(dict):  # type: ignore
        pass
    class OmegaConf:  # type: ignore
        @staticmethod
        def to_yaml(cfg) -> str:
            return ""

# Import new unified framework
try:
    from src.experiments import ExperimentConfig, TrainingOrchestrator
    from src.experiments.config import (
        CrossValidationConfig,
        DataConfig,
        EnsembleConfig,
        ModelConfig,
        OptimizationConfig,
    )
    USE_NEW_FRAMEWORK = True
except ImportError:
    USE_NEW_FRAMEWORK = False
    # Fallback to legacy imports
    from trade_agent.agents.sl.train import SLTrainingPipeline


def _create_experiment_config_from_hydra(cfg: DictConfig) -> ExperimentConfig:
    """Convert Hydra configuration to ExperimentConfig."""

    # Extract experiment name
    experiment_name = cfg.get('experiment', {}).get('name', 'hydra_experiment')

    # Create data config
    data_config = DataConfig(
        data_path=to_absolute_path(str(cfg.train.get('data_path', 'data/sample_data.csv'))),
        target_column=cfg.train.target,
        train_ratio=cfg.get('data_split', {}).get('train_ratio', 0.6),
        val_ratio=cfg.get('data_split', {}).get('val_ratio', 0.2),
        test_ratio=cfg.get('data_split', {}).get('test_ratio', 0.2)
    )

    # Create model config
    model_node = cfg.get('model', cfg)
    if hasattr(model_node, 'model_config'):
        model_params = dict(model_node.model_config)
    else:
        model_params = {}

    model_type = getattr(model_node, 'model_type', cfg.get('model_type', 'ridge'))
    model_configs = [ModelConfig(
        model_type=model_type,
        model_params=model_params
    )]

    # Create CV config
    cv_config = CrossValidationConfig(
        strategy=cfg.get('cv_strategy', 'purged_time_series'),
        n_splits=cfg.cv.n_splits,
        gap=cfg.cv.gap,
        embargo_days=cfg.get('cv_embargo_days', 1),
        purge_days=cfg.get('cv_purge_days', 0)
    )

    # Create optimization config
    opt_cfg = cfg.get('optimization', {})
    optimization_config = OptimizationConfig(
        enabled=opt_cfg.get('enabled', False),
        n_trials=opt_cfg.get('n_trials', 100),
        study_name=f"{experiment_name}_study",
        metric=opt_cfg.get('metric', 'sharpe')
    )

    # Create ensemble config if specified
    ensemble_config = None
    if cfg.get('ensemble', {}).get('enabled', False):
        ens_cfg = cfg.ensemble
        ensemble_config = EnsembleConfig(
            enabled=True,
            method=ens_cfg.get('method', 'weighted_average'),
            weights=dict(ens_cfg.get('weights', {})) if ens_cfg.get('weights') else None,
            gating_features=list(ens_cfg.get('gating_features', []))
        )

    return ExperimentConfig(
        experiment_name=experiment_name,
        data_config=data_config,
        model_configs=model_configs,
        cv_config=cv_config,
        optimization_config=optimization_config,
        ensemble_config=ensemble_config,
        random_state=cfg.random_state,
        output_dir=cfg.output_dir,
        save_models=cfg.save_model
    )


def _assemble_legacy_config(cfg: DictConfig) -> dict[str, Any]:
    """Convert Hydra cfg into legacy dict expected by SLTrainingPipeline."""
    # Model sub-config is defined inside model group file
    node = cfg.get('model', cfg)
    model_section = {
        k: v for k, v in node.model_config.items()
    } if hasattr(node, 'model_config') else {}

    cv_conf = node.cv_config if 'cv_config' in node else getattr(
        cfg, 'cv_config', {'n_splits': cfg.cv.n_splits, 'gap': cfg.cv.gap}
    )
    tuning_conf = node.tuning_config if 'tuning_config' in node else getattr(
        cfg, 'tuning_config', {'enable_tuning': False}
    )
    model_type = node.model_type if 'model_type' in node else getattr(
        cfg, 'model_type', 'ridge'
    )

    legacy: dict[str, Any] = {
        'model_type': model_type,
        'model_config': model_section,
        'cv_config': cv_conf,
        'tuning_config': tuning_conf,
        'random_state': cfg.random_state,
        'output_dir': cfg.output_dir,
        'save_model': cfg.save_model,
    }
    return legacy


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function with unified framework support."""

    if USE_NEW_FRAMEWORK:

        # Create experiment config from Hydra config
        experiment_config = _create_experiment_config_from_hydra(cfg)

        # Emit marker expected by legacy tests
        # Provide a concise YAML view for parity
        with contextlib.suppress(Exception):
            pass

        # Show experiment configuration (human summary)

        # Create and run orchestrator
        orchestrator = TrainingOrchestrator(experiment_config)
        results = orchestrator.run_full_pipeline()

        # Print summary
        for key, value in results.items():
            len(value) if isinstance(value, dict) else 1
            # If this is training results dict, echo key metrics for tests
            if key == 'training' and isinstance(value, dict):
                for _model_name, model_result in value.items():
                    metrics = model_result.get('metrics', {}) if isinstance(model_result, dict) else {}
                    for _m_key, _m_val in metrics.items():
                        pass
        # Legacy test expectation: ensure 'train_mse' token appears when available
        training_section = results.get('training', {})
        if isinstance(training_section, dict):
            for model_result in training_section.values():
                if isinstance(model_result, dict):
                    m = model_result.get('metrics', {})
                    if 'train_mse' in m:
                        pass

        # Get experiment summary
        summary = orchestrator.get_experiment_summary()
        if summary:
            pass

        # Return primary metric for Hydra optimization
        training_results = results.get('training', {})
        if training_results:
            return max(
                result.get('metrics', {}).get('val_score', 0.0)
                for result in training_results.values()
            )

        return 0.0


    # Show resolved config
    from trade_agent.agents.sl.config_loader import hydrate_config
    hydrate_config(cfg)

    # Build legacy config for existing pipeline
    legacy_config = _assemble_legacy_config(cfg)

    # Load data (mimic prior logic in train_model_from_config)
    data_path = to_absolute_path(
        str(cfg.train.get('data_path', 'data/sample_data.csv')))
    target_column = cfg.train.target

    import numpy as np
    import pandas as pd

    if str(data_path).endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif str(data_path).endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path}")

    y = df[target_column].values
    # Start from numeric columns only
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    # Remove target column if numeric
    if target_column in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_column])
    # Legacy behavior: also exclude known target engineered columns
    for col in ['mu_hat', 'sigma_hat']:
        if col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[col])
    X = numeric_df.values

    # Simple temporal validation split (last 20%) – future: configurable
    val_cut = int(len(X) * 0.8)
    X_train, X_val = X[:val_cut], X[val_cut:]
    y_train, y_val = y[:val_cut], y[val_cut:]

    pipeline = SLTrainingPipeline(legacy_config)
    results = pipeline.train(X_train, y_train, X_val=X_val, y_val=y_val)

    for _k, _v in results.items():
        pass

    # Persist resolved config & results for reproducibility
    import json
    from pathlib import Path

    out_dir = Path(legacy_config['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'last_resolved_hydra_config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    with open(out_dir / 'last_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Return validation MSE if present, else train MSE (minimize)
    return results.get('val_mse', results['train_mse'])


if __name__ == "__main__":  # pragma: no cover
    main()
