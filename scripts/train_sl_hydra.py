#!/usr/bin/env python3
"""Hydra-enabled entrypoint for supervised learning model training.

Usage examples:
  1) Single run (ridge):
      python scripts/train_sl_hydra.py \
         model=ridge \
         train.data_path=data/sample_data.parquet \
         train.target=mu_hat

  2) Switch model:
      python scripts/train_sl_hydra.py model=mlp train.target=mu_hat

  3) Simple sweep without Optuna (cartesian):
      python scripts/train_sl_hydra.py -m \
         model=ridge,linear random_state=42,1337

  4) Optuna sweep (enable sweeper, define search space):
      python scripts/train_sl_hydra.py -m hydra/sweeper=optuna \
         optuna.n_trials=10 \
         hydra.sweeper.search_space.model.alpha.low=0.0001 \
         hydra.sweeper.search_space.model.alpha.high=10.0 \
         hydra.sweeper.search_space.model.alpha.type=float \
         model=ridge

The script adapts train_model_from_config() expectations by synthesizing a
legacy dict from Hydra's composed configuration.
"""
from __future__ import annotations

import json
import sys as _sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# Reuse existing training function
from src.sl.train import SLTrainingPipeline


def _assemble_legacy_config(cfg: DictConfig) -> dict[str, Any]:
    """Convert Hydra cfg into legacy dict expected by SLTrainingPipeline.

    We keep keys consistent with prior JSON files so downstream remains
    unchanged.
    """
    # Model sub-config is defined inside model group file
    # (e.g. conf/model/ridge.yaml)
    node = cfg.get('model', cfg)
    model_section = {
        k: v for k, v in node.model_config.items()  # type: ignore[attr-defined]
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
    # Show resolved config
    from src.sl.config_loader import hydrate_config
    print("=== Validated Config ===")
    validated_config = hydrate_config(cfg)
    print(validated_config.model_dump_json(indent=2))

    # Build legacy config for existing pipeline
    legacy_config = _assemble_legacy_config(cfg)

    # Load data (mimic prior logic in train_model_from_config)
    data_path = to_absolute_path(
        str(cfg.train.get('data_path', 'data/sample_data.csv')))
    target_column = cfg.train.target

    import pandas as pd
    if str(data_path).endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif str(data_path).endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path}")

    import numpy as np
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

    print("=== Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Persist resolved config & results for reproducibility
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
