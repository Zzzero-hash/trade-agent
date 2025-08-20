"""Tests for Hydra structured config validation.

Covers: successful validation of base config, ratio sum check, and model
presence requirement. These tests exercise the pydantic models in
`trade_agent.config.structured`.
"""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from trade_agent.config.structured import (
    RootConfig,
    validate_root_config,
)


def _load_conf_text() -> str:
    return """
    train:
      data_path: data/sample_data.csv
      target: close
    cv:
      n_splits: 5
      gap: 0
    model_type: ridge
    random_state: 1
    output_dir: models
    save_model: true
    """


def test_validate_minimal_config_ok() -> None:
    cfg = OmegaConf.create(_load_conf_text())
    root = validate_root_config(cfg)  # type: ignore[arg-type]
    assert isinstance(root, RootConfig)
    assert root.model_type == "ridge"


def test_data_split_ratio_validation() -> None:
    bad = OmegaConf.create(
        {
            "train": {"data_path": "data/sample.csv", "target": "y"},
            "cv": {"n_splits": 3, "gap": 0},
            "model_type": "ridge",
            "data_split": {
                "train_ratio": 0.7,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            },
        }
    )
    with pytest.raises(Exception):
        validate_root_config(bad)


def test_requires_model_or_model_type() -> None:
    missing = OmegaConf.create(
        {
            "train": {"data_path": "d.csv", "target": "y"},
            "cv": {"n_splits": 3, "gap": 0},
        }
    )
    with pytest.raises(Exception):
        validate_root_config(missing)
