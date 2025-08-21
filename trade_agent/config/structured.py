"""Structured (Pydantic) configuration models for Hydra configs.

These models provide runtime validation of composed Hydra configurations.
They intentionally mirror the keys present in the existing YAML files under
`conf/` while remaining permissive for model-specific hyperparameters.

Usage:
    from omegaconf import DictConfig, OmegaConf
    from trade_agent.config.structured import validate_root_config

    def main(cfg: DictConfig):
        root = validate_root_config(cfg)  # raises ValidationError on problems
        ...

Design notes:
* We keep model.model_config as a free-form mapping to avoid duplicating every
  algorithm's parameter schema. Light constraints (e.g. positive ints) can be
  incrementally added once stabilized.
* Separate sections (optimization, ensemble, experiment, tuning) are optional
  so legacy minimal configs remain valid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class TrainConfig(BaseModel):
    data_path: str = Field(
        ..., description="Path to training dataset (csv/parquet)"
    )
    target: str = Field(
        ..., description="Target column name"
    )

    @property
    def path(self) -> Path:  # convenience accessor
        return Path(self.data_path)


class DataSplitConfig(BaseModel):
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    @model_validator(mode="after")
    def _ratios_sum(self):  # type: ignore[no-untyped-def]
        total = self.train_ratio + self.val_ratio + self.test_ratio
        # Allow minor FP rounding error
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Data split ratios must sum to 1.0 (got {total:.6f})"
            )
        return self


class CVConfig(BaseModel):
    n_splits: PositiveInt = Field(
        ..., description="Number of CV splits"
    )
    gap: int = Field(
        0, ge=0, description="Temporal gap between train and test"
    )


class OptimizationConfig(BaseModel):
    enabled: bool = False
    n_trials: PositiveInt = 50
    metric: str = "sharpe"


class EnsembleConfig(BaseModel):
    enabled: bool = False
    method: str = Field(
        "weighted_average", description="Ensemble reduction method"
    )
    weights: dict[str, float] | None = None
    gating_features: list[str] = Field(default_factory=list)


class TuningConfig(BaseModel):
    enable_tuning: bool = False
    scoring_metric: str = "neg_mean_squared_error"
    param_grid: dict[str, Any] | None = None
    n_trials: PositiveInt | None = None


class ModelGroup(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='allow')
    model_type: str = Field(
        ..., description="Identifier for model family (ridge, mlp, etc.)"
    )
    # Primary canonical store for params
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Free-form model hyperparameters"
    )
    legacy_model_config: dict[str, Any] | None = Field(
        default=None,
        alias="model_config",
        description="Legacy model_config field",
    )
    cv_config: CVConfig | None = None
    tuning_config: TuningConfig | None = None

    @model_validator(mode="after")
    def _merge_legacy(self):  # type: ignore[no-untyped-def]
        if self.legacy_model_config and not self.parameters:
            self.parameters = dict(self.legacy_model_config)
        return self


class ExperimentSection(BaseModel):
    name: str = Field("default_experiment", description="Experiment name")


class RootConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')
    # Defaults section omitted (Hydra internal)
    model: ModelGroup | None = None
    model_type: str | None = Field(
        None,
        description=(
            "Legacy top-level model_type (deprecated in favor of model group)"
        ),
    )
    train: TrainConfig
    data_split: DataSplitConfig | None = None
    cv: CVConfig
    cv_strategy: str | None = None
    cv_embargo_days: int | None = Field(ge=0, default=None)
    cv_purge_days: int | None = Field(ge=0, default=None)
    optimization: OptimizationConfig | None = None
    ensemble: EnsembleConfig | None = None
    experiment: ExperimentSection | None = None
    random_state: int = 42
    output_dir: str = Field("models", description="Directory for outputs")
    save_model: bool = True
    tuning: TuningConfig | None = None  # legacy top-level tuner block
    optuna: dict[str, Any] | None = None  # hydra/optuna sweeper settings

    @model_validator(mode="after")
    def _ensure_model_type(self):  # type: ignore[no-untyped-def]
        if not self.model and not self.model_type:
            raise ValueError(
                "Either 'model' group or legacy 'model_type' must be provided"
            )
        return self


def _to_container(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(  # type: ignore[return-value]
            cfg, resolve=True
        )
    return cfg  # type: ignore[return-value]


def validate_root_config(cfg: DictConfig | dict[str, Any]) -> RootConfig:
    """Validate a composed Hydra config and return the structured object.

    Raises:
        ValidationError if the config is invalid.
    """
    data = _to_container(cfg)
    return RootConfig.model_validate(data)


__all__ = [
    "TrainConfig",
    "DataSplitConfig",
    "CVConfig",
    "OptimizationConfig",
    "EnsembleConfig",
    "TuningConfig",
    "ModelGroup",
    "ExperimentSection",
    "RootConfig",
    "validate_root_config",
]
