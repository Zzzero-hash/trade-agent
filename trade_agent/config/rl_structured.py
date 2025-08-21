"""Structured (Pydantic) models for RL Hydra configs (ppo/sac).

These models provide runtime validation for the RL training configuration
under ``conf/rl``. The schema is intentionally permissive for algorithm
hyperparameters, treating each algo section (``ppo`` / ``sac``) as an
opaque mapping so new keys can be added without code changes.
"""
from __future__ import annotations

from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, PositiveInt, RootModel, model_validator


class EnvConfig(BaseModel):
    data_path: str = Field(..., description="Path to features parquet file")
    window_size: PositiveInt = Field(30, description="Observation window")
    validation_split: float = Field(0.2, ge=0.0, le=0.9)
    initial_capital: float = Field(100000.0, gt=0)
    transaction_cost: float = Field(0.001, ge=0.0, le=0.1)


class TrainingLoopConfig(BaseModel):
    total_timesteps: PositiveInt = Field(1000)
    eval_freq: PositiveInt = Field(10000)
    checkpoint_freq: PositiveInt = Field(50000)
    save_best: bool = True
    normalize_obs: bool = True
    normalize_reward: bool = True


class FeaturesConfig(BaseModel):
    input_dim: int | None = None
    hidden_layers: list[int] | None = None
    output_dim: int | None = None
    activation: str | None = None


class PPOSection(RootModel[dict[str, Any]]):  # free-form mapping
    pass


class SACSection(RootModel[dict[str, Any]]):
    pass


class OptimizationSection(BaseModel):
    enabled: bool = False
    n_trials: PositiveInt = Field(20)
    direction: Literal["maximize", "minimize"] = "maximize"
    metric: str = "sharpe"


class RLRootConfig(BaseModel):
    algo: Literal["ppo", "sac"]
    experiment: dict[str, Any] | None = None
    paths: dict[str, str] | None = None
    seed: int = 42
    device: str = Field("auto", pattern=r"^(cpu|cuda|auto)$")
    env: EnvConfig
    training: TrainingLoopConfig
    mlp_features: FeaturesConfig | None = None
    ppo: PPOSection | None = None
    sac: SACSection | None = None
    optimization: OptimizationSection | None = None

    @model_validator(mode="after")
    def _algo_section_present(self):  # type: ignore[no-untyped-def]
        if self.algo == "ppo" and self.ppo is None:
            raise ValueError("algo=ppo but ppo section missing")
        if self.algo == "sac" and self.sac is None:
            raise ValueError("algo=sac but sac section missing")
        return self


def _to_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(  # type: ignore[return-value]
            cfg, resolve=True
        )
    return cfg  # type: ignore[return-value]


def validate_rl_config(cfg: DictConfig | dict[str, Any]) -> RLRootConfig:
    data = _to_dict(cfg)
    # Determine active algo from defaults
    # Hydra sets 'algo' to a dict with single key (ppo or sac)
    # Hydra composes algo: ppo -> structure {algo: {ppo: {...}}}; flatten.
    if "algo" in data and isinstance(data["algo"], dict):
        for name in ("ppo", "sac"):
            if name in data["algo"]:
                # Promote nested section
                data[name] = data["algo"][name]
                data["algo"] = name
                break
    return RLRootConfig.model_validate(data)


__all__ = ["RLRootConfig", "validate_rl_config"]
