from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class CoreConfig(BaseModel):
    """Base configuration schema for all pipelines."""
    model_config = ConfigDict(extra='forbid')

    random_state: int = 42
    output_dir: str = "models"
    save_model: bool = True


class RLConfig(CoreConfig):
    """Reinforcement Learning configuration schema."""
    agent_type: str
    agent_settings: dict[str, Any]
    env_settings: dict[str, Any]


class EnsembleConfig(CoreConfig):
    """Ensemble Learning configuration schema."""
    constituent_models: list[str]
    ensemble_strategy: dict[str, Any]
