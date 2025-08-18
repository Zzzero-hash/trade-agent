import json
import warnings
from pathlib import Path
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from src.common.config import CoreConfig


class ConfigValidationError(ValueError):
    """Raised when configuration fails validation"""


class SLConfig(CoreConfig):
    """Supervised Learning configuration schema."""
    model_type: str
    model_settings: dict[str, Any]
    cv_config: dict[str, Any]
    tuning_config: dict[str, Any]
    data_path: str | None = None  # optional path to originating feature file


def load_config(config_input: Union[str, dict[str, Any]]) -> SLConfig:
    """Load and validate configuration from either a dict or JSON file path."""
    raw_config = _load_raw_config(config_input)
    return _validate_config(raw_config)


def _load_raw_config(input_data: Union[str, dict[str, Any]]) -> dict[str, Any]:
    """Handle both path and dict inputs with deprecation warning."""
    if isinstance(input_data, str):
        msg = "Path-based configs are deprecated - use Hydra or pass dicts"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return _load_config_from_path(input_data)
    return input_data


def _load_config_from_path(config_path: str) -> dict[str, Any]:
    """Load JSON config with path validation and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        return json.load(f)


def _validate_config(raw: dict[str, Any]) -> SLConfig:
    """Validate configuration structure against SLConfig schema."""
    try:
        return SLConfig(**raw)
    except ValidationError as e:
        raise ConfigValidationError(
            f"Configuration validation failed: {e.errors()}"
        ) from e


def hydrate_config(cfg: DictConfig) -> SLConfig:
    """Convert Hydra config to validated SLConfig."""
    raw: dict[str, Any] = OmegaConf.to_container(  # type: ignore
        cfg, resolve=True)
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"Expected dict config, got {type(raw)}")
    return _validate_config(raw)
