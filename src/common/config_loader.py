import json
import warnings
from pathlib import Path
from typing import Any, TypeVar, Union

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)


class ConfigValidationError(ValueError):
    """Raised when configuration fails validation"""


def load_config(
    config_input: Union[str, dict[str, Any]], config_type: type[T]
) -> T:
    """
    Load and validate configuration from either a dict or JSON file path.

    Args:
        config_input: Configuration dict or path to JSON config file
        config_type: Type[T],  # The Pydantic model type to validate against
        # (e.g., SLConfig, RLConfig)

    Returns:
        Validated Pydantic model instance

    Raises:
        ConfigValidationError: For invalid configuration structure
        FileNotFoundError: If provided path doesn't exist
        JSONDecodeError: For invalid JSON files
    """
    raw_config = _load_raw_config(config_input)
    return _validate_config(raw_config, config_type)


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


def _validate_config(raw: dict[str, Any], config_type: type[T]) -> T:
    """
    Validate configuration structure against the specified Pydantic schema.
    """
    try:
        return config_type(**raw)
    except ValidationError as e:
        raise ConfigValidationError(
            f"Configuration validation failed for {config_type.__name__}: "
            f"{e.errors()}"
        ) from e


def hydrate_pipeline_config(cfg: DictConfig, config_type: type[T]) -> T:
    """
    Convert Hydra DictConfig to a validated Pydantic config model.

    Args:
        cfg: Hydra configuration object
        config_type: Type[T],  # The Pydantic model type to validate against
        # (e.g., SLConfig, RLConfig)

    Returns:
        Validated Pydantic model instance
    """
    raw: dict[str, Any] = OmegaConf.to_container(  # type: ignore
        cfg, resolve=True)
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"Expected dict config, got {type(raw)}")
    return _validate_config(raw, config_type)
