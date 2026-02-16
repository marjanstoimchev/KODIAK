"""Configuration management for M³-Net."""

import yaml
from pathlib import Path
from typing import Any, Dict, Union
from dataclasses import dataclass, field


class Config:
    """Configuration container with dot notation access."""

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Config object with dot notation access
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Union[Config, Dict], save_path: Union[str, Path]):
    """
    Save configuration to YAML file.

    Args:
        config: Config object or dictionary
        save_path: Path to save config.yaml
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def override_config(config: Config, overrides: Dict[str, Any]) -> Config:
    """
    Override configuration values.

    Args:
        config: Base configuration
        overrides: Dictionary of overrides (supports dot notation keys)

    Returns:
        Updated configuration

    Example:
        override_config(config, {"model.num_motifs": 32, "training.max_epochs": 200})
    """
    config_dict = config.to_dict()

    for key, value in overrides.items():
        # Support dot notation
        keys = key.split('.')
        current = config_dict

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    return Config(config_dict)


def validate_and_load(config_path: Union[str, Path]) -> Config:
    """
    Load and validate configuration in one step.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Config object with dot notation access

    Raises:
        ValueError: If configuration is invalid
    """
    from .config_validation import check_config

    config = load_config(config_path)
    config_dict = config.to_dict()

    if not check_config(config_dict):
        raise ValueError("Configuration validation failed")

    return config
