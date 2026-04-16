"""
Configuration loader module for Air Quality Prediction System.

This module handles loading configuration from YAML files and environment
variables, with support for parameter validation and default values.
"""

import os
import re
from typing import Any, Dict, Optional
import yaml

from .logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Load and manage system configuration from YAML and environment variables."""

    def __init__(self, config_file: str = 'config.yaml'):
        """
        Initialize configuration loader.

        Args:
            config_file: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file does not exist
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}

        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}"
            )

        self._load_config()

    def _load_config(self) -> None:
        """
        Load configuration from YAML file.

        Raises:
            yaml.YAMLError: If YAML parsing fails
        """
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {self.config_file}")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _resolve_env_variables(self, value: Any) -> Any:
        """
        Resolve environment variable references in configuration values.

        Supports ${VAR_NAME} syntax for environment variable substitution.

        Args:
            value: Configuration value that may contain env var references

        Returns:
            Resolved value with environment variables substituted

        Example:
            >>> os.environ['API_KEY'] = 'secret123'
            >>> loader._resolve_env_variables('${API_KEY}')
            'secret123'
        """
        if not isinstance(value, str):
            return value

        # Pattern to match ${VAR_NAME}
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)

        for var_name in matches:
            env_value = os.environ.get(var_name)
            if env_value is None:
                logger.warning(
                    f"Environment variable not found: {var_name}. "
                    f"Using placeholder value."
                )
                env_value = f"${{{var_name}}}"
            value = value.replace(f"${{{var_name}}}", env_value)

        return value

    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override configuration values with environment variables.

        Environment variables follow the pattern: AQI_SECTION_KEY
        For nested values: AQI_SECTION_SUBSECTION_KEY

        Args:
            config: Configuration dictionary to override

        Returns:
            Configuration with environment variable overrides applied

        Example:
            >>> os.environ['AQI_SYSTEM_LOG_LEVEL'] = 'DEBUG'
            >>> config = {'system': {'log_level': 'INFO'}}
            >>> loader._override_with_env(config)
            {'system': {'log_level': 'DEBUG'}}
        """
        for key, value in config.items():
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                config[key] = self._override_with_env(value)
            else:
                # Check for environment variable override
                env_key = f"AQI_{key.upper()}"
                if env_key in os.environ:
                    config[key] = os.environ[env_key]
                    logger.info(
                        f"Configuration overridden by environment variable: "
                        f"{env_key}"
                    )
                else:
                    # Resolve any ${VAR} references in the value
                    config[key] = self._resolve_env_variables(value)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Configuration key (supports dot notation for nested values)
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> loader.get('system.log_level')
            'INFO'
            >>> loader.get('nonexistent.key', 'default_value')
            'default_value'
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                logger.warning(
                    f"Configuration key not found: {key}. "
                    f"Using default: {default}"
                )
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name

        Returns:
            Configuration section as dictionary

        Raises:
            KeyError: If section not found
        """
        if section not in self.config:
            raise KeyError(f"Configuration section not found: {section}")
        return self.config[section]

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        errors = []

        # Validate system section
        system = self.config.get('system', {})
        if system.get('log_level') not in [
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        ]:
            errors.append("Invalid log_level in system section")

        # Validate cities list
        cities = self.config.get('cities', [])
        if not cities or not isinstance(cities, list):
            errors.append("Cities list is empty or invalid")

        # Validate storage paths
        storage = self.config.get('storage', {})
        required_paths = ['bronze_path', 'silver_path', 'gold_path']
        for path_key in required_paths:
            if not storage.get(path_key):
                errors.append(f"Missing storage path: {path_key}")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Configuration validation passed")
        return True

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        logger.info("Configuration reloaded")

    def to_dict(self) -> Dict[str, Any]:
        """
        Get entire configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()


def load_config(config_file: str = 'config.yaml') -> ConfigLoader:
    """
    Load configuration from file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        ConfigLoader instance

    Example:
        >>> config = load_config('config.yaml')
        >>> log_level = config.get('system.log_level')
    """
    try:
        loader = ConfigLoader(config_file)
        loader.validate()
        return loader
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def get_config_value(
    key: str,
    default: Any = None,
    config_file: str = 'config.yaml'
) -> Any:
    """
    Convenience function to get a single configuration value.

    Args:
        key: Configuration key (dot notation)
        default: Default value if not found
        config_file: Path to configuration file

    Returns:
        Configuration value or default

    Example:
        >>> log_level = get_config_value('system.log_level', 'INFO')
    """
    try:
        config = load_config(config_file)
        return config.get(key, default)
    except Exception as e:
        logger.warning(f"Failed to get config value: {e}. Using default.")
        return default
