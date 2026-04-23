"""Application startup utilities.

This module provides utilities for initializing the SmartOffice application:
- Configuration loading from YAML files
- Logging setup
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any

import yaml
from loguru import logger


def load_config(config_path: Optional[str] = None, default_config: Optional[Dict] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, looks for config.yaml
                    in configs/ directory.
        default_config: Default configuration to use if file not found.

    Returns:
        Configuration dictionary.
    """
    if default_config is None:
        default_config = {}

    if config_path is None:
        # Look for config in standard locations
        search_paths = [
            Path.cwd() / "configs" / "config.yaml",
            Path.cwd() / "config" / "config.yaml",
            Path.cwd() / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path is None:
        logger.debug("No config file found, using defaults")
        return default_config

    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return default_config

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from {config_path}")
        return {**default_config, **config}
    except yaml.YAMLError as e:
        logger.exception(f"Failed to parse config file {config_path}: {e}")
        return default_config
    except Exception as e:
        logger.exception(f"Failed to load config file {config_path}: {e}")
        return default_config


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: str = "logs",
    app_name: str = "smart_office"
) -> None:
    """Setup application logging with file and console output.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR).
                  If None, uses SO_LOG_LEVEL from settings.
        log_dir: Directory for log files.
        app_name: Application name for log file naming.
    """
    if log_level is None:
        from config.settings import settings
        log_level = settings.log_level

    # Bind service name to all log records
    logger.configure(extra={"service": app_name})

    # Remove default loguru handler
    logger.remove()

    # Console — human-readable with colors (mandatory)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=log_level,
        format=console_format,
        colorize=True,
    )

    # app.log — INFO and DEBUG, JSON, 7-day retention
    logger.add(
        f"{log_dir}/app_{{time:YYYY-MM-DD}}.log",
        level=log_level,
        filter=lambda record: record["level"].no < 30,  # below WARNING
        serialize=True,
        rotation="50 MB",
        retention="7 days",
        compression="gz",
    )

    # error.log — WARNING, ERROR, CRITICAL, JSON, 90-day retention
    logger.add(
        f"{log_dir}/error_{{time:YYYY-MM-DD}}.log",
        level="WARNING",
        filter=lambda record: record["level"].no >= 30,  # WARNING and above
        serialize=True,
        rotation="50 MB",
        retention="90 days",
        compression="gz",
    )

    logger.debug(f"Logging initialized: level={log_level}, log_dir={log_dir}")


def init_smart_office_app(
    config_path: Optional[str] = None,
    log_level: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize SmartOffice application with standard setup.

    1. Sets up logging
    2. Loads configuration from YAML

    Args:
        config_path: Path to YAML config file.
        log_level: Override log level.

    Returns:
        Configuration dictionary.
    """
    setup_logging(log_level=log_level)

    from config.settings import settings
    config_path_to_use = config_path or settings.config_path or None
    return load_config(config_path_to_use)


def log_startup_info(client_slug: str, **extra_info) -> None:
    """Log standard startup information."""
    logger.info("=" * 40)
    logger.info("SmartOfficeEngine Starting")
    logger.info("=" * 40)
    logger.info(f"Client: {client_slug}")

    for key, value in extra_info.items():
        logger.info(f"{key}: {value}")
