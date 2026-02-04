"""Application startup utilities.

This module provides utilities for initializing the SmartOffice application:
- Configuration loading from YAML files
- Environment variable validation
- Logging setup
- .env file loading for local development
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from loguru import logger

from .logging.setup import setup_structured_logging


def load_dotenv_if_exists(env_path: Optional[Path] = None) -> bool:
    """Load .env file if it exists (for local development).

    Docker deployments should use environment variables from docker-compose.

    Args:
        env_path: Optional path to .env file. If None, looks in current directory.

    Returns:
        True if .env file was loaded, False otherwise.
    """
    try:
        from dotenv import load_dotenv

        if env_path is None:
            env_path = Path.cwd() / '.env'

        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.debug(f"Loaded .env file from {env_path}")
            return True
        return False
    except ImportError:
        # dotenv not installed, use system environment variables
        return False


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
        logger.error(f"Failed to parse config file {config_path}: {e}")
        return default_config
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return default_config


def validate_environment(required_vars: List[str]) -> List[str]:
    """Validate that required environment variables are set.

    Args:
        required_vars: List of required environment variable names.

    Returns:
        List of missing variable names (empty if all present).
    """
    missing = [var for var in required_vars if not os.getenv(var)]
    return missing


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: str = "logs",
    app_name: str = "smart_office"
) -> None:
    """Setup application logging with file and console output.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR).
                  If None, uses LOG_LEVEL env var or defaults to INFO.
        log_dir: Directory for log files.
        app_name: Application name for log file naming.
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    # Remove default loguru handler
    logger.remove()

    # Console logging with colors
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

    # File logging with rotation
    log_file = f"{log_dir}/{app_name}_{{time:YYYY-MM-DD}}.log"
    logger.add(
        log_file,
        rotation="1 day",
        retention="7 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    logger.debug(f"Logging initialized: level={log_level}, log_dir={log_dir}")


def init_smart_office_app(
    required_env_vars: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    env_file: Optional[Path] = None,
    log_level: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize SmartOffice application with standard setup.

    This function performs the following:
    1. Loads .env file if present (for local development)
    2. Validates required environment variables
    3. Sets up logging
    4. Loads configuration from YAML

    Args:
        required_env_vars: List of required environment variable names.
                          Defaults to SA_EMAIL, SA_PASSWORD, HB_CLIENTSLUG, API_HOST.
        config_path: Path to YAML config file.
        env_file: Path to .env file for local development.
        log_level: Override log level.

    Returns:
        Configuration dictionary.

    Raises:
        SystemExit: If required environment variables are missing.
    """
    if required_env_vars is None:
        required_env_vars = ["SA_EMAIL", "SA_PASSWORD", "HB_CLIENTSLUG", "API_HOST"]

    # Step 1: Load .env file if present
    load_dotenv_if_exists(env_file)

    # Step 2: Setup logging first so we can log subsequent steps
    setup_logging(log_level=log_level)

    # Step 3: Validate environment variables
    missing = validate_environment(required_env_vars)
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        sys.exit(1)

    # Step 4: Load configuration
    config_path_to_use = config_path or os.getenv("SMART_OFFICE_CONFIG")
    config = load_config(config_path_to_use)

    return config


def log_startup_info(client_slug: str, api_host: str, **extra_info) -> None:
    """Log standard startup information.

    Args:
        client_slug: Client organization slug.
        api_host: API host URL.
        **extra_info: Additional key-value pairs to log.
    """
    logger.info("=" * 60)
    logger.info("SmartOfficeEngine Starting")
    logger.info("=" * 60)
    logger.info(f"Client: {client_slug}")
    logger.info(f"API Host: {api_host}")

    for key, value in extra_info.items():
        logger.info(f"{key}: {value}")
