"""
Configuration

Application settings, constants, and startup utilities.
"""

from .constants import *
from .camera_loader import load_cameras
from .startup import (
    init_smart_office_app,
    load_config,
    load_dotenv_if_exists,
    validate_environment,
    setup_logging,
    log_startup_info,
)
from .logging import setup_structured_logging

__all__ = [
    # Camera
    "load_cameras",
    # Startup
    "init_smart_office_app",
    "load_config",
    "load_dotenv_if_exists",
    "validate_environment",
    "setup_logging",
    "log_startup_info",
    "setup_structured_logging",
]
