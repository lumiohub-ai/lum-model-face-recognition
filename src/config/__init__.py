"""
Configuration

Application settings and startup utilities.
"""

from .camera_loader import load_cameras_from_db
from .settings import settings
from .startup import (
    init_smart_office_app,
    load_config,
    setup_logging,
    log_startup_info,
)

__all__ = [
    # Camera
    "load_cameras_from_db",
    # Settings
    "settings",
    # Startup
    "init_smart_office_app",
    "load_config",
    "setup_logging",
    "log_startup_info",
]
