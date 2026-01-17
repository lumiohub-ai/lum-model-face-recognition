"""Configuration management using Pydantic models."""

from .models import (
    CameraConfig,
    ModelConfig,
    APIConfig,
    StorageConfig,
    TrackingConfig,
    DashboardConfig,
    DatabaseConfig,
    FaceRecognitionConfig as SystemConfig,  # Alias for backward compatibility
)
from .manager import ConfigurationManager
from .constants import *
from .camera_loader import (
    load_cameras,
    load_cameras_from_api,
    get_camera_configs,
    convert_to_smart_office_format,
    parse_roi,
    parse_line_points,
)

__all__ = [
    "CameraConfig",
    "ModelConfig",
    "APIConfig",
    "StorageConfig",
    "TrackingConfig",
    "DashboardConfig",
    "DatabaseConfig",
    "SystemConfig",
    "FaceRecognitionConfig",
    "ConfigurationManager",
    "load_cameras",
    "load_cameras_from_api",
    "get_camera_configs",
    "convert_to_smart_office_format",
    "parse_roi",
    "parse_line_points",
]
