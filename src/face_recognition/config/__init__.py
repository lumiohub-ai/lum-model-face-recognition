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
]
