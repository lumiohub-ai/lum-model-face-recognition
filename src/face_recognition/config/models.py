"""Configuration management for the face recognition system.

This module provides Pydantic models for configuration validation and management,
consolidating configuration from YAML files, environment variables, and kwargs.
"""

import os
from typing import List, Optional, Union, Any, Dict
from pydantic import BaseModel, Field, validator, root_validator
import yaml

from .constants import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MINIMUM_FACE_SIZE,
    DEFAULT_GPU_ID,
    DEFAULT_TRACK_LIFETIME_SECONDS,
    DEFAULT_TIMEZONE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_STORAGE_BASE_PATH,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_DASHBOARD_HOST,
    MIN_MATCH_THRESHOLD,
    MAX_MATCH_THRESHOLD,
    MIN_GPU_ID,
    MAX_GPU_ID,
    MIN_FACE_SIZE,
    MAX_FACE_SIZE,
)


class CameraConfig(BaseModel):
    """Configuration for a single camera.

    Attributes:
        camera_id: Unique identifier for the camera
        camera_name: Human-readable name for the camera
        camera_type: Type of camera (IN, OUT, MANAGEMENT)
        video_path: Path or URL to the video source
        match_threshold: Similarity threshold for face matching (0.0-1.0)
        roi: Region of interest coordinates (x1, y1, x2, y2)
        line_points: Points defining the counting line [(x1, y1), (x2, y2)]
    """

    camera_id: int = Field(..., description="Unique camera identifier")
    camera_name: str = Field(..., description="Camera name")
    camera_type: str = Field(..., description="Camera type (IN/OUT/MANAGEMENT)")
    video_path: str = Field(..., description="Video source path or URL")
    match_threshold: float = Field(
        default=DEFAULT_MATCH_THRESHOLD,
        ge=MIN_MATCH_THRESHOLD,
        le=MAX_MATCH_THRESHOLD,
        description="Face matching threshold"
    )
    roi: Optional[tuple] = Field(None, description="Region of interest (x1, y1, x2, y2)")
    line_points: Optional[List[tuple]] = Field(None, description="Counting line points")

    @validator('camera_type')
    def validate_camera_type(cls, v):
        """Validate that camera type is one of the allowed values."""
        allowed_types = ['IN', 'OUT', 'MANAGEMENT']
        if v.upper() not in allowed_types:
            raise ValueError(f"camera_type must be one of {allowed_types}, got {v}")
        return v.upper()

    @validator('roi')
    def validate_roi(cls, v):
        """Validate ROI coordinates."""
        if v is not None:
            if len(v) != 4:
                raise ValueError("ROI must have 4 coordinates (x1, y1, x2, y2)")
            x1, y1, x2, y2 = v
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid ROI: x2 must be > x1 and y2 must be > y1")
        return v

    @validator('line_points')
    def validate_line_points(cls, v):
        """Validate counting line points."""
        if v is not None:
            if len(v) != 2:
                raise ValueError("line_points must have exactly 2 points")
            for point in v:
                if len(point) != 2:
                    raise ValueError("Each point must have 2 coordinates (x, y)")
        return v


class ModelConfig(BaseModel):
    """Configuration for AI models.

    Attributes:
        gpu_id: GPU device ID to use
        minimum_face_size: Minimum face size in pixels for detection
        model_name: Name of the face recognition model
    """

    gpu_id: int = Field(
        default=DEFAULT_GPU_ID,
        ge=MIN_GPU_ID,
        le=MAX_GPU_ID,
        description="GPU device ID"
    )
    minimum_face_size: int = Field(
        default=DEFAULT_MINIMUM_FACE_SIZE,
        ge=MIN_FACE_SIZE,
        le=MAX_FACE_SIZE,
        description="Minimum face size for detection"
    )
    model_name: str = Field(default="buffalo_l", description="Face recognition model name")


class DatabaseConfig(BaseModel):
    """Configuration for face embeddings database.

    Attributes:
        db_path: Path to the embeddings database file
        auto_update: Whether to automatically update database from API
    """

    db_path: str = Field(default=DEFAULT_DB_PATH, description="Path to embeddings database")
    auto_update: bool = Field(default=True, description="Auto-update database from API")

    @validator('db_path')
    def validate_db_path(cls, v):
        """Validate that database directory exists or can be created."""
        db_dir = os.path.dirname(v)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create database directory {db_dir}: {e}")
        return v


class APIConfig(BaseModel):
    """Configuration for API communication.

    Attributes:
        api_host: API server host URL
        email: Authentication email
        password: Authentication password
        client_slug: Client identifier slug
        timeout: API request timeout in seconds
    """

    api_host: str = Field(..., description="API host URL")
    email: str = Field(..., description="Authentication email")
    password: str = Field(..., description="Authentication password")
    client_slug: str = Field(..., description="Client slug identifier")
    timeout: int = Field(default=30, ge=1, le=300, description="API timeout in seconds")

    @validator('api_host')
    def validate_api_host(cls, v):
        """Validate that API host is a valid URL."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("API host must start with http:// or https://")
        return v.rstrip('/')  # Remove trailing slash


class StorageConfig(BaseModel):
    """Configuration for storage paths.

    Attributes:
        base_path: Base storage path
        fr_slug: Face recognition service slug
        save_recognized_frames: Whether to save recognized face frames
        save_unrecognized_frames: Whether to save unrecognized face frames
    """

    base_path: str = Field(
        default_factory=lambda: os.getenv("STORAGE_BASE_PATH", DEFAULT_STORAGE_BASE_PATH),
        description="Base storage path"
    )
    fr_slug: str = Field(
        default_factory=lambda: os.getenv("FR_SLUG", "face-recognition"),
        description="Face recognition service slug"
    )
    save_recognized_frames: bool = Field(default=True, description="Save recognized frames")
    save_unrecognized_frames: bool = Field(default=True, description="Save unrecognized frames")

    def get_recognized_frames_dir(self, client_slug: str) -> str:
        """Get the directory path for storing recognized frames."""
        return os.path.join(self.base_path, self.fr_slug, "data", client_slug, "recognized_frames")

    def get_unrecognized_frames_dir(self, client_slug: str) -> str:
        """Get the directory path for storing unrecognized frames."""
        return os.path.join(self.base_path, self.fr_slug, "data", client_slug, "unrecognized_frames")

    def get_collection_dir(self, client_slug: str) -> str:
        """Get the directory path for data collection."""
        return os.path.join(self.base_path, self.fr_slug, "data", client_slug, "collection")


class TrackingConfig(BaseModel):
    """Configuration for face tracking.

    Attributes:
        max_track_lifetime_seconds: Maximum lifetime for a track in seconds
        min_frames_for_recognition: Minimum frames required for recognition
        min_unrecognized_track_lifetime: Minimum track lifetime for unrecognized faces (seconds)
    """

    max_track_lifetime_seconds: int = Field(
        default=DEFAULT_TRACK_LIFETIME_SECONDS,
        ge=1,
        le=3600,
        description="Maximum track lifetime in seconds"
    )
    min_frames_for_recognition: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Minimum frames required for recognition"
    )
    min_unrecognized_track_lifetime: float = Field(
        default=1.0,
        ge=0.0,
        le=60.0,
        description="Minimum track lifetime for unrecognized faces to be sent to dashboard (seconds)"
    )


class DashboardConfig(BaseModel):
    """Configuration for the web dashboard.

    Attributes:
        enabled: Whether the dashboard is enabled
        host: Dashboard server host
        port: Dashboard server port
        cors_origins: Allowed CORS origins (comma-separated)
    """

    enabled: bool = Field(default=True, description="Enable dashboard")
    host: str = Field(
        default_factory=lambda: os.getenv("DASHBOARD_HOST", DEFAULT_DASHBOARD_HOST),
        description="Dashboard host"
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("DASHBOARD_PORT", str(DEFAULT_DASHBOARD_PORT))),
        ge=1,
        le=65535,
        description="Dashboard port"
    )
    cors_origins: str = Field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", ""),
        description="Allowed CORS origins (comma-separated)"
    )

    @validator('cors_origins')
    def validate_cors_origins(cls, v):
        """Validate CORS origins configuration."""
        if not v or v == "*":
            import warnings
            warnings.warn(
                "CORS is set to allow all origins (*). "
                "This is insecure for production. "
                "Set CORS_ORIGINS environment variable to specific domains.",
                UserWarning
            )
        return v


class FaceRecognitionConfig(BaseModel):
    """Main configuration for the face recognition system.

    This class consolidates all configuration from YAML files, environment variables,
    and runtime arguments with proper validation and precedence.
    """

    cameras: List[CameraConfig] = Field(default_factory=list, description="Camera configurations")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    api: APIConfig = Field(..., description="API configuration")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage configuration")
    tracking: TrackingConfig = Field(default_factory=TrackingConfig, description="Tracking configuration")
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig, description="Dashboard configuration")

    timezone: str = Field(default=DEFAULT_TIMEZONE, description="Timezone for timestamps")
    production: bool = Field(default=False, description="Production mode flag")
    log_level: str = Field(default="INFO", description="Logging level")

    @validator('timezone')
    def validate_timezone(cls, v):
        """Validate timezone string."""
        try:
            import pytz
            pytz.timezone(v)
        except Exception:
            raise ValueError(f"Invalid timezone: {v}")
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f"log_level must be one of {allowed_levels}")
        return v.upper()

    @root_validator(pre=True)
    def merge_environment_variables(cls, values):
        """Merge environment variables with configuration values."""
        # This allows environment variables to override config file values
        if 'api' in values and isinstance(values['api'], dict):
            # Override API config with environment variables if present
            values['api']['email'] = os.getenv('SA_EMAIL', values['api'].get('email'))
            values['api']['password'] = os.getenv('SA_PASSWORD', values['api'].get('password'))
            values['api']['client_slug'] = os.getenv('HB_CLIENTSLUG', values['api'].get('client_slug'))
            values['api']['api_host'] = os.getenv('API_HOST', values['api'].get('api_host'))

        return values

    @classmethod
    def from_yaml(cls, yaml_path: str, **kwargs) -> 'FaceRecognitionConfig':
        """Load configuration from YAML file with optional overrides.

        Args:
            yaml_path: Path to YAML configuration file
            **kwargs: Override values for any configuration field

        Returns:
            Validated FaceRecognitionConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If configuration is invalid
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}

        # Merge YAML config with kwargs (kwargs take precedence)
        merged_config = {**yaml_config, **kwargs}

        return cls(**merged_config)

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'FaceRecognitionConfig':
        """Create configuration from keyword arguments only.

        This is useful when no YAML file is available and all configuration
        comes from environment variables and runtime arguments.

        Args:
            **kwargs: Configuration values

        Returns:
            Validated FaceRecognitionConfig instance
        """
        # Build minimal config from kwargs
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return self.dict()

    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            output_path: Path where YAML file should be saved
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)


# Helper function for backward compatibility
def load_configuration(
    config_path: Optional[str] = None,
    **kwargs
) -> FaceRecognitionConfig:
    """Load configuration with precedence: kwargs > env vars > YAML > defaults.

    Args:
        config_path: Optional path to YAML configuration file
        **kwargs: Override values for any configuration field

    Returns:
        Validated FaceRecognitionConfig instance

    Example:
        >>> config = load_configuration(
        ...     config_path="configs/config.yaml",
        ...     production=True,
        ...     log_level="DEBUG"
        ... )
    """
    if config_path and os.path.exists(config_path):
        return FaceRecognitionConfig.from_yaml(config_path, **kwargs)
    else:
        # No YAML file, build from kwargs and env vars only
        return FaceRecognitionConfig.from_kwargs(**kwargs)
