"""
Pydantic configuration models for Person Tracking system.

These models define and validate all configuration parameters
for cameras, tracking, and detection.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import os


class PersonDetectionConfig(BaseModel):
    """Configuration for person detection using YOLOv8-Pose."""

    model_size: Literal["n", "s", "m", "l", "x"] = Field(
        default="s",
        description="YOLOv8-Pose model size (n=nano, s=small, m=medium, l=large, x=xlarge)"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for person detection"
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS"
    )

    @field_validator('model_size')
    @classmethod
    def validate_model_size(cls, v):
        if v not in ["n", "s", "m", "l", "x"]:
            raise ValueError(f"model_size must be one of: n, s, m, l, x")
        return v


class PersonTrackingConfig(BaseModel):
    """Configuration for person tracking."""

    tracker_type: Literal["botsort", "bytetrack", "ocsort"] = Field(
        default="botsort",
        description="Tracking algorithm to use"
    )
    max_track_age: int = Field(
        default=1,
        ge=1,
        description="Maximum seconds to keep track without updates"
    )
    min_track_hits: int = Field(
        default=3,
        ge=1,
        description="Minimum frames before confirming track"
    )
    iou_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="IoU threshold for track association"
    )


class FaceRecognitionConfig(BaseModel):
    """Configuration for face recognition integration."""

    match_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for face matching"
    )
    identity_lock_frames: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of frames (M) for identity lock voting window"
    )
    identity_consensus: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Percentage of frames needed for identity lock (e.g., 0.60 = 60%)"
    )
    min_window_duration_ms: int = Field(
        default=333,
        ge=0,
        description="Minimum window duration in milliseconds"
    )


class PerformanceConfig(BaseModel):
    """Performance and optimization settings."""

    target_fps: int = Field(
        default=12,
        ge=1,
        le=60,
        description="Target FPS for processing"
    )
    resize_width: Optional[int] = Field(
        default=1280,
        ge=320,
        le=3840,
        description="Resize frame width for performance (None = no resize)"
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Batch size for model inference"
    )


class StorageConfig(BaseModel):
    """Storage and output configuration."""

    save_annotated_frames: bool = Field(
        default=True,
        description="Save annotated frames to disk"
    )
    output_dir: str = Field(
        default="volumes/storage/person-tracking",
        description="Output directory for saved files"
    )
    clip_duration_seconds: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Duration of saved clips in seconds"
    )


class CameraConfig(BaseModel):
    """Configuration for a single camera."""

    camera_id: int = Field(
        description="Unique camera identifier"
    )
    camera_name: str = Field(
        description="Human-readable camera name"
    )
    video_path: str = Field(
        description="RTSP URL or video file path"
    )

    # Sub-configurations
    person_detection: PersonDetectionConfig = Field(
        default_factory=PersonDetectionConfig
    )
    person_tracking: PersonTrackingConfig = Field(
        default_factory=PersonTrackingConfig
    )
    face_recognition: FaceRecognitionConfig = Field(
        default_factory=FaceRecognitionConfig
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig
    )

    @field_validator('video_path')
    @classmethod
    def validate_video_path(cls, v):
        """Validate video path or RTSP URL."""
        if v.startswith('rtsp://') or v.startswith('http://'):
            return v
        # Check if file exists
        if not os.path.exists(v) and not v.startswith('$'):
            # Allow environment variables
            pass
        return v


class DatabaseConfig(BaseModel):
    """Database configuration (shared with face recognition service)."""

    use_pgvector: bool = Field(
        default=True,
        description="Use pgvector for embeddings (recommended)"
    )
    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL host"
    )
    postgres_port: int = Field(
        default=5434,
        ge=1,
        le=65535,
        description="PostgreSQL port"
    )
    postgres_user: str = Field(
        default="face_recognition",
        description="PostgreSQL username"
    )
    postgres_password: str = Field(
        default="",
        description="PostgreSQL password"
    )
    postgres_db: str = Field(
        default="face_embeddings",
        description="PostgreSQL database name"
    )

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


class RedisConfig(BaseModel):
    """Redis configuration for pub/sub."""

    host: str = Field(
        default="localhost",
        description="Redis host"
    )
    port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port"
    )
    db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    console: bool = Field(
        default=True,
        description="Enable console logging"
    )
    file: bool = Field(
        default=True,
        description="Enable file logging"
    )
    file_path: str = Field(
        default="logs/person_tracking.log",
        description="Log file path"
    )
    csv_logging: bool = Field(
        default=True,
        description="Enable CSV event logging"
    )
    csv_path: str = Field(
        default="volumes/storage/person-tracking/logs",
        description="CSV log directory"
    )


class APIConfig(BaseModel):
    """API configuration."""

    host: str = Field(
        default="0.0.0.0",
        description="API host"
    )
    port: int = Field(
        default=5002,
        ge=1,
        le=65535,
        description="API port"
    )
    backend_url: Optional[str] = Field(
        default=None,
        description="Backend API URL for real integration (post-MVP)"
    )
    enable_docs: bool = Field(
        default=True,
        description="Enable Swagger/OpenAPI documentation"
    )


class PersonTrackingAppConfig(BaseModel):
    """Main configuration for Person Tracking system."""

    project_name: str = Field(
        default="person_tracking",
        description="Project name"
    )
    client_slug: str = Field(
        description="Client organization slug"
    )

    # Sub-configurations
    cameras: List[CameraConfig] = Field(
        default_factory=list,
        description="List of camera configurations"
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig
    )
    redis: RedisConfig = Field(
        default_factory=RedisConfig
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig
    )
    api: APIConfig = Field(
        default_factory=APIConfig
    )

    @field_validator('cameras')
    @classmethod
    def validate_cameras(cls, v):
        """Ensure at least one camera is configured."""
        if not v:
            raise ValueError("At least one camera must be configured")

        # Check for duplicate camera IDs
        camera_ids = [cam.camera_id for cam in v]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("Duplicate camera IDs found")

        return v

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment
