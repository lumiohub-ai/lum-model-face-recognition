"""Configuration manager for the face recognition system.

This module provides a unified configuration management system that bridges
the legacy configuration approach with the new Pydantic-based configuration.
"""

import os
from typing import Any, Dict, List, Optional, Union
from loguru import logger

from .models import (
    FaceRecognitionConfig,
    CameraConfig,
    ModelConfig,
    DatabaseConfig,
    APIConfig,
    StorageConfig,
    TrackingConfig,
    DashboardConfig,
    load_configuration,
)
from .constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MINIMUM_FACE_SIZE,
    DEFAULT_GPU_ID,
)


class ConfigurationManager:
    """Manages configuration for the face recognition system.

    This class provides a bridge between the legacy kwargs-based configuration
    and the new Pydantic-based configuration system.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        cam_types: Optional[List[str]] = None,
        video_paths: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        """Initialize the configuration manager.

        Args:
            config_path: Path to YAML configuration file
            cam_types: List of camera types (IN, OUT, MANAGEMENT)
            video_paths: Video source paths or URLs
            **kwargs: Additional configuration overrides
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.cam_types = cam_types or []
        self.video_paths = self._normalize_video_paths(video_paths)
        self.kwargs = kwargs

        # Build configuration
        self.config = self._build_config()

    def _normalize_video_paths(
        self,
        video_paths: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """Normalize video paths to a list.

        Args:
            video_paths: Single path or list of paths

        Returns:
            List of video paths
        """
        if video_paths is None:
            return []
        if isinstance(video_paths, str):
            return [video_paths]
        return video_paths

    def _build_config(self) -> FaceRecognitionConfig:
        """Build the Pydantic configuration from various sources.

        Configuration precedence (highest to lowest):
        1. Explicit kwargs
        2. Environment variables
        3. YAML config file
        4. Default values

        Returns:
            Validated FaceRecognitionConfig instance
        """
        # Start with kwargs
        config_dict = self._build_config_dict()

        # Try to load from YAML if it exists
        if self.config_path and os.path.exists(self.config_path):
            try:
                return FaceRecognitionConfig.from_yaml(self.config_path, **config_dict)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Falling back to kwargs and environment variables")

        # Fallback to kwargs-only configuration
        return FaceRecognitionConfig.from_kwargs(**config_dict)

    def _build_config_dict(self) -> Dict[str, Any]:
        """Build configuration dictionary from cameras and kwargs.

        Returns:
            Dictionary suitable for FaceRecognitionConfig
        """
        config_dict: Dict[str, Any] = {}

        # Build camera configurations
        if self.cam_types and self.video_paths:
            config_dict['cameras'] = self._build_camera_configs()

        # Build API configuration (required)
        config_dict['api'] = self._build_api_config()

        # Build model configuration
        config_dict['model'] = self._build_model_config()

        # Build database configuration
        config_dict['database'] = self._build_database_config()

        # Build storage configuration
        config_dict['storage'] = self._build_storage_config()

        # Build tracking configuration
        config_dict['tracking'] = self._build_tracking_config()

        # Build dashboard configuration
        config_dict['dashboard'] = self._build_dashboard_config()

        # Global settings
        config_dict['timezone'] = self.kwargs.get('timezone', 'UTC')
        config_dict['production'] = self.kwargs.get('production', False)
        config_dict['log_level'] = self.kwargs.get('log_level', 'INFO')

        return config_dict

    def _build_camera_configs(self) -> List[Dict[str, Any]]:
        """Build camera configuration list.

        Returns:
            List of camera configuration dictionaries
        """
        cameras = []
        camera_ids = self.kwargs.get('camera_id', [])
        camera_names = self.kwargs.get('camera_name', [])
        match_thresholds = self.kwargs.get('match_threshold', [])
        rois = self.kwargs.get('roi', [])
        line_points = self.kwargs.get('line_points', [])

        # Normalize to lists
        if not isinstance(camera_ids, list):
            camera_ids = [camera_ids] if camera_ids else []
        if not isinstance(camera_names, list):
            camera_names = [camera_names] if camera_names else []
        if not isinstance(match_thresholds, list):
            match_thresholds = [match_thresholds] if match_thresholds else []

        for i in range(len(self.cam_types)):
            camera = {
                'camera_id': camera_ids[i] if i < len(camera_ids) else i,
                'camera_name': camera_names[i] if i < len(camera_names) else f"Camera {i}",
                'camera_type': self.cam_types[i],
                'video_path': self.video_paths[i] if i < len(self.video_paths) else "",
                'match_threshold': match_thresholds[i] if i < len(match_thresholds) else DEFAULT_MATCH_THRESHOLD,
            }

            # Optional ROI
            if rois and i < len(rois):
                camera['roi'] = rois[i]

            # Optional line points
            if line_points and i < len(line_points):
                camera['line_points'] = line_points[i]

            cameras.append(camera)

        return cameras

    def _build_api_config(self) -> Dict[str, Any]:
        """Build API configuration.

        Returns:
            API configuration dictionary
        """
        return {
            'api_host': self.kwargs.get('api_host', os.getenv('API_HOST', 'http://localhost:7091')),
            'email': self.kwargs.get('email', os.getenv('SA_EMAIL', '')),
            'password': self.kwargs.get('password', os.getenv('SA_PASSWORD', '')),
            'client_slug': self.kwargs.get('client_slug', os.getenv('HB_CLIENTSLUG', 'default')),
        }

    def _build_model_config(self) -> Dict[str, Any]:
        """Build model configuration.

        Returns:
            Model configuration dictionary
        """
        return {
            'gpu_id': self.kwargs.get('gpu_id', DEFAULT_GPU_ID),
            'minimum_face_size': self.kwargs.get('minimum_face_size', DEFAULT_MINIMUM_FACE_SIZE),
            'model_name': self.kwargs.get('model_name', 'buffalo_l'),
        }

    def _build_database_config(self) -> Dict[str, Any]:
        """Build database configuration.

        Returns:
            Database configuration dictionary
        """
        return {
            'db_path': self.kwargs.get('db_path', DEFAULT_DB_PATH),
            'auto_update': self.kwargs.get('auto_update', True),
        }

    def _build_storage_config(self) -> Dict[str, Any]:
        """Build storage configuration.

        Returns:
            Storage configuration dictionary
        """
        return {
            'base_path': os.getenv('STORAGE_BASE_PATH', '/app/volumes/storage'),
            'fr_slug': os.getenv('FR_SLUG', 'face-recognition'),
            'save_recognized_frames': self.kwargs.get('save_recognized_frames', True),
            'save_unrecognized_frames': self.kwargs.get('save_unrecognized_frames', True),
        }

    def _build_tracking_config(self) -> Dict[str, Any]:
        """Build tracking configuration.

        Returns:
            Tracking configuration dictionary
        """
        return {
            'max_track_lifetime_seconds': self.kwargs.get('max_track_lifetime_seconds', 120),
            'min_frames_for_recognition': self.kwargs.get('min_frames_for_recognition', 3),
            'min_unrecognized_track_lifetime': float(self.kwargs.get('min_unrecognized_track_lifetime', 1.0)),
        }

    def _build_dashboard_config(self) -> Dict[str, Any]:
        """Build dashboard configuration.

        Returns:
            Dashboard configuration dictionary
        """
        return {
            'enabled': os.getenv('DASHBOARD_ENABLED', 'true').lower() == 'true',
            'host': os.getenv('DASHBOARD_HOST', '0.0.0.0'),
            'port': int(os.getenv('DASHBOARD_PORT', '5001')),
            'cors_origins': os.getenv('CORS_ORIGINS', ''),
        }

    def get_camera_config(self, index: int) -> Optional[CameraConfig]:
        """Get configuration for a specific camera.

        Args:
            index: Camera index

        Returns:
            CameraConfig instance or None if index is out of range
        """
        if index < len(self.config.cameras):
            return self.config.cameras[index]
        return None

    def get_all_camera_configs(self) -> List[CameraConfig]:
        """Get all camera configurations.

        Returns:
            List of CameraConfig instances
        """
        return self.config.cameras

    def validate(self) -> bool:
        """Validate the configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Pydantic validation happens automatically during construction
            # Additional custom validation can be added here
            if not self.config.cameras:
                logger.warning("No cameras configured")
                return False

            logger.info(f"Configuration validated: {len(self.config.cameras)} camera(s)")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return self.config.dict()

    def __repr__(self) -> str:
        """String representation of the configuration manager."""
        return (
            f"ConfigurationManager("
            f"cameras={len(self.config.cameras)}, "
            f"client={self.config.api.client_slug}, "
            f"db={self.config.database.db_path}"
            f")"
        )
