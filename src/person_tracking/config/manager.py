"""
Configuration Manager for Person Tracking system.

Loads configuration from YAML files with environment variable interpolation
and validates using Pydantic models.
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from loguru import logger

from .models import PersonTrackingAppConfig


class ConfigurationManager:
    """Manages loading and validation of configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file.
                        If None, uses default path or falls back to kwargs.
        """
        self.config_path = config_path
        self.config: Optional[PersonTrackingAppConfig] = None

    @staticmethod
    def interpolate_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively interpolate environment variables in configuration.

        Supports ${VAR_NAME} syntax for environment variables.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Dictionary with interpolated environment variables
        """
        if isinstance(config_dict, dict):
            return {
                key: ConfigurationManager.interpolate_env_vars(value)
                for key, value in config_dict.items()
            }
        elif isinstance(config_dict, list):
            return [
                ConfigurationManager.interpolate_env_vars(item)
                for item in config_dict
            ]
        elif isinstance(config_dict, str):
            # Match ${VAR_NAME} pattern
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, config_dict)

            for var_name in matches:
                env_value = os.getenv(var_name, '')
                if not env_value:
                    logger.warning(
                        f"Environment variable '{var_name}' not found, using empty string"
                    )
                config_dict = config_dict.replace(f'${{{var_name}}}', env_value)

            return config_dict
        else:
            return config_dict

    @classmethod
    def load_config(
        cls,
        config_path: Optional[str] = None,
        **kwargs
    ) -> "ConfigurationManager":
        """
        Load configuration from YAML file or kwargs.

        Args:
            config_path: Path to YAML configuration file
            **kwargs: Override configuration values

        Returns:
            ConfigurationManager with validated PersonTrackingAppConfig

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If configuration is invalid
        """
        manager = cls(config_path)

        # Try to load from YAML file
        config_dict = {}
        if config_path and os.path.exists(config_path):
            logger.info(f"Loading configuration from: {config_path}")
            config_dict = manager._load_yaml(config_path)
        elif config_path:
            logger.warning(
                f"Configuration file not found: {config_path}. "
                "Using kwargs or defaults."
            )

        # Interpolate environment variables
        if config_dict:
            config_dict = cls.interpolate_env_vars(config_dict)

        # Merge with kwargs (kwargs take precedence)
        config_dict.update(kwargs)

        # Load environment variables for common settings
        config_dict = manager._load_env_overrides(config_dict)

        # Validate and create Pydantic model
        try:
            config = PersonTrackingAppConfig(**config_dict)
            logger.info("Configuration loaded and validated successfully")
            manager.config = config
            return manager
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")

    @staticmethod
    def _load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file.

        Args:
            file_path: Path to YAML file

        Returns:
            Dictionary with configuration

        Raises:
            FileNotFoundError: If file not found
            yaml.YAMLError: If YAML is invalid
        """
        try:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                return config_dict or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")

    @staticmethod
    def _load_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load common environment variable overrides.

        Args:
            config_dict: Current configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        # Client slug
        if 'client_slug' not in config_dict:
            config_dict['client_slug'] = os.getenv('HB_CLIENTSLUG', '')

        # Database configuration
        if 'database' not in config_dict:
            config_dict['database'] = {}

        db_config = config_dict['database']
        db_config['use_pgvector'] = os.getenv('USE_PGVECTOR', 'true').lower() == 'true'
        db_config['postgres_host'] = os.getenv('POSTGRES_HOST', db_config.get('postgres_host', 'localhost'))
        db_config['postgres_port'] = int(os.getenv('POSTGRES_PORT', db_config.get('postgres_port', 5434)))
        db_config['postgres_user'] = os.getenv('POSTGRES_USER', db_config.get('postgres_user', 'face_recognition'))
        db_config['postgres_password'] = os.getenv('POSTGRES_PASSWORD', db_config.get('postgres_password', ''))
        db_config['postgres_db'] = os.getenv('POSTGRES_DB', db_config.get('postgres_db', 'face_embeddings'))

        # Redis configuration
        if 'redis' not in config_dict:
            config_dict['redis'] = {}

        redis_config = config_dict['redis']
        redis_config['host'] = os.getenv('REDIS_HOST', redis_config.get('host', 'localhost'))
        redis_config['port'] = int(os.getenv('REDIS_PORT', redis_config.get('port', 6379)))

        # API configuration
        if 'api' not in config_dict:
            config_dict['api'] = {}

        api_config = config_dict['api']
        api_config['backend_url'] = os.getenv('SO_BACKEND_API_URL', api_config.get('backend_url'))

        return config_dict

    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to save configuration

        Raises:
            ValueError: If no configuration loaded
        """
        if not self.config:
            raise ValueError("No configuration loaded")

        # Convert Pydantic model to dict
        config_dict = self.config.model_dump()

        # Save to YAML
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {output_path}")

    @staticmethod
    def create_default_config(
        camera_id: int = 1,
        camera_name: str = "Default Camera",
        video_path: str = "${HB_IN}",
        client_slug: str = "${HB_CLIENTSLUG}"
    ) -> PersonTrackingAppConfig:
        """
        Create a default configuration for quick setup.

        Args:
            camera_id: Camera ID
            camera_name: Camera name
            video_path: Video source path or RTSP URL
            client_slug: Client organization slug

        Returns:
            Default PersonTrackingAppConfig object
        """
        from .models import CameraConfig

        camera_config = CameraConfig(
            camera_id=camera_id,
            camera_name=camera_name,
            video_path=video_path
        )

        config = PersonTrackingAppConfig(
            client_slug=client_slug,
            cameras=[camera_config]
        )

        logger.info("Created default configuration")
        return config

    @classmethod
    def load_config_from_env(cls) -> "ConfigurationManager":
        """
        Load configuration from environment variables only.

        Returns:
            ConfigurationManager with PersonTrackingAppConfig loaded from environment
        """
        from .models import CameraConfig, PersonTrackingAppConfig

        manager = cls()
        client_slug = os.getenv('HB_CLIENTSLUG', 'default_client')

        # Create camera config from environment
        cameras = []

        # Check for HB_IN camera
        if os.getenv('HB_IN'):
            cameras.append(CameraConfig(
                camera_id=1,
                camera_name="Entry Camera",
                video_path=os.getenv('HB_IN')
            ))

        # Check for HB_OUT camera
        if os.getenv('HB_OUT'):
            cameras.append(CameraConfig(
                camera_id=2,
                camera_name="Exit Camera",
                video_path=os.getenv('HB_OUT')
            ))

        # Check for HB_IN_MANAGEMENT camera
        if os.getenv('HB_IN_MANAGEMENT'):
            cameras.append(CameraConfig(
                camera_id=3,
                camera_name="Management Camera",
                video_path=os.getenv('HB_IN_MANAGEMENT')
            ))

        if not cameras:
            raise ValueError(
                "No cameras configured. Set HB_IN, HB_OUT, or HB_IN_MANAGEMENT "
                "environment variables."
            )

        config = PersonTrackingAppConfig(
            client_slug=client_slug,
            cameras=cameras
        )

        manager.config = config
        logger.info(f"Loaded configuration from environment for {len(cameras)} camera(s)")
        return manager
