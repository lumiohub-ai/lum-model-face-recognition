"""Camera configuration loader from YAML files with environment variable support.

This module provides utilities to load camera configurations from YAML files
instead of fetching them from the API, supporting environment variable interpolation.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from loguru import logger


def interpolate_env_vars(value: Any) -> Any:
    """Recursively interpolate environment variables in configuration values.

    Supports ${VAR_NAME} syntax for environment variables.

    Args:
        value: Configuration value (can be dict, list, str, or other)

    Returns:
        Value with interpolated environment variables
    """
    if isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [interpolate_env_vars(item) for item in value]
    elif isinstance(value, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)

        for var_name in matches:
            env_value = os.getenv(var_name, '')
            if not env_value:
                logger.warning(
                    f"Environment variable '{var_name}' not found, using empty string"
                )
            value = value.replace(f'${{{var_name}}}', env_value)

        return value
    else:
        return value


def load_cameras_from_yaml(yaml_path: str) -> List[Dict[str, Any]]:
    """Load camera configurations from YAML file.

    Args:
        yaml_path: Path to cameras YAML configuration file

    Returns:
        List of camera configuration dictionaries

    Raises:
        FileNotFoundError: If YAML file not found
        ValueError: If YAML is invalid or cameras not configured
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Camera configuration file not found: {yaml_path}")

    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in camera configuration file: {e}")

    if not config or 'cameras' not in config:
        raise ValueError(
            f"No cameras configured in {yaml_path}. "
            "Expected 'cameras' key with list of camera configs."
        )

    cameras = config['cameras']
    if not cameras:
        raise ValueError(f"Empty cameras list in {yaml_path}")

    # Interpolate environment variables
    cameras = interpolate_env_vars(cameras)

    # Filter out cameras with empty video paths (env var not set)
    valid_cameras = []
    for cam in cameras:
        video_path = cam.get('video_path', '')
        if not video_path:
            logger.warning(
                f"Skipping camera {cam.get('camera_id')} "
                f"({cam.get('camera_name')}) - video_path is empty"
            )
            continue
        valid_cameras.append(cam)

    if not valid_cameras:
        raise ValueError(
            "No valid cameras found after environment variable interpolation. "
            "Check that required environment variables (HB_IN, HB_OUT, etc.) are set."
        )

    logger.info(f"Loaded {len(valid_cameras)} camera configuration(s) from {yaml_path}")
    for cam in valid_cameras:
        logger.info(
            f"  Camera {cam.get('camera_id')}: {cam.get('camera_name')} | "
            f"Type: {cam.get('camera_type')} | "
            f"App: {cam.get('application', 'N/A')}"
        )

    return valid_cameras


def load_cameras_from_env() -> List[Dict[str, Any]]:
    """Load camera configurations from environment variables only.

    Looks for HB_IN, HB_OUT, and HB_IN_MANAGEMENT environment variables
    and creates camera configs for each one that's set.

    Returns:
        List of camera configuration dictionaries

    Raises:
        ValueError: If no cameras configured via environment
    """
    cameras = []

    # Check for HB_IN camera
    if os.getenv('HB_IN'):
        cameras.append({
            'camera_id': 1,
            'camera_name': 'Entry Camera',
            'camera_type': 'IN',
            'video_path': os.getenv('HB_IN'),
            'application': ['attendance'],
            'match_threshold': float(os.getenv('MATCH_THRESHOLD', '0.3')),
            'roi': None,
            'line_points': None
        })

    # Check for HB_OUT camera
    if os.getenv('HB_OUT'):
        cameras.append({
            'camera_id': 2,
            'camera_name': 'Exit Camera',
            'camera_type': 'OUT',
            'video_path': os.getenv('HB_OUT'),
            'application': ['attendance'],
            'match_threshold': float(os.getenv('MATCH_THRESHOLD', '0.3')),
            'roi': None,
            'line_points': None
        })

    # Check for HB_IN_MANAGEMENT camera
    if os.getenv('HB_IN_MANAGEMENT'):
        cameras.append({
            'camera_id': 3,
            'camera_name': 'Management Camera',
            'camera_type': 'MANAGEMENT',
            'video_path': os.getenv('HB_IN_MANAGEMENT'),
            'application': ['attendance'],
            'match_threshold': float(os.getenv('MATCH_THRESHOLD', '0.3')),
            'roi': None,
            'line_points': None
        })

    if not cameras:
        raise ValueError(
            "No cameras configured via environment variables. "
            "Set HB_IN, HB_OUT, or HB_IN_MANAGEMENT environment variables."
        )

    logger.info(f"Loaded {len(cameras)} camera configuration(s) from environment")
    for cam in cameras:
        logger.info(
            f"  Camera {cam.get('camera_id')}: {cam.get('camera_name')} | "
            f"Type: {cam.get('camera_type')}"
        )

    return cameras


def get_camera_configs(config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get camera configurations from file or environment.

    Priority:
    1. If config_path provided and exists, load from YAML file
    2. If CAMERAS_CONFIG env var set, load from that path
    3. Check default path: configs/cameras.yaml
    4. Fall back to environment variables (HB_IN, HB_OUT, etc.)

    Args:
        config_path: Optional path to cameras YAML config file

    Returns:
        List of camera configuration dictionaries

    Raises:
        ValueError: If no valid camera configs found
    """
    # Priority 1: Explicit config path
    if config_path and os.path.exists(config_path):
        return load_cameras_from_yaml(config_path)

    # Priority 2: CAMERAS_CONFIG environment variable
    env_config_path = os.getenv('CAMERAS_CONFIG')
    if env_config_path and os.path.exists(env_config_path):
        return load_cameras_from_yaml(env_config_path)

    # Priority 3: Default path
    default_path = Path(__file__).parent.parent.parent.parent / 'configs' / 'cameras.yaml'
    if default_path.exists():
        return load_cameras_from_yaml(str(default_path))

    # Priority 4: Fall back to environment variables
    logger.info("No camera config file found, loading from environment variables")
    return load_cameras_from_env()


def convert_to_smart_office_format(cameras: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert camera configs to SmartOfficeEngine format.

    Args:
        cameras: List of camera configs from YAML

    Returns:
        List of camera configs in SmartOfficeEngine format
    """
    converted = []
    for cam in cameras:
        config = {
            'camera_id': cam.get('camera_id'),
            'camera_name': cam.get('camera_name', 'Unknown'),
            'cam_type': cam.get('camera_type', 'IN').upper(),
            'stream_url': cam.get('video_path', ''),
            'application': cam.get('application', ['attendance']),
            'match_threshold': float(cam.get('match_threshold', 0.3)),
            'roi': cam.get('roi'),
            'line_points': cam.get('line_points')
        }
        converted.append(config)

    return converted


def convert_to_system_setup_format(cameras: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Convert camera configs to SystemSetup format.

    Args:
        cameras: List of camera configs from YAML

    Returns:
        Dictionary with lists for each camera config field
    """
    result = {
        'cam_types': [],
        'video_path': [],
        'camera_id': [],
        'camera_name': [],
        'match_threshold': [],
        'roi': [],
        'line_points': []
    }

    for cam in cameras:
        result['cam_types'].append(cam.get('camera_type', 'IN').upper())
        result['video_path'].append(cam.get('video_path', ''))
        result['camera_id'].append(cam.get('camera_id'))
        result['camera_name'].append(cam.get('camera_name', 'Unknown'))
        result['match_threshold'].append(float(cam.get('match_threshold', 0.3)))
        result['roi'].append(cam.get('roi'))
        result['line_points'].append(cam.get('line_points'))

    return result
