"""Camera configuration loader from YAML files, API, and environment variables.

This module provides utilities to load camera configurations from:
1. API (recommended for production)
2. YAML files with environment variable interpolation
3. Environment variables (fallback)
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
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
    3. Fall back to environment variables (HB_IN, HB_OUT, etc.)

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

    # Priority 3: Fall back to environment variables
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


# =============================================================================
# API Camera Loading
# =============================================================================


def parse_roi(roi_points: Optional[List]) -> Optional[Tuple[int, int, int, int]]:
    """Parse ROI points from API format.

    Args:
        roi_points: List of [[x1, y1], [x2, y2]] coordinates from API

    Returns:
        Tuple of (x1, y1, x2, y2) or None if invalid
    """
    if roi_points and len(roi_points) >= 2:
        return tuple(roi_points[0] + roi_points[1])
    return None


def parse_line_points(line_points: Optional[List]) -> Optional[List[Tuple[int, int]]]:
    """Parse virtual line points from API format.

    Args:
        line_points: List of [[x1, y1], [x2, y2]] coordinates from API

    Returns:
        List of (x, y) tuples or None if invalid
    """
    if line_points and len(line_points) >= 2:
        return [tuple(line_points[0]), tuple(line_points[1])]
    return None


def load_cameras_from_api(
    api_client: Any,
    applications: List[str]
) -> List[Dict[str, Any]]:
    """Load camera configurations from API.

    Args:
        api_client: Authenticated APIClient instance
        applications: List of application types to fetch (e.g., ['attendance'])

    Returns:
        List of camera configuration dictionaries in SmartOfficeEngine format
    """
    all_configs = []

    for application in applications:
        cameras = api_client.get_cameras(application=application)

        for cam in cameras:
            config = {
                'camera_id': cam.get('id'),
                'camera_name': cam.get('name', 'Unknown'),
                'cam_type': cam.get('camera_type', 'IN').upper(),
                'stream_url': cam.get('stream_url', ''),
                'application': cam.get('application', application),
                'match_threshold': float(cam.get('matching_threshold', 0.3)),
                'roi': parse_roi(cam.get('roi_points')),
                'line_points': parse_line_points(cam.get('virtual_line_points'))
            }
            all_configs.append(config)

            logger.info(
                f"Camera: {config['camera_name']} | "
                f"Type: {config['cam_type']} | "
                f"App: {application}"
            )

    return all_configs


def load_cameras(
    api_client: Any,
    use_api: bool,
    applications: List[str],
    config_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load camera configurations from API or file.

    This is the main entry point for loading cameras.

    Args:
        api_client: Authenticated APIClient instance
        use_api: Whether to load from API (True) or file/env (False)
        applications: List of application types to filter
        config_path: Optional path to YAML config file

    Returns:
        List of camera configuration dictionaries

    Raises:
        ValueError: If no cameras configured
    """
    if use_api:
        logger.info("Loading camera configs from API (use_api_for_cameras=true)")
        configs = load_cameras_from_api(api_client, applications)
    else:
        logger.info("Loading camera configs from config file")
        cameras = get_camera_configs(config_path)
        configs = convert_to_smart_office_format(cameras)

        # Filter by applications if specified
        if applications:
            filtered_configs = []
            for config in configs:
                app = config.get('application')
                # Handle both string and list application values
                if isinstance(app, list):
                    if any(a in applications for a in app):
                        filtered_configs.append(config)
                elif app in applications:
                    filtered_configs.append(config)
            configs = filtered_configs

    if not configs:
        raise ValueError("No cameras configured. Check config file or API.")

    logger.info(f"Loaded {len(configs)} camera configuration(s)")
    return configs
