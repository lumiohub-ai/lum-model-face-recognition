"""Camera configuration loader from database."""

from typing import List, Dict, Any, Optional, Tuple

from loguru import logger


def _parse_roi(roi_points: Optional[List]) -> Optional[Tuple[int, int, int, int]]:
    """Parse ROI points from database format."""
    if roi_points and len(roi_points) >= 2:
        return tuple(roi_points[0] + roi_points[1])
    return None


def _parse_line_points(line_points: Optional[List]) -> Optional[List[Tuple[int, int]]]:
    """Parse virtual line points from database format."""
    if line_points and len(line_points) >= 2:
        return [tuple(line_points[0]), tuple(line_points[1])]
    return None


def load_cameras_from_db(
    client_slug: str,
    applications: List[str]
) -> List[Dict[str, Any]]:
    """Load camera configurations from database.

    Args:
        client_slug: Organization slug
        applications: List of application types to fetch (e.g., ['attendance'])

    Returns:
        List of camera configuration dictionaries
    """
    from infrastructure.storage import Repository

    repository = Repository(client_slug)
    all_configs = []

    for application in applications:
        cameras = repository.get_cameras(application=application)

        for cam in cameras:
            config = {
                'camera_id': cam.get('id'),
                'camera_name': cam.get('name', 'Unknown'),
                'cam_type': cam.get('camera_type', 'in').upper(),
                'stream_url': cam.get('stream_url', ''),
                'application': cam.get('application', application),
                'match_threshold': (
                    float(cam['matching_threshold'])
                    if cam.get('matching_threshold') is not None
                    else 0.3
                ),
                'roi': _parse_roi(cam.get('roi_points')),
                'line_points': _parse_line_points(cam.get('virtual_line_points'))
            }
            all_configs.append(config)

    return all_configs
