"""Camera configuration loader from database."""

from typing import List, Dict, Any, Optional, Tuple

from loguru import logger


def _parse_roi(roi_points: Optional[List]) -> Optional[Tuple[int, int, int, int]]:
    """Parse ROI points from database format."""
    if roi_points and len(roi_points) >= 2:
        return tuple(roi_points[0] + roi_points[1])
    return None


def _parse_virtual_lines(cam: Dict) -> List[Dict]:
    """Parse virtual lines from database camera dict.

    Supports new multi-line format (virtual_lines) and old single-line format
    (virtual_line_points). Old format is always treated as person_counting with
    no timer so existing cameras are unaffected.
    """
    raw = cam.get("virtual_lines")
    if raw:
        result = []
        for i, vl in enumerate(raw):
            pts = vl.get("points", [])
            if len(pts) < 2 or len(pts[0]) < 2 or len(pts[1]) < 2:
                continue
            result.append({
                "id": vl.get("id", f"line_{i}"),
                "name": vl.get("name", f"Line {i}"),
                "points": [tuple(pts[0]), tuple(pts[1])],
                "inside_side": int(vl.get("inside_side", 1)),
                "line_type": vl.get("line_type", "person_counting"),
                "timer_enabled": bool(vl.get("timer_enabled", False)),
                "max_distance": float(vl.get("max_distance", 80.0)),
            })
        return result

    lp = cam.get("virtual_line_points")
    if lp and len(lp) >= 2 and len(lp[0]) >= 2 and len(lp[1]) >= 2:
        return [{
            "id": "main",
            "name": "Main Line",
            "points": [tuple(lp[0]), tuple(lp[1])],
            "inside_side": 1,
            "line_type": "person_counting",
            "timer_enabled": False,
            "max_distance": 80.0,
        }]
    return []


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
                'match_threshold': float(cam.get('matching_threshold') or 0.3),
                'roi': _parse_roi(cam.get('roi_points')),
                'virtual_lines': _parse_virtual_lines(cam),
            }
            all_configs.append(config)

    return all_configs
