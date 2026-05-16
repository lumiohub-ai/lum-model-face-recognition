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
    applications: List[str],
    global_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Load camera configurations from database.

    Args:
        client_slug: Organization slug
        applications: List of application types to fetch (e.g., ['attendance'])
        global_config: Optional global config dict (from config.yaml) for defaults

    Returns:
        List of camera configuration dictionaries
    """
    from infrastructure.storage import Repository

    repository = Repository(client_slug)
    all_configs = []

    cfg = global_config or {}
    default_min_face_size = int(cfg.get('min_face_size', 60))
    default_blur_threshold = float(cfg.get('blur_threshold', 30.0))
    default_match_margin = float(cfg.get('match_margin', 0.10))
    # Global config threshold overrides the DB per-camera value (DB value often left as null/default)
    default_match_threshold = float(cfg.get('match_threshold', 0.42))

    for application in applications:
        cameras = repository.get_cameras(application=application)

        for cam in cameras:
            config = {
                'camera_id': cam.get('id'),
                'camera_name': cam.get('name', 'Unknown'),
                'cam_type': cam.get('camera_type', 'in').upper(),
                'stream_url': cam.get('stream_url', ''),
                'application': cam.get('application', application),
                'match_threshold': float(cam.get('matching_threshold') or default_match_threshold),
                'match_margin': default_match_margin,
                'min_face_size': default_min_face_size,
                'blur_threshold': default_blur_threshold,
                'roi': _parse_roi(cam.get('roi_points')),
                'virtual_lines': _parse_virtual_lines(cam),
            }
            all_configs.append(config)

    return all_configs
