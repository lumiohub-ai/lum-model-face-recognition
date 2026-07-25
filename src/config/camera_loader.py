"""Camera configuration loader from database."""

from typing import List, Dict, Any, Optional, Tuple


from config.settings import settings
from config.camera_slug import mediamtx_path as _mediamtx_path


def _resolve_stream_url(cam: Dict[str, Any]) -> str:
    """Source URL for a camera.

    LSO-27: when SO_EDGE_RTSP_BASE is set, read via the Edge MediaMTX
    (rtsp://<base>/<slug(name)>, the high-res main path) — single pull per
    camera, no credentials in the AI, and immune to stream_url being rewritten
    in the dashboard. Otherwise fall back to the DB stream_url.
    """
    base = settings.edge_rtsp_base
    db_url = cam.get("stream_url", "") or ""
    if not base:
        return db_url
    path = _mediamtx_path(cam.get("name") or "")
    if not path:
        logger.warning(
            f"[camera_loader] camera {cam.get('id')} has no name to derive an edge "
            f"path; falling back to DB stream_url"
        )
        return db_url
    return f"{base}/{path}"


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
                'stream_url': _resolve_stream_url(cam),
                'application': cam.get('application', application),
                'match_threshold': float(cam.get('matching_threshold') or 0.3),
                'roi': _parse_roi(cam.get('roi_points')),
                'line_points': _parse_line_points(cam.get('virtual_line_points'))
            }
            all_configs.append(config)

    return all_configs
