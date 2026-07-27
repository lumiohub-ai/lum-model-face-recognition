"""Camera configuration loader from database."""

from typing import List, Dict, Any, Optional, Tuple

from loguru import logger

from config.settings import settings
from config.camera_slug import mediamtx_path as _mediamtx_path


def _resolve_stream_url(cam: Dict[str, Any]) -> Optional[str]:
    """Edge MediaMTX source URL for a camera — NEVER the camera directly (LSO-27).

    The AI reads rtsp://<SO_EDGE_RTSP_BASE>/<slug(name)> (the high-res main path):
    one pull per camera, no camera credentials in the AI, and immune to the
    dashboard rewriting stream_url. Pulling from a camera directly is
    deliberately unsupported — the AI must not hold camera credentials nor open a
    second connection to the camera. A camera that can't be mapped to an edge
    path is skipped (returns None) rather than fetched directly.
    """
    base = settings.edge_rtsp_base
    if not base:
        raise RuntimeError(
            "SO_EDGE_RTSP_BASE is not set. The AI reads exclusively via the Edge "
            "MediaMTX and never pulls cameras directly — set SO_EDGE_RTSP_BASE "
            "(e.g. rtsp://host.docker.internal:8554)."
        )
    path = _mediamtx_path(cam.get("name") or "")
    if not path:
        logger.warning(
            f"[camera_loader] camera {cam.get('id')} has no name to derive an edge "
            f"path; skipping (the AI never falls back to a direct camera pull)"
        )
        return None
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
            stream_url = _resolve_stream_url(cam)
            if not stream_url:
                # Unmappable to an edge path — skip rather than pull the camera
                # directly (the AI never opens a direct/credentialed connection).
                continue
            config = {
                'camera_id': cam.get('id'),
                'camera_name': cam.get('name', 'Unknown'),
                'cam_type': cam.get('camera_type', 'in').upper(),
                'stream_url': stream_url,
                'application': cam.get('application', application),
                'match_threshold': float(cam.get('matching_threshold') or 0.3),
                'roi': _parse_roi(cam.get('roi_points')),
                'line_points': _parse_line_points(cam.get('virtual_line_points'))
            }
            all_configs.append(config)

    return all_configs
