"""In-memory lazy cache of per-(client_slug, camera_id) camera projections.

Misses query the shared DB via HomographyRepository on first request and cache
the result. Negative results (camera has no calibration row) are cached too so
repeated misses don't hammer the DB. The cache is invalidated externally on
HomographyCalibrated events.

Each cached entry is a CameraProjection: the homography plus a resolved
`undistort` decision (and the intrinsics to act on it) made once at load time,
per docs/calibration-plan.md item 3 — `frame_source` on the DB row says which
space `H` was fit in, and that space is what runtime must feed it.
"""

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from infrastructure.storage.homography_repository import HomographyRepository


_MISSING = object()


@dataclass
class CameraProjection:
    """Resolved per-camera projection: homography + how to prep a point for it."""

    H: np.ndarray
    map_id: int
    undistort: bool
    K: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    model: str = "fisheye"


class HomographyRegistry:
    """Thread-safe lazy cache: (client_slug, camera_id) -> CameraProjection | None."""

    def __init__(self):
        self._cache: Dict[Tuple[str, int], object] = {}
        self._missing_logged: set = set()
        self._lock = threading.Lock()

    def get(
        self, client_slug: str, camera_id: int
    ) -> Optional[CameraProjection]:
        key = (client_slug, camera_id)

        with self._lock:
            cached = self._cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached  # may be None (cached negative)

        loaded = self._load_from_db(client_slug, camera_id)
        with self._lock:
            self._cache[key] = loaded
        if loaded is None and key not in self._missing_logged:
            logger.warning(
                f"No homography for ({client_slug}, camera_id={camera_id}); "
                f"positions will not be emitted until calibration is computed"
            )
            with self._lock:
                self._missing_logged.add(key)
        return loaded

    def invalidate(self, client_slug: str, camera_id: int) -> None:
        key = (client_slug, camera_id)
        with self._lock:
            self._cache.pop(key, None)
            self._missing_logged.discard(key)
        logger.info(
            f"Homography cache invalidated for ({client_slug}, camera_id={camera_id})"
        )

    @staticmethod
    def _load_from_db(
        client_slug: str, camera_id: int
    ) -> Optional[CameraProjection]:
        try:
            repo = HomographyRepository(client_slug)
        except ValueError as e:
            logger.warning(f"Invalid client_slug '{client_slug}': {e}")
            return None
        row = repo.get_for_camera(camera_id)
        if row is None:
            return None
        H = np.asarray(row.homography_matrix, dtype=np.float64)
        if H.shape != (3, 3):
            logger.warning(
                f"Unexpected homography shape {H.shape} for ({client_slug}, "
                f"camera_id={camera_id}); treating as missing"
            )
            return None

        # frame_source is the source of truth for which space H was fit in.
        # NULL / legacy values mean captured_raw (today's behavior).
        frame_source = row.frame_source or "captured_raw"
        wants_undistorted = frame_source == "captured_undistorted"
        has_intrinsics = bool(row.calibration_matrix and row.distortion_coefficients)

        if wants_undistorted and not has_intrinsics:
            logger.warning(
                f"frame_source=captured_undistorted for ({client_slug}, "
                f"camera_id={camera_id}) but no usable lens calibration; "
                f"falling back to raw projection"
            )
            wants_undistorted = False

        if (
            frame_source == "captured_raw"
            and row.calibration_status == "calibrated"
        ):
            logger.warning(
                f"Stale homography for ({client_slug}, camera_id={camera_id}): "
                f"fit in raw space but camera now has a lens calibration; "
                f"projecting in raw space until recalibrated"
            )

        K = D = None
        model = row.calibration_model or "fisheye"
        if wants_undistorted:
            K = np.asarray(row.calibration_matrix, dtype=np.float64)
            D = np.asarray(row.distortion_coefficients, dtype=np.float64)

        return CameraProjection(
            H=H,
            map_id=row.map_id,
            undistort=wants_undistorted,
            K=K,
            D=D,
            model=model,
        )
