"""In-memory lazy cache of per-(client_slug, camera_id) homography matrices.

Misses query the shared DB via HomographyRepository on first request and cache
the result. Negative results (camera has no calibration row) are cached too so
repeated misses don't hammer the DB. The cache is invalidated externally on
HomographyCalibrated events.
"""

import threading
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from infrastructure.storage.homography_repository import HomographyRepository


_MISSING = object()


class HomographyRegistry:
    """Thread-safe lazy cache: (client_slug, camera_id) -> (H_3x3, map_id) | None."""

    def __init__(self):
        self._cache: Dict[Tuple[str, int], object] = {}
        self._missing_logged: set = set()
        self._lock = threading.Lock()
        # Bumped on every invalidate() so an in-flight _load_from_db() that
        # started before a recalibration can detect it raced with an
        # invalidate() and avoid clobbering the fresh state with a
        # potentially-stale result once it finishes.
        self._generation = 0

    def get(
        self, client_slug: str, camera_id: int
    ) -> Optional[Tuple[np.ndarray, int]]:
        key = (client_slug, camera_id)

        with self._lock:
            cached = self._cache.get(key, _MISSING)
            generation_before_load = self._generation
        if cached is not _MISSING:
            return cached  # may be None (cached negative)

        try:
            loaded = self._load_from_db(client_slug, camera_id)
        except Exception as e:
            # A real DB/query failure — do NOT cache this as "not
            # calibrated"; that would permanently disable position
            # streaming for a camera that's actually fine after one
            # transient error. Just return None for this call and let the
            # next get() retry.
            logger.exception(
                f"Failed to load homography for ({client_slug}, "
                f"camera_id={camera_id}), will retry on next request: {e}"
            )
            return None

        with self._lock:
            if self._generation == generation_before_load:
                self._cache[key] = loaded
            # else: invalidate() ran while this load was in flight — discard
            # this result rather than re-populating a just-invalidated entry.
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
            self._generation += 1
        logger.info(
            f"Homography cache invalidated for ({client_slug}, camera_id={camera_id})"
        )

    @staticmethod
    def _load_from_db(
        client_slug: str, camera_id: int
    ) -> Optional[Tuple[np.ndarray, int]]:
        try:
            repo = HomographyRepository(client_slug)
        except ValueError as e:
            logger.warning(f"Invalid client_slug '{client_slug}': {e}")
            return None
        result = repo.get_for_camera(camera_id)
        if result is None:
            return None
        matrix, map_id = result
        H = np.asarray(matrix, dtype=np.float64)
        if H.shape != (3, 3):
            logger.warning(
                f"Unexpected homography shape {H.shape} for ({client_slug}, "
                f"camera_id={camera_id}); treating as missing"
            )
            return None
        return H, map_id
