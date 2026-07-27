"""Cross-camera track ID allocation."""

import threading

from loguru import logger


class GlobalTrackIDGenerator:
    """Thread-safe global track ID generator for cross-camera unique IDs.

    Ensures track IDs are globally unique across all cameras by using
    a shared atomic counter with thread-safe increment operations.
    """

    def __init__(self, start_id: int = 1):
        """Initialize global track ID generator.

        Args:
            start_id: Starting track ID (default: 1)
        """
        self._current_id = start_id
        self._lock = threading.Lock()
        logger.debug(f"GlobalTrackIDGenerator initialized (start_id={start_id})")

    def get_next_id(self) -> int:
        """Get next globally unique track ID (thread-safe).

        Returns:
            Next unique track ID
        """
        with self._lock:
            track_id = self._current_id
            self._current_id += 1
            return track_id
