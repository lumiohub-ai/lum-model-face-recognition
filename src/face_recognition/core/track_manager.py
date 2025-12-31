"""Track lifecycle management for face tracking."""

from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pytz
from loguru import logger


class TrackManager:
    """Manages the lifecycle of face tracks.

    This class handles:
    - Track history storage (embeddings, boxes, crops, landmarks)
    - Track expiration based on lifetime
    - Track cleanup and cache management
    - Track filtering and validation
    """

    def __init__(self, timezone: str = "UTC", max_track_lifetime_seconds: int = 120):
        """Initialize the track manager.

        Args:
            timezone: Timezone for track timestamps
            max_track_lifetime_seconds: Maximum time a track can be active (seconds)
        """
        self.timezone = pytz.timezone(timezone)
        self.max_track_lifetime_seconds = max_track_lifetime_seconds

        # Track history storage - no limits (production tracks are 30-60 sec)
        self.track_emb_frame_history: Dict[int, Dict[int, np.ndarray]] = {}
        self.track_boxes_frame: Dict[int, Dict[int, List[float]]] = {}
        self.track_road_history: Dict[int, List[Tuple[int, int]]] = {}
        self.track_crop_history: Dict[int, Dict[int, np.ndarray]] = {}
        self.track_landmarks_history: Dict[int, Dict[int, np.ndarray]] = {}
        self.track_frame_history: Dict[int, Dict[int, np.ndarray]] = {}  # Full frames

        # Track metadata
        self.all_tracks: Set[int] = set()
        self.id_appear_time: Dict[int, datetime] = {}
        # Increase from 5000 to 50000 to prevent premature ID reuse
        self.passed_tracks: deque = deque(maxlen=50000)

    def register_track(self, track_id: int) -> None:
        """Register a new track.

        Args:
            track_id: Track ID to register
        """
        now = datetime.now(self.timezone)
        self.all_tracks.add(track_id)

        if track_id not in self.id_appear_time:
            self.id_appear_time[track_id] = now

    def store_track_data(
        self,
        track_id: int,
        frame_num: int,
        embedding: np.ndarray,
        box: List[float],
        center: Tuple[int, int],
        face_crop: Optional[np.ndarray] = None,
        full_frame: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> None:
        """Store track data for a specific frame.

        Args:
            track_id: Track ID
            frame_num: Frame number
            embedding: Face embedding
            box: Bounding box [x1, y1, x2, y2, confidence, class_id]
            center: Center point of the bounding box
            face_crop: Cropped face image
            full_frame: Full frame image
            landmarks: Facial landmarks
        """
        # Store embedding
        self.track_emb_frame_history.setdefault(track_id, {})[frame_num] = embedding

        # Store bounding box
        self.track_boxes_frame.setdefault(track_id, {})[frame_num] = box

        # Store trajectory center point
        self.track_road_history.setdefault(track_id, []).append(center)

        # Store face crop if provided
        if face_crop is not None:
            self.track_crop_history.setdefault(track_id, {})[frame_num] = face_crop

        # Store full frame if provided
        if full_frame is not None:
            self.track_frame_history.setdefault(track_id, {})[frame_num] = full_frame

        # Store landmarks if provided
        if landmarks is not None:
            self.track_landmarks_history.setdefault(track_id, {})[frame_num] = landmarks

    def get_track_embeddings(self, track_id: int) -> Dict[int, np.ndarray]:
        """Get all embeddings for a track.

        Args:
            track_id: Track ID

        Returns:
            Dictionary mapping frame numbers to embeddings
        """
        return self.track_emb_frame_history.get(track_id, {})

    def get_track_landmarks(self, track_id: int) -> Dict[int, np.ndarray]:
        """Get all landmarks for a track.

        Args:
            track_id: Track ID

        Returns:
            Dictionary mapping frame numbers to landmarks
        """
        return self.track_landmarks_history.get(track_id, {})

    def get_track_crop(self, track_id: int, frame_num: int) -> Optional[np.ndarray]:
        """Get face crop for a specific track and frame.

        Args:
            track_id: Track ID
            frame_num: Frame number

        Returns:
            Face crop image or None if not found
        """
        return self.track_crop_history.get(track_id, {}).get(frame_num, None)

    def get_track_frame(self, track_id: int, frame_num: int) -> Optional[np.ndarray]:
        """Get full frame for a specific track and frame.

        Args:
            track_id: Track ID
            frame_num: Frame number (actual frame number from recognition)

        Returns:
            Full frame image or None if not found
        """
        # Direct retrieval from frame history - no caching, no complexity
        # frame_num comes directly from recognizer and matches storage keys
        return self.track_frame_history.get(track_id, {}).get(frame_num, None)

    def set_best_frame(self, track_id: int, frame_num: int, frame: np.ndarray) -> None:
        """Store the best quality frame for a track.

        This frame is preserved even when it falls out of the frame window,
        ensuring it's always available for sending to the dashboard.

        Args:
            track_id: Track ID
            frame_num: Frame number of the best frame
            frame: The full frame image
        """
        self.track_best_frame[track_id] = frame
        self.track_best_frame_num[track_id] = frame_num

    def get_track_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """Get trajectory (center points) for a track.

        Args:
            track_id: Track ID

        Returns:
            List of center points
        """
        return self.track_road_history.get(track_id, [])

    def is_track_expired(self, track_id: int, active_track_ids: List[int]) -> bool:
        """Check if a track has exceeded its maximum lifetime.

        Args:
            track_id: Track ID to check
            active_track_ids: List of currently active track IDs

        Returns:
            True if track is expired, False otherwise
        """
        if track_id not in self.id_appear_time:
            return False

        if track_id not in active_track_ids:
            return False

        now = datetime.now(self.timezone)
        track_age = (now - self.id_appear_time[track_id]).total_seconds()

        return track_age > self.max_track_lifetime_seconds

    def find_expired_tracks(self, active_tracks: List) -> List[int]:
        """Find all tracks that have exceeded their maximum lifetime.

        Args:
            active_tracks: List of active track objects

        Returns:
            List of expired track IDs
        """
        now = datetime.now(self.timezone)
        expired_track_ids = []

        for track in active_tracks:
            if track.id in self.id_appear_time:
                track_age = (now - self.id_appear_time[track.id]).total_seconds()
                if track_age > self.max_track_lifetime_seconds:
                    emb_count = len(self.track_emb_frame_history.get(track.id, {}))
                    logger.debug(
                        f"Track {track.id} has expired after {track_age:.1f}s. "
                        f"Embeddings stored: {emb_count}"
                    )
                    expired_track_ids.append(track.id)

        return expired_track_ids

    def mark_track_passed(self, track_id: int) -> None:
        """Mark a track as having been processed.

        Args:
            track_id: Track ID to mark as passed
        """
        self.passed_tracks.append(track_id)

    def is_track_passed(self, track_id: int) -> bool:
        """Check if a track has already been processed.

        Args:
            track_id: Track ID to check

        Returns:
            True if track has been processed, False otherwise
        """
        return track_id in self.passed_tracks

    def get_unprocessed_tracks(self, last_frame: bool = False) -> List[int]:
        """Get list of tracks that haven't been processed yet.

        Args:
            last_frame: If True, return all unprocessed tracks; if False, return empty list

        Returns:
            List of unprocessed track IDs
        """
        if last_frame:
            return sorted(list(self.all_tracks - set(self.passed_tracks)))
        else:
            return []

    def delete_track_cache(self, track_id: int) -> None:
        """Delete all cached data for a specific track.

        Args:
            track_id: Track ID to delete
        """
        try:
            for storage_dict in [
                self.track_emb_frame_history,
                self.track_boxes_frame,
                self.track_crop_history,
                self.track_road_history,
                self.id_appear_time,
                self.track_landmarks_history,
                self.track_frame_history
            ]:
                storage_dict.pop(track_id, None)
        except KeyError:
            pass

    def get_track_appear_time(self, track_id: int) -> Optional[datetime]:
        """Get the time when a track first appeared.

        Args:
            track_id: Track ID

        Returns:
            Appearance time or None if not found
        """
        return self.id_appear_time.get(track_id)

    def has_embeddings(self, track_id: int) -> bool:
        """Check if a track has any stored embeddings.

        Args:
            track_id: Track ID

        Returns:
            True if track has embeddings, False otherwise
        """
        track_embeddings = self.track_emb_frame_history.get(track_id, {})
        return len(track_embeddings) > 0
