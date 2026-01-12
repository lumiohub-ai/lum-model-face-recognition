"""
Global Track Manager - Phase 0 (Stub Implementation).

This is a stub implementation for Phase 0 that:
- Checks the ENABLE_GLOBAL_TRACKING feature flag
- Logs track lifecycle events for baseline metrics
- Does NOT modify any tracking behavior (zero impact)

Phase 0 Goals:
- Measure baseline: track duration, face visibility rate, tracks per hour
- Collect data for Phase 1 design decisions
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
import numpy as np


class GlobalTrackManager:
    """
    Stub implementation for cross-camera global track management.

    Phase 0: Only logs events, no actual cross-camera matching.
    This allows us to collect baseline metrics without changing behavior.
    """

    def __init__(self):
        """Initialize GlobalTrackManager in stub mode."""
        # Feature flag (default: false)
        self.enabled = os.getenv('ENABLE_GLOBAL_TRACKING', 'false').lower() == 'true'

        # Metrics tracking for baseline analysis
        self.track_stats: Dict[tuple, Dict[str, Any]] = {}  # (camera_id, local_track_id) -> stats

        # Counters for baseline metrics
        self.total_tracks_created = 0
        self.total_tracks_removed = 0
        self.total_faces_detected = 0
        self.total_faces_not_visible = 0

        if self.enabled:
            logger.info("GlobalTrackManager initialized (PHASE 0 - STUB MODE)")
            logger.info("Collecting baseline metrics: track lifecycles, face visibility")
        else:
            logger.debug("GlobalTrackManager disabled (ENABLE_GLOBAL_TRACKING=false)")

    def on_track_created(
        self,
        camera_id: int,
        local_track_id: int,
        bbox: Optional[np.ndarray] = None,
        frame_num: int = 0
    ) -> None:
        """
        Log when a new track is created.

        Args:
            camera_id: Camera identifier
            local_track_id: Local track ID from PersonTracker
            bbox: Person bounding box
            frame_num: Current frame number
        """
        if not self.enabled:
            return

        track_key = (camera_id, local_track_id)
        self.track_stats[track_key] = {
            'camera_id': camera_id,
            'local_track_id': local_track_id,
            'created_at': datetime.now(),
            'first_frame': frame_num,
            'total_frames': 0,
            'faces_detected': 0,
            'faces_not_visible': 0
        }

        self.total_tracks_created += 1

        logger.info(
            f"TRACK_CREATED | camera={camera_id} local_id={local_track_id} "
            f"duration=0s frame={frame_num}"
        )

    def on_track_removed(
        self,
        camera_id: int,
        local_track_id: int,
        total_frames: int = 0
    ) -> None:
        """
        Log when a track is removed.

        Args:
            camera_id: Camera identifier
            local_track_id: Local track ID
            total_frames: Total frames this track was active
        """
        if not self.enabled:
            return

        track_key = (camera_id, local_track_id)
        stats = self.track_stats.get(track_key)

        if stats:
            duration = (datetime.now() - stats['created_at']).total_seconds()
            face_detection_rate = (
                stats['faces_detected'] / max(stats['total_frames'], 1)
            )

            logger.info(
                f"TRACK_REMOVED | camera={camera_id} local_id={local_track_id} "
                f"duration={duration:.1f}s frames={stats['total_frames']} "
                f"face_rate={face_detection_rate:.2%}"
            )

            # Clean up stats
            del self.track_stats[track_key]
        else:
            # Track created before GlobalTrackManager was initialized
            logger.info(
                f"TRACK_REMOVED | camera={camera_id} local_id={local_track_id} "
                f"duration=unknown"
            )

        self.total_tracks_removed += 1

    def on_face_detected(
        self,
        camera_id: int,
        local_track_id: int,
        quality: float = 0.0,
        recognized: bool = False,
        identity: Optional[str] = None
    ) -> None:
        """
        Log when a face is detected for a track.

        Args:
            camera_id: Camera identifier
            local_track_id: Local track ID
            quality: Face detection quality/confidence
            recognized: Whether face was recognized
            identity: Identity name if recognized
        """
        if not self.enabled:
            return

        track_key = (camera_id, local_track_id)
        if track_key in self.track_stats:
            self.track_stats[track_key]['faces_detected'] += 1

        self.total_faces_detected += 1

        logger.debug(
            f"FACE_DETECTED | camera={camera_id} track={local_track_id} "
            f"quality={quality:.2f} recognized={recognized} "
            f"identity={identity if identity else 'unknown'}"
        )

    def on_face_not_visible(
        self,
        camera_id: int,
        local_track_id: int
    ) -> None:
        """
        Log when face is not visible for a track.

        Args:
            camera_id: Camera identifier
            local_track_id: Local track ID
        """
        if not self.enabled:
            return

        track_key = (camera_id, local_track_id)
        if track_key in self.track_stats:
            self.track_stats[track_key]['faces_not_visible'] += 1

        self.total_faces_not_visible += 1

        logger.debug(
            f"FACE_NOT_VISIBLE | camera={camera_id} track={local_track_id}"
        )

    def on_track_update(
        self,
        camera_id: int,
        local_track_id: int
    ) -> None:
        """
        Update track frame counter.

        Args:
            camera_id: Camera identifier
            local_track_id: Local track ID
        """
        if not self.enabled:
            return

        track_key = (camera_id, local_track_id)
        if track_key in self.track_stats:
            self.track_stats[track_key]['total_frames'] += 1

    def get_baseline_metrics(self) -> Dict[str, Any]:
        """
        Get baseline metrics for analysis.

        Returns:
            Dictionary with baseline statistics
        """
        if not self.enabled:
            return {}

        total_frames = sum(
            stats['total_frames']
            for stats in self.track_stats.values()
        )

        avg_track_duration = 0.0
        if self.track_stats:
            durations = [
                (datetime.now() - stats['created_at']).total_seconds()
                for stats in self.track_stats.values()
            ]
            avg_track_duration = sum(durations) / len(durations)

        face_visibility_rate = 0.0
        if self.total_faces_detected + self.total_faces_not_visible > 0:
            face_visibility_rate = (
                self.total_faces_detected /
                (self.total_faces_detected + self.total_faces_not_visible)
            )

        return {
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_removed': self.total_tracks_removed,
            'active_tracks': len(self.track_stats),
            'avg_track_duration_sec': avg_track_duration,
            'total_faces_detected': self.total_faces_detected,
            'total_faces_not_visible': self.total_faces_not_visible,
            'face_visibility_rate': face_visibility_rate
        }

    def log_baseline_summary(self) -> None:
        """Log a summary of baseline metrics."""
        if not self.enabled:
            return

        metrics = self.get_baseline_metrics()

        logger.info(
            f"BASELINE_METRICS | "
            f"tracks_created={metrics['total_tracks_created']} "
            f"tracks_removed={metrics['total_tracks_removed']} "
            f"active={metrics['active_tracks']} "
            f"avg_duration={metrics['avg_track_duration_sec']:.1f}s "
            f"face_visibility={metrics['face_visibility_rate']:.1%}"
        )
