"""
Person Tracker using BoT-SORT (or ByteTrack/OC-SORT).

Assigns and maintains stable TrackIDs for persons across video frames.
Integrates with Ultralytics tracking API for multi-object tracking.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from loguru import logger


class PersonTracker:
    """
    Multi-person tracker using BoT-SORT algorithm.

    Assigns unique TrackIDs to persons and maintains tracking across frames.
    Handles occlusions, brief disappearances, and track lifecycle management.
    """

    def __init__(
        self,
        tracker_type: str = "botsort",
        max_age: int = 120,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        model_size: str = "s",
        global_id_generator: Optional['GlobalTrackIDGenerator'] = None
    ):
        """
        Initialize Person Tracker.

        Args:
            tracker_type: Tracking algorithm ('botsort', 'bytetrack', 'ocsort')
            max_age: Maximum frames to keep track without updates (in frames, not seconds)
            min_hits: Minimum consecutive hits before track is confirmed
            iou_threshold: IoU threshold for track association
            model_size: YOLOv8 model size (needed for tracker initialization)
            global_id_generator: Optional global ID generator for cross-camera unique IDs
        """
        self.tracker_type = tracker_type.lower()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.global_id_generator = global_id_generator

        logger.info(
            f"Initializing PersonTracker with {tracker_type.upper()} "
            f"(max_age={max_age}, min_hits={min_hits}, iou={iou_threshold}, "
            f"global_ids={'enabled' if global_id_generator else 'disabled'})"
        )

        # Initialize tracking state
        self.active_tracks: Dict[int, Dict] = {}  # {track_id: track_data}
        self.removed_tracks: List[Dict] = []  # Tracks that left the frame
        self.next_track_id = 1  # Only used if global_id_generator is None
        self.frame_count = 0

        # Track configuration for ultralytics
        self.tracker_config = {
            'tracker_type': self.tracker_type,
            'max_age': max_age,
            'min_hits': min_hits,
            'iou_threshold': iou_threshold
        }

        logger.info(f"PersonTracker initialized with {tracker_type.upper()}")

    def update(
        self,
        detections: List[Dict],
        frame: Optional[NDArray] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Update tracker with new detections.

        Args:
            detections: List of person detections from PersonDetector
                       Each dict contains: bbox, confidence, keypoints
            frame: Optional frame for appearance-based tracking

        Returns:
            Tuple of (active_tracks, removed_tracks)
            - active_tracks: Currently tracked persons
            - removed_tracks: Persons that left the frame in this update
        """
        self.frame_count += 1
        self.removed_tracks = []  # Reset removed tracks for this frame

        if not detections:
            # No detections, age all tracks
            logger.debug(f"Frame {self.frame_count}: No detections")
            self._age_tracks()
            return self._get_active_tracks(), self.removed_tracks

        # Convert detections to tracking format
        track_results = self._track_detections(detections)

        # Update active tracks
        self._update_active_tracks(track_results)

        # Remove old tracks
        self._age_tracks()

        logger.debug(
            f"Frame {self.frame_count}: {len(self.active_tracks)} active tracks, "
            f"{len(self.removed_tracks)} removed"
        )

        return self._get_active_tracks(), self.removed_tracks

    def _track_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Perform tracking on detections.

        Args:
            detections: List of detection dicts

        Returns:
            List of tracked detections with track_id assigned
        """
        # Simple IoU-based tracking for MVP
        # For production, integrate with Ultralytics YOLO.track()

        tracked = []

        for det in detections:
            bbox = np.array(det['bbox'])

            # Try to match with existing tracks
            matched_track_id = self._match_detection_to_track(bbox)

            if matched_track_id is not None:
                # Update existing track
                track_id = matched_track_id
                self.active_tracks[track_id]['age'] = 0  # Reset age
                self.active_tracks[track_id]['hits'] += 1
            else:
                # Create new track with global or local ID
                if self.global_id_generator:
                    track_id = self.global_id_generator.get_next_id()
                else:
                    track_id = self.next_track_id
                    self.next_track_id += 1

                self.active_tracks[track_id] = {
                    'track_id': track_id,
                    'age': 0,
                    'hits': 1,
                    'first_frame': self.frame_count
                }

            # Add tracking info to detection
            tracked_det = det.copy()
            tracked_det['track_id'] = track_id
            tracked_det['frame_num'] = self.frame_count
            tracked.append(tracked_det)

            # Update track bbox
            self.active_tracks[track_id]['bbox'] = bbox

        return tracked

    def _match_detection_to_track(self, bbox: NDArray) -> Optional[int]:
        """
        Match a detection bbox to existing tracks using IoU.

        Args:
            bbox: Detection bounding box [x1, y1, x2, y2]

        Returns:
            Matched track_id or None
        """
        best_iou = 0.0
        best_track_id = None

        for track_id, track_data in self.active_tracks.items():
            if 'bbox' not in track_data:
                continue

            iou = self._calculate_iou(bbox, track_data['bbox'])

            if iou > self.iou_threshold and iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        return best_track_id

    @staticmethod
    def _calculate_iou(bbox1: NDArray, bbox2: NDArray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bboxes.

        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]

        Returns:
            IoU value (0.0 to 1.0)
        """
        # Intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _update_active_tracks(self, tracked_detections: List[Dict]) -> None:
        """
        Update active tracks with new detections.

        Args:
            tracked_detections: List of detections with track_id assigned
        """
        # Mark all tracks as not updated
        for track_id in self.active_tracks:
            self.active_tracks[track_id]['updated'] = False

        # Update tracks with new detections
        for det in tracked_detections:
            track_id = det['track_id']
            if track_id in self.active_tracks:
                self.active_tracks[track_id]['updated'] = True
                self.active_tracks[track_id]['last_detection'] = det
                self.active_tracks[track_id]['last_frame'] = self.frame_count

    def _age_tracks(self) -> None:
        """
        Age tracks and remove old ones.

        Tracks that haven't been updated for max_age frames are removed.
        """
        tracks_to_remove = []

        for track_id, track_data in self.active_tracks.items():
            if not track_data.get('updated', False):
                track_data['age'] += 1

                # Remove track if too old
                if track_data['age'] > self.max_age:
                    tracks_to_remove.append(track_id)

        # Remove old tracks
        for track_id in tracks_to_remove:
            removed_track = self.active_tracks.pop(track_id)

            # Only add to removed_tracks if track was confirmed (min_hits)
            if removed_track.get('hits', 0) >= self.min_hits:
                self.removed_tracks.append({
                    'track_id': track_id,
                    'last_detection': removed_track.get('last_detection'),
                    'total_frames': self.frame_count - removed_track.get('first_frame', 0),
                    'total_hits': removed_track.get('hits', 0)
                })

                logger.debug(
                    f"Track {track_id} removed after {removed_track.get('age', 0)} frames "
                    f"of inactivity (total frames: {removed_track.get('hits', 0)})"
                )

    def _get_active_tracks(self) -> List[Dict]:
        """
        Get list of currently active tracks.

        Returns:
            List of active track dicts with detection info
        """
        active = []

        for track_id, track_data in self.active_tracks.items():
            # Only return confirmed tracks (min_hits) that are currently being detected (age = 0)
            # This prevents ghost bboxes from aging tracks that left the frame
            if track_data.get('hits', 0) >= self.min_hits:
                if 'last_detection' in track_data and track_data.get('age', 0) == 0:
                    track_dict = track_data['last_detection'].copy()
                    track_dict['track_id'] = track_id
                    track_dict['track_age'] = track_data.get('age', 0)
                    track_dict['track_hits'] = track_data.get('hits', 0)
                    active.append(track_dict)

        return active

    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """
        Get information about a specific track.

        Args:
            track_id: Track ID to query

        Returns:
            Track info dict or None if not found
        """
        return self.active_tracks.get(track_id)

    def reset(self) -> None:
        """Reset tracker state."""
        self.active_tracks = {}
        self.removed_tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        logger.info("Tracker state reset")

    def get_statistics(self) -> Dict:
        """
        Get tracking statistics.

        Returns:
            Dictionary with tracking stats
        """
        total_tracks = len(self.active_tracks)
        confirmed_tracks = sum(
            1 for t in self.active_tracks.values()
            if t.get('hits', 0) >= self.min_hits
        )

        return {
            'frame_count': self.frame_count,
            'total_active_tracks': total_tracks,
            'confirmed_tracks': confirmed_tracks,
            'next_track_id': self.next_track_id,
            'tracks_removed_this_frame': len(self.removed_tracks)
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PersonTracker(type={self.tracker_type}, "
            f"active_tracks={len(self.active_tracks)}, "
            f"frame={self.frame_count})"
        )
