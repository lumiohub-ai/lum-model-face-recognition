"""
Phone Tracker for persistent phone detection across frames.

Assigns and maintains stable TrackIDs for phones to handle:
- Temporary detection failures
- Occlusions
- Brief disappearances

Uses IoU-based matching with configurable tracking parameters.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from loguru import logger


class PhoneTracker:
    """
    Multi-phone tracker using IoU-based association.

    Assigns unique TrackIDs to phones and maintains tracking across frames.
    Provides temporal persistence to handle detection inconsistencies.
    """

    def __init__(
        self,
        max_age: int = 30,  # Shorter than person tracking (phones move faster)
        min_hits: int = 2,  # Lower threshold (phones appear/disappear quickly)
        iou_threshold: float = 0.25,  # Slightly lower (phones are smaller)
    ):
        """
        Initialize Phone Tracker.

        Args:
            max_age: Maximum frames to keep track without updates
            min_hits: Minimum consecutive hits before track is confirmed
            iou_threshold: IoU threshold for track association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        logger.info(
            f"Initializing PhoneTracker "
            f"(max_age={max_age}, min_hits={min_hits}, iou={iou_threshold})"
        )

        # Initialize tracking state
        self.active_tracks: Dict[int, Dict] = {}  # {phone_track_id: track_data}
        self.removed_tracks: List[Dict] = []  # Tracks that left the frame
        self.next_track_id = 1000  # Start at 1000 to differentiate from person IDs
        self.frame_count = 0

        logger.info("PhoneTracker initialized")

    def update(
        self,
        phone_detections: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Update tracker with new phone detections.

        Args:
            phone_detections: List of phone detections from PhoneDetector
                             Each dict contains: bbox, confidence, class_id

        Returns:
            Tuple of (active_tracks, removed_tracks)
            - active_tracks: Currently tracked phones with phone_track_id
            - removed_tracks: Phones that left the frame in this update
        """
        self.frame_count += 1
        self.removed_tracks = []  # Reset removed tracks for this frame

        if not phone_detections:
            # No detections, age all tracks
            logger.debug(f"Frame {self.frame_count}: No phone detections")
            self._age_tracks()
            return self._get_active_tracks(), self.removed_tracks

        # Perform tracking on detections
        tracked_phones = self._track_detections(phone_detections)

        # Update active tracks
        self._update_active_tracks(tracked_phones)

        # Remove old tracks
        self._age_tracks()

        logger.debug(
            f"Frame {self.frame_count}: {len(self.active_tracks)} active phone tracks, "
            f"{len(self.removed_tracks)} removed"
        )

        return self._get_active_tracks(), self.removed_tracks

    def _track_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Perform tracking on phone detections.

        Args:
            detections: List of phone detection dicts

        Returns:
            List of tracked phones with phone_track_id assigned
        """
        tracked = []

        for det in detections:
            bbox = np.array(det['bbox'])

            # Try to match with existing tracks
            matched_track_id = self._match_detection_to_track(bbox)

            if matched_track_id is not None:
                # Update existing track
                phone_track_id = matched_track_id
                self.active_tracks[phone_track_id]['age'] = 0  # Reset age
                self.active_tracks[phone_track_id]['hits'] += 1
            else:
                # Create new phone track
                phone_track_id = self.next_track_id
                self.next_track_id += 1

                self.active_tracks[phone_track_id] = {
                    'phone_track_id': phone_track_id,
                    'age': 0,
                    'hits': 1,
                    'first_frame': self.frame_count
                }

                logger.debug(f"New phone track created: phone_track_id={phone_track_id}")

            # Add tracking info to detection
            tracked_det = det.copy()
            tracked_det['phone_track_id'] = phone_track_id
            tracked_det['frame_num'] = self.frame_count
            tracked.append(tracked_det)

            # Update track bbox and metadata
            self.active_tracks[phone_track_id]['bbox'] = bbox
            self.active_tracks[phone_track_id]['confidence'] = det.get('confidence', 0.0)

        return tracked

    def _match_detection_to_track(self, bbox: NDArray) -> Optional[int]:
        """
        Match a detection bbox to existing tracks using IoU.

        Args:
            bbox: Detection bounding box [x1, y1, x2, y2]

        Returns:
            Matched phone_track_id or None
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

        if best_track_id is not None:
            logger.debug(
                f"Phone matched to existing track {best_track_id} with IoU={best_iou:.3f}"
            )

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
            tracked_detections: List of phone detections with phone_track_id assigned
        """
        # Mark all tracks as not updated
        for track_id in self.active_tracks:
            self.active_tracks[track_id]['updated'] = False

        # Update tracks with new detections
        for det in tracked_detections:
            phone_track_id = det['phone_track_id']
            if phone_track_id in self.active_tracks:
                self.active_tracks[phone_track_id]['updated'] = True
                self.active_tracks[phone_track_id]['last_detection'] = det
                self.active_tracks[phone_track_id]['last_frame'] = self.frame_count

    def _age_tracks(self) -> None:
        """
        Age tracks and remove old ones.

        Tracks that haven't been updated for max_age frames are removed.
        Also maintains predicted phone positions for aging tracks.
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
                    'phone_track_id': track_id,
                    'last_detection': removed_track.get('last_detection'),
                    'total_frames': self.frame_count - removed_track.get('first_frame', 0),
                    'total_hits': removed_track.get('hits', 0)
                })

                logger.debug(
                    f"Phone track {track_id} removed after {removed_track.get('age', 0)} "
                    f"frames of inactivity (total hits: {removed_track.get('hits', 0)})"
                )

    def _get_active_tracks(self) -> List[Dict]:
        """
        Get list of currently active phone tracks.

        Returns phones that are:
        1. Confirmed (met min_hits threshold)
        2. Either currently detected (age=0) OR recently seen (age <= max_age)

        Returns:
            List of active phone track dicts with detection info
        """
        active = []

        for track_id, track_data in self.active_tracks.items():
            # Return confirmed tracks
            if track_data.get('hits', 0) >= self.min_hits:
                # Include both actively detected AND aging tracks (for temporal persistence)
                if 'last_detection' in track_data:
                    track_dict = track_data['last_detection'].copy()
                    track_dict['phone_track_id'] = track_id
                    track_dict['track_age'] = track_data.get('age', 0)
                    track_dict['track_hits'] = track_data.get('hits', 0)

                    # Mark if this is a predicted position (not actively detected)
                    track_dict['is_predicted'] = track_data.get('age', 0) > 0

                    active.append(track_dict)

        return active

    def get_track_info(self, phone_track_id: int) -> Optional[Dict]:
        """
        Get information about a specific phone track.

        Args:
            phone_track_id: Phone track ID to query

        Returns:
            Track info dict or None if not found
        """
        return self.active_tracks.get(phone_track_id)

    def reset(self) -> None:
        """Reset tracker state."""
        self.active_tracks = {}
        self.removed_tracks = []
        self.next_track_id = 1000
        self.frame_count = 0
        logger.info("PhoneTracker state reset")

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
        predicted_tracks = sum(
            1 for t in self.active_tracks.values()
            if t.get('age', 0) > 0 and t.get('hits', 0) >= self.min_hits
        )

        return {
            'frame_count': self.frame_count,
            'total_active_tracks': total_tracks,
            'confirmed_tracks': confirmed_tracks,
            'predicted_tracks': predicted_tracks,
            'next_track_id': self.next_track_id,
            'tracks_removed_this_frame': len(self.removed_tracks)
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PhoneTracker("
            f"active_tracks={len(self.active_tracks)}, "
            f"frame={self.frame_count})"
        )
