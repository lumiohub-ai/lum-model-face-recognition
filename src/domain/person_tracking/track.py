"""
Person Track Manager.

Manages track history and stores per-track data across frames including:
- Bounding boxes per frame
- Keypoints per frame
- Face crops (for recognition)
- Trajectories
- Timestamps
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime
from numpy.typing import NDArray
from loguru import logger


class PersonTrackManager:
    """
    Manages historical data for person tracks.

    Stores comprehensive track information for later processing:
    - Detection history (bboxes, keypoints per frame)
    - Face crops for recognition
    - Movement trajectories
    - Timestamps
    """

    def __init__(self, max_history_frames: int = 300):
        """
        Initialize Track Manager.

        Args:
            max_history_frames: Maximum frames to store per track
        """
        self.max_history_frames = max_history_frames

        # Track data storage
        # Format: {track_id: {frame_num: data}}
        self.track_bbox_history: Dict[int, Dict[int, NDArray]] = defaultdict(dict)
        self.track_keypoints_history: Dict[int, Dict[int, NDArray]] = defaultdict(dict)
        self.track_confidence_history: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.track_crop_history: Dict[int, Dict[int, NDArray]] = defaultdict(dict)

        # Track metadata
        # Format: {track_id: value}
        self.track_first_seen: Dict[int, datetime] = {}
        self.track_last_seen: Dict[int, datetime] = {}
        self.track_trajectories: Dict[int, List[tuple]] = defaultdict(list)

        # Track state
        self.track_identity: Dict[int, Optional[str]] = {}  # Person name if recognized

        logger.debug(
            f"PersonTrackManager initialized (max_history={max_history_frames} frames)"
        )

    def add_track_detection(
        self,
        track_id: int,
        frame_num: int,
        bbox: NDArray,
        keypoints: NDArray,
        confidence: float,
        crop: Optional[NDArray] = None
    ) -> None:
        """
        Add detection data for a track.

        Args:
            track_id: Track identifier
            frame_num: Frame number
            bbox: Bounding box [x1, y1, x2, y2]
            keypoints: Pose keypoints (17, 3)
            confidence: Detection confidence
            crop: Optional person crop image
        """
        # Store detection data
        self.track_bbox_history[track_id][frame_num] = bbox
        self.track_keypoints_history[track_id][frame_num] = keypoints
        self.track_confidence_history[track_id][frame_num] = confidence

        if crop is not None:
            self.track_crop_history[track_id][frame_num] = crop

        # Update timestamps
        now = datetime.now()
        if track_id not in self.track_first_seen:
            self.track_first_seen[track_id] = now
        self.track_last_seen[track_id] = now

        # Update trajectory (center of bbox)
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.track_trajectories[track_id].append((center_x, center_y, frame_num))

        # Limit history size
        self._limit_history(track_id)

    def _limit_history(self, track_id: int) -> None:
        """
        Limit history size to max_history_frames.

        Args:
            track_id: Track to limit
        """
        # Bbox history
        if len(self.track_bbox_history[track_id]) > self.max_history_frames:
            frames = sorted(self.track_bbox_history[track_id].keys())
            frames_to_remove = frames[:-self.max_history_frames]
            for frame in frames_to_remove:
                del self.track_bbox_history[track_id][frame]

        # Keypoints history
        if len(self.track_keypoints_history[track_id]) > self.max_history_frames:
            frames = sorted(self.track_keypoints_history[track_id].keys())
            frames_to_remove = frames[:-self.max_history_frames]
            for frame in frames_to_remove:
                del self.track_keypoints_history[track_id][frame]

        # Confidence history
        if len(self.track_confidence_history[track_id]) > self.max_history_frames:
            frames = sorted(self.track_confidence_history[track_id].keys())
            frames_to_remove = frames[:-self.max_history_frames]
            for frame in frames_to_remove:
                del self.track_confidence_history[track_id][frame]

        # Crop history (keep fewer crops due to memory)
        max_crops = min(50, self.max_history_frames)
        if len(self.track_crop_history[track_id]) > max_crops:
            frames = sorted(self.track_crop_history[track_id].keys())
            frames_to_remove = frames[:-max_crops]
            for frame in frames_to_remove:
                del self.track_crop_history[track_id][frame]

        # Trajectory
        if len(self.track_trajectories[track_id]) > self.max_history_frames:
            self.track_trajectories[track_id] = \
                self.track_trajectories[track_id][-self.max_history_frames:]

    def get_track_data(
        self,
        track_id: int,
        num_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get all data for a track.

        Args:
            track_id: Track identifier
            num_frames: Optional limit to last N frames

        Returns:
            Dictionary with track data
        """
        if track_id not in self.track_bbox_history:
            return {}

        # Get frame numbers
        frames = sorted(self.track_bbox_history[track_id].keys())
        if num_frames:
            frames = frames[-num_frames:]

        # Collect data
        data = {
            'track_id': track_id,
            'frames': frames,
            'bboxes': [self.track_bbox_history[track_id][f] for f in frames],
            'keypoints': [self.track_keypoints_history[track_id][f] for f in frames],
            'confidences': [self.track_confidence_history[track_id].get(f, 0.0) for f in frames],
            'trajectory': self.track_trajectories[track_id][-num_frames:] if num_frames else self.track_trajectories[track_id],
            'first_seen': self.track_first_seen.get(track_id),
            'last_seen': self.track_last_seen.get(track_id),
            'identity': self.track_identity.get(track_id),
            'total_frames': len(frames)
        }

        # Add crops if available
        crops = []
        for f in frames:
            if f in self.track_crop_history[track_id]:
                crops.append((f, self.track_crop_history[track_id][f]))
        if crops:
            data['crops'] = crops

        return data

    def remove_track(self, track_id: int) -> Dict[str, Any]:
        """
        Remove a track and return its final data.

        Args:
            track_id: Track identifier

        Returns:
            Final track data dictionary
        """
        # Get final data
        final_data = self.get_track_data(track_id)

        # Clean up
        if track_id in self.track_bbox_history:
            del self.track_bbox_history[track_id]
        if track_id in self.track_keypoints_history:
            del self.track_keypoints_history[track_id]
        if track_id in self.track_confidence_history:
            del self.track_confidence_history[track_id]
        if track_id in self.track_crop_history:
            del self.track_crop_history[track_id]
        if track_id in self.track_trajectories:
            del self.track_trajectories[track_id]
        if track_id in self.track_first_seen:
            del self.track_first_seen[track_id]
        if track_id in self.track_last_seen:
            del self.track_last_seen[track_id]
        if track_id in self.track_identity:
            del self.track_identity[track_id]

        return final_data
