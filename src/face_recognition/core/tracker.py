"""Face tracking using DeepOCSORT."""

from typing import List, Tuple

import numpy as np
from boxmot import DeepOCSORT  # type: ignore
from loguru import logger


class FaceTracker:
    """Face tracker using DeepOCSORT algorithm.

    This class manages face tracking across video frames, maintaining
    track IDs and trajectories for detected faces.
    """

    def __init__(self, gpu_id: int = 0):
        """Initialize the face tracker.

        Args:
            gpu_id: GPU device ID to use for tracking
        """
        self.gpu_id = gpu_id
        self.tracker = DeepOCSORT(
            device=f'cuda:{gpu_id}',
            custom_features=True,
        )
        logger.info(f"Initialized DeepOCSORT tracker on GPU {gpu_id}")

    def update(
        self,
        boxes: np.ndarray,
        frame: np.ndarray,
        features: np.ndarray
    ) -> Tuple[List, List]:
        """Update tracker with new detections.

        Args:
            boxes: Array of bounding boxes [x1, y1, x2, y2, confidence, class_id]
            frame: Current video frame
            features: Array of face embeddings for each detection

        Returns:
            Tuple of (active_tracks, removed_tracks):
            - active_tracks: List of currently active track objects
            - removed_tracks: List of track IDs that were removed this frame
        """
        if len(boxes) == 0:
            # No detections, update with empty arrays
            self.tracker.update(np.empty((0, 6)), frame, np.empty((0, 512)))
        else:
            self.tracker.update(boxes, frame, features)

        return self.tracker.active_tracks, self.tracker.removed_tracks

    def get_active_tracks(self) -> List:
        """Get list of currently active tracks.

        Returns:
            List of active track objects
        """
        return self.tracker.active_tracks

    def get_removed_tracks(self) -> List:
        """Get list of tracks removed in the last update.

        Returns:
            List of removed track IDs
        """
        return self.tracker.removed_tracks

    def remove_tracks(self, track_ids: List[int]) -> None:
        """Remove specific tracks from the active list.

        Args:
            track_ids: List of track IDs to remove
        """
        if not track_ids:
            return

        self.tracker.active_tracks = [
            t for t in self.tracker.active_tracks
            if t.id not in track_ids
        ]

    def visualize(self, frame: np.ndarray, show_trajectories: bool = True) -> np.ndarray:
        """Visualize tracked faces on the frame.

        Args:
            frame: Input frame to visualize on
            show_trajectories: Whether to show track trajectories

        Returns:
            Frame with visualizations
        """
        visualization_frame = frame.copy()
        self.tracker.plot_results(visualization_frame, show_trajectories=show_trajectories)
        return visualization_frame
