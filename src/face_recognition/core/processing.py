"""Frame processing and recognition handling.

This module handles:
- Processing frames through camera engines
- Handling recognized/unrecognized persons
- Frame annotation with detection results
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

import numpy as np
from loguru import logger

from ..logging.entry_logger import EntryLogger
from person_tracking.video.frame_annotator import FrameAnnotator
from person_tracking.core.global_track_manager import GlobalTrackManager

if TYPE_CHECKING:
    from ..camera_engine import CameraEngine


class FrameProcessor:
    """Processes frames and handles recognition results.

    Coordinates between camera engines, entry logging, and frame annotation.
    """

    def __init__(
        self,
        camera_engines: List[CameraEngine],
        camera_configs: List[Dict[str, Any]],
        entry_logger: EntryLogger,
        frame_annotator: FrameAnnotator,
        global_track_manager: Optional[GlobalTrackManager] = None
    ):
        """Initialize frame processor.

        Args:
            camera_engines: List of CameraEngine instances
            camera_configs: List of camera configuration dicts
            entry_logger: EntryLogger for recording attendance
            frame_annotator: FrameAnnotator for visual annotation
            global_track_manager: Optional global track manager for cross-camera IDs
        """
        self.camera_engines = camera_engines
        self.camera_configs = camera_configs
        self.entry_logger = entry_logger
        self.frame_annotator = frame_annotator
        self.global_track_manager = global_track_manager

        # Performance tracking
        self.start_time = time.time()
        self.total_frames = 0

        # FPS sampling for system monitoring (min/max/avg per interval)
        self._interval_start_time = time.time()
        self._interval_frame_count = 0

    def process_all_frames(
        self,
        frames: List[Tuple[int, np.ndarray, int]]
    ) -> List[Tuple[int, np.ndarray]]:
        """Process frames from all cameras.

        Args:
            frames: List of (camera_idx, frame, frame_num) tuples

        Returns:
            List of (camera_idx, annotated_frame) tuples
        """
        annotated_frames = []

        for camera_idx, frame, frame_num in frames:
            engine = self.camera_engines[camera_idx]

            # Process frame through camera engine
            recognized, processed = engine.process_frame(frame, frame_num)

            # Handle recognized persons
            for person in recognized:
                self._handle_recognized_person(person)

            # Annotate frame
            annotated = self._annotate_frame(processed, engine)
            annotated_frames.append((camera_idx, annotated))

            self.total_frames += 1
            self._interval_frame_count += 1

        return annotated_frames

    def _handle_recognized_person(self, person: Dict) -> None:
        """Handle a recognized/unrecognized person.

        Args:
            person: Dictionary with recognition result
        """
        name = person['name']
        status = person['status']  # IN or OUT
        appear_time = person['appear_time']
        camera_name = person['camera_name']
        camera_id = person['camera_id']
        face_image = person.get('face_image')
        proof_image = person.get('proof_image')

        if person['recognized'] and name:
            # Log recognized person
            recorded = self.entry_logger.log_person_entry(
                name=name,
                status=status,
                appear_time=appear_time,
                camera_name=camera_name,
                camera_id=camera_id,
                proof_image=proof_image
            )

            if recorded:
                logger.info(
                    f"ATTENDANCE | {name} {status} at {camera_name} | "
                    f"Confidence: {person['confidence']:.2f}"
                )
        else:
            # Send unrecognized face (only if enabled for this camera)
            send_unrecognized = person.get('send_unrecognized', False)
            if send_unrecognized and face_image is not None and face_image.size > 0:
                self.entry_logger.send_unrecognized_face(
                    face=face_image,
                    status=status
                )
                logger.info(f"UNRECOGNIZED | Sent face from {camera_name} ({status})")

    def _annotate_frame(self, frame: np.ndarray, engine: CameraEngine) -> np.ndarray:
        """Annotate frame with detections and status.

        Args:
            frame: Processed frame
            engine: Camera engine with state

        Returns:
            Annotated frame
        """
        # Get all person states
        states = engine.state_manager.get_all_states()

        person_states = []
        for state in states:
            # Get track info to check if person is currently visible
            track_info = engine.person_tracker.get_track_info(state.track_id)
            if track_info is None:
                continue

            # Remove bbox immediately when person not detected in current frame (age > 0)
            age = track_info.get('age', 0)
            if age > 0:
                continue

            track_data = engine.track_manager.get_track_data(state.track_id)
            bbox = engine.track_manager.get_latest_bbox(state.track_id)
            keypoints = None
            if track_data:
                kp_history = track_data.get('keypoints', {})
                if kp_history:
                    if isinstance(kp_history, dict):
                        latest_frame = max(kp_history.keys())
                        keypoints = kp_history[latest_frame]
                    elif isinstance(kp_history, list) and len(kp_history) > 0:
                        keypoints = kp_history[-1]

            # Get global ID for display (if global tracking is enabled)
            global_id = None
            if self.global_track_manager and self.global_track_manager.enabled:
                global_id = self.global_track_manager.get_global_id(
                    engine.camera_id, state.track_id
                )

            person_states.append({
                'track_id': state.track_id,
                'global_id': global_id,
                'bbox': bbox if bbox is not None else [0, 0, 0, 0],
                'keypoints': keypoints,
                'identity': state.identity,
                'identity_locked': state.identity_locked,
                'track_age': age,
                'in_current_frame': (age == 0),
                'last_detected_action': state.last_detected_action,
            })

        # Calculate FPS
        elapsed = time.time() - self.start_time
        fps = self.total_frames / elapsed if elapsed > 0 else 0

        # Annotate frame
        annotated = self.frame_annotator.annotate_frame(
            frame=frame,
            person_states=person_states,
            fps=fps
        )

        return annotated

    def log_metrics(self) -> None:
        """Log baseline metrics summary."""
        if self.global_track_manager and self.global_track_manager.enabled:
            self.global_track_manager.log_baseline_summary()

    def sample_fps(self) -> Dict:
        """Sample FPS for the current interval and reset counters.

        Returns:
            Dict with avg_fps, min_fps, max_fps for the interval.
        """
        current_time = time.time()
        elapsed = current_time - self._interval_start_time

        if elapsed > 0 and self._interval_frame_count > 0:
            fps = self._interval_frame_count / elapsed
        else:
            fps = 0.0

        # Reset for next interval
        self._interval_start_time = current_time
        self._interval_frame_count = 0

        return {"avg_fps": fps, "min_fps": fps, "max_fps": fps}

    def get_fps(self) -> float:
        """Get current FPS."""
        elapsed = time.time() - self.start_time
        return self.total_frames / elapsed if elapsed > 0 else 0
