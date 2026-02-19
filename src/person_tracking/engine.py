"""PersonTrackingEngine - Main orchestrator for the person tracking pipeline.

This module coordinates all components for person detection, tracking,
face recognition, and phone usage detection.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from loguru import logger

# Core components
from .core.person_detector import PersonDetector
from .core.person_tracker import PersonTracker
from .core.track_manager import PersonTrackManager
from .core.phone_detector import PhoneDetector
from .core.identity_manager import IdentityManager
from .core.phone_usage_logic import PhoneUsageSpatialLogic
from .core.phone_usage_filter import PhoneUsageFilter
from .core.state_manager import PersonStateManager
from .core.face_adapter import crop_person_roi

# Configuration
from .config.manager import ConfigurationManager
from .config.models import CameraConfig, PersonTrackingAppConfig

# Logging
from .logging.csv_logger import CSVLogger

# Import from face_recognition module
from face_recognition.core.detector import FaceDetector
from face_recognition.core.recognizer import FaceRecognition
from face_recognition.storage.pgvector_store import PgVectorStore


class FaceRecognitionArgs:
    """Arguments object for FaceRecognition class compatibility."""

    def __init__(
        self,
        client_slug: str,
        use_pgvector: bool = True,
        match_threshold: float = 0.3,
        db_path: Optional[str] = None
    ):
        self.client_slug = client_slug
        self.use_pgvector = use_pgvector
        self.match_threshold = match_threshold
        self.db_path = db_path
        self.logger = logger


class PersonTrackingEngine:
    """Main orchestrator for the person tracking pipeline.

    This class coordinates all components to process video frames and:
    - Detect and track persons
    - Recognize faces and lock identities
    - Detect phone usage with temporal filtering
    - Emit events for state changes
    - Log results to CSV and API
    """

    def __init__(
        self,
        config: PersonTrackingAppConfig,
        camera_config: CameraConfig
    ):
        """Initialize the PersonTrackingEngine.

        Args:
            config: Global application configuration
            camera_config: Configuration for this specific camera
        """
        self.config = config
        self.camera_config = camera_config
        self.camera_id = camera_config.camera_id
        self.client_slug = config.client_slug

        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()

        # Initialize all components
        self._init_components()

        logger.info(
            f"PersonTrackingEngine initialized for camera {self.camera_id} "
            f"({camera_config.camera_name})"
        )

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        pc = self.camera_config.person_detection
        tc = self.camera_config.person_tracking
        fc = self.camera_config.face_recognition
        phd = self.camera_config.phone_detection
        phu = self.camera_config.phone_usage

        # Person Detection
        self.person_detector = PersonDetector(
            model_size=pc.model_size,
            confidence_threshold=pc.confidence_threshold,
            iou_threshold=pc.iou_threshold
        )
        logger.info(f"Initialized PersonDetector (YOLOv8{pc.model_size}-pose)")

        # Person Tracking
        self.person_tracker = PersonTracker(
            tracker_type=tc.tracker_type,
            max_age=tc.max_track_age,
            min_hits=tc.min_track_hits,
            iou_threshold=tc.iou_threshold
        )
        logger.info(f"Initialized PersonTracker ({tc.tracker_type})")

        # Track Manager
        self.track_manager = PersonTrackManager(
            max_history_frames=300
        )
        logger.info("Initialized PersonTrackManager")

        # Face Detection and Recognition
        self.face_detector = FaceDetector(
            gpu_id=0,
            model_name='buffalo_l'
        )
        logger.info("Initialized FaceDetector (InsightFace buffalo_l)")

        # Create args object for FaceRecognition compatibility
        face_args = FaceRecognitionArgs(
            client_slug=self.client_slug,
            use_pgvector=self.config.database.use_pgvector,
            match_threshold=fc.match_threshold
        )

        self.face_recognizer = FaceRecognition(face_args)
        logger.info(f"Initialized FaceRecognition (pgvector: {self.config.database.use_pgvector})")

        # Phone Detection
        self.phone_detector = PhoneDetector(
            model_size=phd.model_size,
            confidence_threshold=phd.confidence_threshold
        )
        logger.info(f"Initialized PhoneDetector (YOLOv8{phd.model_size})")

        # Identity Manager
        self.identity_manager = IdentityManager(
            identity_lock_frames=fc.identity_lock_frames,
            identity_consensus=fc.identity_consensus,
            min_window_duration_ms=fc.min_window_duration_ms,
            similarity_threshold=fc.match_threshold
        )
        logger.info(f"Initialized IdentityManager (M={fc.identity_lock_frames}, consensus={fc.identity_consensus})")

        # Phone Usage Spatial Logic
        self.phone_usage_logic = PhoneUsageSpatialLogic(
            hand_distance_threshold=phu.hand_distance_threshold,
            head_distance_threshold=phu.head_distance_threshold,
            upper_body_zone_margin=phu.upper_body_zone_margin,
            required_checks=phu.required_checks
        )
        logger.info("Initialized PhoneUsageSpatialLogic")

        # Phone Usage Filter
        self.phone_usage_filter = PhoneUsageFilter(
            confirmation_frames=phu.confirmation_frames,
            confirmation_consensus=phu.confirmation_consensus,
            min_duration_ms=phu.min_duration_ms
        )
        logger.info(f"Initialized PhoneUsageFilter (N={phu.confirmation_frames}, consensus={phu.confirmation_consensus})")

        # State Manager
        self.state_manager = PersonStateManager(
            camera_id=self.camera_id
        )
        logger.info("Initialized PersonStateManager")

        # CSV Logger
        output_dir = self.camera_config.storage.output_dir
        self.csv_logger = CSVLogger(
            output_dir=output_dir,
            client_slug=self.client_slug,
            camera_id=self.camera_id
        )
        logger.info(f"Initialized CSVLogger (output: {output_dir})")

        # Output directory for frames
        self.output_dir = Path(output_dir) / self.client_slug / f"camera_{self.camera_id}"
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single video frame through the entire pipeline.

        Args:
            frame: Input video frame (BGR format)
            frame_num: Current frame number

        Returns:
            Tuple of (annotated_frame, events):
                - annotated_frame: Frame with visualizations
                - events: List of events generated this frame
        """
        start_time = time.time()

        # Step 1: Detect persons with pose keypoints
        detections = self.person_detector.detect_persons(frame)

        # Step 2: Update tracker with new detections
        active_tracks, removed_tracks = self.person_tracker.update(detections, frame)

        # Step 3: Process each active track
        for track in active_tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            keypoints = track.get('keypoints')
            confidence = track.get('confidence', 0.0)

            # Store track data
            self.track_manager.add_track_detection(
                track_id=track_id,
                frame_num=frame_num,
                bbox=bbox,
                keypoints=keypoints,
                confidence=confidence
            )

            # Step 3a: Face recognition
            identity = None
            identity_locked = False
            identity_confidence = 0.0

            if not self.identity_manager.is_identity_locked(track_id):
                recognition_result = self._recognize_face_in_person(frame, bbox)

                if recognition_result['face_detected']:
                    # Update identity manager with vote
                    self.identity_manager.update_identity(track_id, recognition_result)

            # Get current identity status
            if self.identity_manager.is_identity_locked(track_id):
                locked_identity = self.identity_manager.get_locked_identity(track_id)
                identity = locked_identity['name']
                identity_locked = True
                identity_confidence = locked_identity['confidence']
            else:
                # Check voting status for tentative identity
                voting_status = self.identity_manager.get_voting_status(track_id)
                if voting_status and voting_status.get('top_candidate'):
                    identity = voting_status['top_candidate']
                    identity_confidence = voting_status.get('top_avg_similarity', 0.0)

            # Update track manager with identity
            if identity:
                self.track_manager.set_track_identity(track_id, identity)

            # Step 3b: Phone detection and usage analysis
            using_phone = False
            phone_confidence = 0.0

            # Detect phones in frame
            phones = self.phone_detector.detect_phones(frame)

            # Associate phones with this person
            if phones and keypoints is not None:
                # Calculate person height for pixel-to-meter conversion
                person_height = self.person_detector.calculate_person_height(keypoints)

                for phone in phones:
                    phone_bbox = phone['bbox']

                    # Calculate spatial score
                    spatial_result = self.phone_usage_logic.calculate_phone_usage_score(
                        person_keypoints=keypoints,
                        phone_bbox=phone_bbox,
                        person_height_pixels=person_height
                    )

                    if spatial_result['using_phone']:
                        # Update temporal filter
                        is_using = self.phone_usage_filter.update(track_id, spatial_result)

                        if is_using:
                            using_phone = True
                            phone_confidence = self.phone_usage_filter.get_usage_confidence(track_id)
                            break

            # If no phone detected, still update filter with negative result
            if not using_phone:
                no_phone_result = {
                    'using_phone': False,
                    'confidence': 0.0,
                    'checks_passed': 0,
                    'details': {'zone_check': False, 'hand_check': False, 'head_check': False}
                }
                self.phone_usage_filter.update(track_id, no_phone_result)
                # Check if filter still shows phone usage (temporal smoothing)
                using_phone = self.phone_usage_filter.is_using_phone(track_id)
                if using_phone:
                    phone_confidence = self.phone_usage_filter.get_usage_confidence(track_id)

            # Update track manager with phone usage
            self.track_manager.set_track_phone_usage(track_id, using_phone)

            # Step 4: Update state manager
            self.state_manager.update_person(
                track_id=track_id,
                identity=identity,
                identity_locked=identity_locked,
                identity_confidence=identity_confidence,
                using_phone=using_phone,
                phone_confidence=phone_confidence
            )

        # Step 5: Handle removed tracks
        for track in removed_tracks:
            track_id = track['track_id']

            # Get final track data
            track_data = self.track_manager.remove_track(track_id)

            # Log summary
            if track_data:
                self.csv_logger.log_person_summary(
                    track_id=track_id,
                    person_name=track_data.get('identity'),
                    first_seen=track_data.get('first_seen'),
                    last_seen=track_data.get('last_seen'),
                    total_frames=track_data.get('total_frames', 0),
                    total_phone_usage_seconds=track_data.get('phone_usage_duration', 0.0)
                )

            # Remove from state manager (emits person_exited event)
            self.state_manager.remove_person(track_id)

            # Clean up other managers
            self.identity_manager.reset_track(track_id)
            self.phone_usage_filter.reset_track(track_id)

        # Step 6: Process and log events
        events = self.state_manager.get_events(clear=True)

        for event in events:
            # Log to CSV
            self.csv_logger.log_event(
                track_id=event.track_id,
                event_type=event.event_type.value,
                person_name=event.identity,
                using_phone=event.event_type.value in ['phone_usage_started', 'phone_usage_stopped'],
                confidence=event.confidence or 0.0,
                duration=event.duration,
                timestamp=event.timestamp
            )

            # Log to console
            logger.info(
                f"Event: {event.event_type.value} | "
                f"Track: {event.track_id} | "
                f"Identity: {event.identity or 'Unknown'} | "
                f"Confidence: {event.confidence or 0:.2f}"
            )

        # Step 7: Annotate frame
        annotated_frame = self._annotate_frame(frame, active_tracks, phones if 'phones' in dir() else [])

        # Update performance metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.frame_count += 1

        return annotated_frame, [e.to_dict() for e in events]

    def _recognize_face_in_person(
        self,
        frame: np.ndarray,
        person_bbox: np.ndarray
    ) -> Dict[str, Any]:
        """Recognize face within a person's bounding box.

        Args:
            frame: Full video frame
            person_bbox: Person's bounding box [x1, y1, x2, y2]

        Returns:
            Recognition result dict with keys:
                - face_detected: bool
                - recognized: bool
                - name: str or None
                - similarity: float
                - embedding: np.ndarray or None
        """
        # Crop person ROI
        roi, offset = crop_person_roi(frame, person_bbox, expand=0.1)

        if roi is None or roi.size == 0:
            return {
                'face_detected': False,
                'recognized': False,
                'name': None,
                'similarity': 0.0,
                'embedding': None
            }

        # Detect faces in ROI
        face_features = self.face_detector.extract_face_features(roi)

        if not face_features:
            return {
                'face_detected': False,
                'recognized': False,
                'name': None,
                'similarity': 0.0,
                'embedding': None
            }

        # Use the first detected face
        face = face_features[0]
        embedding = face['embedding']
        landmarks = face.get('landmarks')

        # Search for match in database
        if self.config.database.use_pgvector:
            # Use pgvector store directly for faster search
            pgvector_store = PgVectorStore(self.client_slug)
            matches = pgvector_store.search_similar(
                query_embedding=embedding,
                limit=1,
                threshold=self.camera_config.face_recognition.match_threshold
            )

            if matches:
                best_match = matches[0]
                return {
                    'face_detected': True,
                    'recognized': True,
                    'name': best_match['user_name'],
                    'similarity': best_match['similarity'],
                    'embedding': embedding
                }
            else:
                return {
                    'face_detected': True,
                    'recognized': False,
                    'name': None,
                    'similarity': 0.0,
                    'embedding': embedding
                }
        else:
            # Use FaceRecognition class
            result = self.face_recognizer.recognize_face(
                face_embs=np.array([embedding]),
                frame_nums=[0],
                landmarks_dict={0: landmarks} if landmarks is not None else None
            )

            return {
                'face_detected': True,
                'recognized': result['recognized'] == 'recognized',
                'name': result['name'] if result['recognized'] == 'recognized' else None,
                'similarity': result['similarity'],
                'embedding': embedding
            }

    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        phones: List[Dict]
    ) -> np.ndarray:
        """Annotate frame with detection results.

        Args:
            frame: Input frame
            tracks: List of active tracks
            phones: List of detected phones

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw each tracked person
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            keypoints = track.get('keypoints')

            # Get person state
            state = self.state_manager.get_state(track_id)

            if state:
                identity = state.identity
                identity_locked = state.identity_locked
                using_phone = state.using_phone
            else:
                identity = None
                identity_locked = False
                using_phone = False

            # Determine box color based on state
            if identity_locked:
                color = (0, 255, 0)  # Green for locked identity
            elif identity:
                color = (0, 255, 255)  # Yellow for tentative identity
            else:
                color = (0, 0, 255)  # Red for unknown

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_parts = [f"ID:{track_id}"]
            if identity:
                label_parts.append(identity)
            if using_phone:
                label_parts.append("PHONE")

            label = " | ".join(label_parts)

            # Background for label
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            # Draw keypoints skeleton if available
            if keypoints is not None:
                self._draw_skeleton(annotated, keypoints)

        # Draw phones
        for phone in phones:
            phone_bbox = phone['bbox']
            x1, y1, x2, y2 = map(int, phone_bbox[:4])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                annotated,
                "Phone",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        # Draw FPS and stats
        fps = self.get_current_fps()
        stats_text = f"FPS: {fps:.1f} | Tracks: {len(tracks)} | Phones: {len(phones)}"
        cv2.putText(
            annotated,
            stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return annotated

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> None:
        """Draw pose skeleton on frame.

        Args:
            frame: Frame to draw on
            keypoints: Keypoints array (17, 3)
        """
        # COCO skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        # Draw keypoints
        for i, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        # Draw skeleton
        for start_idx, end_idx in skeleton:
            if (keypoints[start_idx][2] > 0.3 and
                keypoints[end_idx][2] > 0.3):
                start = tuple(map(int, keypoints[start_idx][:2]))
                end = tuple(map(int, keypoints[end_idx][:2]))
                cv2.line(frame, start, end, (0, 255, 255), 1)

    def save_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        prefix: str = "frame"
    ) -> str:
        """Save annotated frame to disk.

        Args:
            frame: Frame to save
            frame_num: Frame number
            prefix: Filename prefix

        Returns:
            Path to saved file
        """
        filename = f"{prefix}_{frame_num:06d}.jpg"
        filepath = self.frames_dir / filename
        cv2.imwrite(str(filepath), frame)
        return str(filepath)

    def get_current_fps(self) -> float:
        """Get current processing FPS.

        Returns:
            Frames per second
        """
        if self.frame_count == 0:
            return 0.0

        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0

    def get_average_latency(self) -> float:
        """Get average processing latency per frame.

        Returns:
            Average latency in milliseconds
        """
        if self.frame_count == 0:
            return 0.0

        return (self.total_processing_time / self.frame_count) * 1000

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'camera_id': self.camera_id,
            'frame_count': self.frame_count,
            'current_fps': self.get_current_fps(),
            'average_latency_ms': self.get_average_latency(),
            'total_runtime_seconds': time.time() - self.start_time,
            'tracker_stats': self.person_tracker.get_statistics(),
            'track_manager_stats': self.track_manager.get_statistics(),
            'identity_manager_stats': self.identity_manager.get_statistics(),
            'phone_filter_stats': self.phone_usage_filter.get_statistics(),
            'state_manager_stats': self.state_manager.get_statistics()
        }

    def reload_embeddings(self) -> None:
        """Reload face embeddings from database.

        Call this when embeddings are updated externally.
        """
        self.face_recognizer.reload_embeddings()
        logger.info(f"Reloaded face embeddings for camera {self.camera_id}")

    def reset(self) -> None:
        """Reset all tracking state.

        This clears all tracks and starts fresh.
        """
        self.person_tracker.reset()
        self.track_manager.reset()
        self.identity_manager.reset()
        self.phone_usage_filter.reset()
        self.state_manager.reset()

        self.frame_count = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()

        logger.info(f"Reset PersonTrackingEngine for camera {self.camera_id}")

    def cleanup(self) -> None:
        """Clean up resources.

        Call this when shutting down the engine.
        """
        # Rotate CSV log files
        self.csv_logger.rotate_log_file()

        # Log final statistics
        stats = self.get_statistics()
        logger.info(
            f"PersonTrackingEngine shutdown | "
            f"Camera: {self.camera_id} | "
            f"Frames: {stats['frame_count']} | "
            f"FPS: {stats['current_fps']:.1f} | "
            f"Avg Latency: {stats['average_latency_ms']:.1f}ms"
        )
