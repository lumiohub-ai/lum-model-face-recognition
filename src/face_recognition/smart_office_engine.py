"""SmartOfficeEngine - Unified person tracking and face recognition system.

This module provides a unified engine that combines:
- Person detection and tracking (YOLOv8-Pose + BoT-SORT)
- Face recognition within person ROIs (InsightFace)
- Phone usage detection (YOLOv8n)
- Attendance logging (IN/OUT status)

Replaces HBFace with improved architecture:
- Track persons instead of faces for stable IDs
- Temporal voting for identity locking
- Configurable phone detection per camera
"""

import os
import sys
import time
import signal
import yaml
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# Face recognition components
from .api.client import APIClient
from .core.detector import FaceDetector
from .core.recognizer import FaceRecognition
from .video.stream_handler import StreamHandler
from .logging.entry_logger import EntryLogger
from .dashboard.visualizer import Visualization
from .dashboard.camera_processor import get_camera_processor, setup_cameras as init_camera_processor

# Person tracking components
from person_tracking.core.person_detector import PersonDetector
from person_tracking.core.person_tracker import PersonTracker
from person_tracking.core.track_manager import PersonTrackManager
from person_tracking.core.phone_detector import PhoneDetector
from person_tracking.core.identity_manager import IdentityManager
from person_tracking.core.phone_usage_logic import PhoneUsageSpatialLogic
from person_tracking.core.phone_usage_filter import PhoneUsageFilter
from person_tracking.core.state_manager import PersonStateManager, EventType
from person_tracking.core.face_adapter import crop_person_roi
from person_tracking.video.frame_annotator import FrameAnnotator
from person_tracking.logging.csv_logger import CSVLogger


class CameraEngine:
    """Engine for processing a single camera stream.

    Each camera has its own tracking and state management components.
    Detection models are shared across cameras for GPU efficiency.
    """

    def __init__(
        self,
        camera_config: Dict[str, Any],
        face_detector: FaceDetector,
        face_recognizer: FaceRecognition,
        person_detector: 'PersonDetector',
        phone_detector: Optional['PhoneDetector'],
        client_slug: str
    ):
        """Initialize camera engine.

        Args:
            camera_config: Camera configuration from API
            face_detector: Shared face detector instance
            face_recognizer: Shared face recognizer instance
            person_detector: Shared person detector instance
            phone_detector: Shared phone detector instance (or None)
            client_slug: Client organization slug
        """
        self.camera_id = camera_config['camera_id']
        self.camera_name = camera_config['camera_name']
        self.cam_type = camera_config['cam_type']  # IN or OUT
        self.stream_url = camera_config['stream_url']
        self.application = camera_config.get('application', 'FaceRecognision')
        self.match_threshold = camera_config.get('match_threshold', 0.3)
        self.min_face_size = camera_config.get('min_face_size', 150)  # Minimum face size for quality check
        self.roi = camera_config.get('roi')
        self.line_points = camera_config.get('line_points')

        # Feature flags based on application
        self.enable_attendance = True  # Always enabled
        self.enable_phone_detection = self.application == 'PhoneUsageDetection'

        # Shared components (models)
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.phone_detector = phone_detector if self.enable_phone_detection else None
        self.client_slug = client_slug

        # Initialize per-camera components (tracking, state)
        self._init_components()

        logger.info(
            f"CameraEngine initialized: {self.camera_name} (ID: {self.camera_id}) | "
            f"Type: {self.cam_type} | "
            f"Phone Detection: {'enabled' if self.enable_phone_detection else 'disabled'}"
        )

    def _init_components(self) -> None:
        """Initialize tracking and state management components.

        Note: Detection models (person_detector, phone_detector) are shared
        and passed in __init__, not created here.
        """
        # Person Tracking (per-camera for isolated state)
        self.person_tracker = PersonTracker(
            tracker_type='botsort',
            max_age=60,
            min_hits=3,
            iou_threshold=0.5  # Increased from default 0.3 for more stable tracking
        )

        # Track Manager
        self.track_manager = PersonTrackManager(max_history_frames=100)

        # Identity Manager (temporal voting)
        self.identity_manager = IdentityManager(
            identity_lock_frames=5,
            identity_consensus=0.60,
            min_window_duration_ms=333,
            similarity_threshold=self.match_threshold
        )

        # Phone usage components (if enabled)
        if self.enable_phone_detection:
            self.phone_usage_logic = PhoneUsageSpatialLogic()
            self.phone_usage_filter = PhoneUsageFilter()
        else:
            self.phone_usage_logic = None
            self.phone_usage_filter = None

        # State Manager
        self.state_manager = PersonStateManager(camera_id=self.camera_id)

        # Frame counter
        self.frame_count = 0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> Tuple[List[Dict], List[Dict], np.ndarray, List[Dict]]:
        """Process a single frame.

        Args:
            frame: Input video frame
            frame_num: Frame number

        Returns:
            Tuple of (recognized_persons, phone_events, processed_frame, detected_phones)
        """
        # Apply ROI if configured
        if self.roi:
            x1, y1, x2, y2 = self.roi
            frame = frame[y1:y2, x1:x2]

        self.frame_count += 1
        recognized_persons = []
        phone_events = []

        # Step 1: Detect persons
        detections = self.person_detector.detect_persons(frame)

        # Step 2: Update tracker
        active_tracks, removed_tracks = self.person_tracker.update(detections, frame)

        # Step 3: Detect phones ONCE per frame (not per track)
        detected_phones = []
        if self.enable_phone_detection and self.phone_detector:
            detected_phones = self.phone_detector.detect_phones(frame)

        # Step 4: Process each active track
        for track in active_tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            keypoints = track.get('keypoints')
            confidence = track.get('confidence', 0.0)

            # Check if this is a new track (person entered frame)
            prev_state = self.state_manager.get_state(track_id)
            if prev_state is None:
                # Color based on camera type: Green for IN, Red for OUT
                if self.cam_type == "IN":
                    logger.opt(colors=True).info(f"<green>→ Track {track_id} ENTERED frame [IN]</green>")
                else:
                    logger.opt(colors=True).info(f"<red>→ Track {track_id} ENTERED frame [OUT]</red>")

            # Store track data
            self.track_manager.add_track_detection(
                track_id=track_id,
                frame_num=frame_num,
                bbox=bbox,
                keypoints=keypoints,
                confidence=confidence
            )

            # Face recognition with early re-identification
            identity = None
            identity_locked = False
            identity_confidence = 0.0
            face_image = None

            if not self.identity_manager.is_identity_locked(track_id):
                result = self._recognize_face(frame, bbox)

                if result['face_detected']:
                    # EARLY RE-ID: If this is a new track and face is recognized,
                    # check if this person already has an active track
                    if prev_state is None and result.get('recognized'):
                        recognized_name = result.get('name')
                        existing_track = self._find_track_with_identity(recognized_name, exclude_track_id=track_id)

                        if existing_track is not None:
                            # RE-IDENTIFICATION: This person already has an active track!
                            # Merge this new track into the existing one
                            logger.warning(
                                f"🔄 RE-ID: Track {track_id} merged into Track {existing_track} "
                                f"(returning person: {recognized_name})"
                            )
                            self._merge_tracks(source_track_id=track_id, target_track_id=existing_track)
                            continue

                    # Normal identity voting
                    self.identity_manager.update_identity(track_id, result)
                    face_image = result.get('face_image')

            # Get identity status (reuse prev_state from above)

            if self.identity_manager.is_identity_locked(track_id):
                locked = self.identity_manager.get_locked_identity(track_id)
                identity = locked['name']
                identity_locked = True
                identity_confidence = locked['confidence']

                # Check if this is the FIRST time identity is locked
                if prev_state and not prev_state.identity_locked:
                    # Check for duplicate tracks (same person with different track_id)
                    existing_track = self._find_track_with_identity(identity, exclude_track_id=track_id)

                    if existing_track is not None:
                        # DUPLICATE DETECTED: Merge this track into existing one
                        logger.warning(f"🔗 Track {track_id} merged into Track {existing_track} (same person: {identity})")
                        self._merge_tracks(source_track_id=track_id, target_track_id=existing_track)
                        # Skip further processing for this track (it's been merged)
                        continue

                    # Identity just locked - emit recognition event for immediate logging
                    logger.opt(colors=True).info(f"<blue>🔒 Track {track_id} recognized as '{identity}' [{self.cam_type}]</blue>")

                    # Get best quality person image as proof (full frame with bbox)
                    proof_image = self._get_best_person_image(track_id)

                    recognized_persons.append({
                        'track_id': track_id,
                        'name': identity,
                        'recognized': True,
                        'confidence': identity_confidence,
                        'appear_time': prev_state.first_seen,
                        'camera_name': self.camera_name,
                        'camera_id': self.camera_id,
                        'status': self.cam_type,  # IN or OUT
                        'face_image': face_image,
                        'proof_image': proof_image,  # Best quality person crop
                    })
            else:
                voting = self.identity_manager.get_voting_status(track_id)
                if voting and voting.get('top_candidate'):
                    identity = voting['top_candidate']
                    identity_confidence = voting.get('top_avg_similarity', 0.0)

            # Phone usage detection (using pre-detected phones)
            using_phone = False
            phone_confidence = 0.0

            if self.enable_phone_detection and keypoints is not None:
                # Use pre-detected phones (detected once per frame above)
                for phone in detected_phones:
                    person_height = self.person_detector.calculate_person_height(keypoints)
                    spatial_result = self.phone_usage_logic.calculate_phone_usage_score(
                        keypoints, phone['bbox'], person_height
                    )

                    if spatial_result['using_phone']:
                        is_using = self.phone_usage_filter.update(track_id, spatial_result)
                        if is_using:
                            using_phone = True
                            phone_confidence = self.phone_usage_filter.get_usage_confidence(track_id)
                            # Log phone usage event (only once when confirmed)
                            if prev_state and not prev_state.using_phone:
                                identity_str = f"{identity}" if identity else f"Track {track_id}"
                                logger.opt(colors=True).info(f"<yellow>📱 {identity_str} using phone [{self.cam_type}]</yellow>")
                            break

                if not using_phone:
                    no_phone = {'using_phone': False, 'confidence': 0.0, 'checks_passed': 0}
                    self.phone_usage_filter.update(track_id, no_phone)
                    using_phone = self.phone_usage_filter.is_using_phone(track_id)
                    if using_phone:
                        phone_confidence = self.phone_usage_filter.get_usage_confidence(track_id)

            # Update state manager
            self.state_manager.update_person(
                track_id=track_id,
                identity=identity,
                identity_locked=identity_locked,
                identity_confidence=identity_confidence,
                using_phone=using_phone,
                phone_confidence=phone_confidence
            )

            # Store face image and person bbox for later use (unrecognized faces)
            if face_image is not None:
                self.track_manager.track_crop_history.setdefault(track_id, {})[frame_num] = {
                    'face': face_image,
                    'bbox': bbox,
                    'frame': frame.copy()
                }

        # Step 5: Process removed tracks (person left frame)
        for track in removed_tracks:
            track_id = track['track_id']

            # Get state before cleanup
            state = self.state_manager.get_state(track_id)

            # Log person exit
            if state:
                identity_str = f"'{state.identity}'" if state.identity_locked else f"Track {track_id}"
                # Color based on camera type: Green for IN, Red/Magenta for OUT
                if self.cam_type == "IN":
                    logger.opt(colors=True).info(f"<green>← {identity_str} LEFT frame [IN]</green>")
                else:
                    logger.opt(colors=True).info(f"<magenta>← {identity_str} LEFT frame [OUT]</magenta>")

            # If person was NOT recognized (identity never locked), send unrecognized person image
            if state and not state.identity_locked:
                logger.info(f"📸 Sending unrecognized person image for Track {track_id}")
                person_image = self._get_best_person_image(track_id)
                if person_image is not None and person_image.size > 0:
                    # Add to recognized_persons list for API submission
                    recognized_persons.append({
                        'track_id': track_id,
                        'name': None,
                        'recognized': False,
                        'confidence': 0.0,
                        'appear_time': state.first_seen,
                        'camera_name': self.camera_name,
                        'camera_id': self.camera_id,
                        'status': self.cam_type,
                        'face_image': person_image,  # Actually person image, but API expects this key
                    })

            # Remove track data
            self.track_manager.remove_track(track_id)

            # Clean up state managers
            self.state_manager.remove_person(track_id)
            self.identity_manager.reset_track(track_id)
            if self.enable_phone_detection:
                self.phone_usage_filter.reset_track(track_id)

        # Step 6: Get phone usage events
        events = self.state_manager.get_events(clear=True)
        for event in events:
            if event.event_type in [EventType.PHONE_USAGE_STARTED, EventType.PHONE_USAGE_STOPPED]:
                phone_events.append({
                    'event_type': event.event_type.value,
                    'track_id': event.track_id,
                    'identity': event.identity,
                    'camera_id': self.camera_id,
                    'camera_name': self.camera_name,
                    'timestamp': event.timestamp,
                    'duration': event.duration
                })

        return recognized_persons, phone_events, frame, detected_phones

    def _recognize_face(self, frame: np.ndarray, bbox: np.ndarray) -> Dict:
        """Recognize face within person bounding box."""
        roi, offset = crop_person_roi(frame, bbox, expand=0.1)

        if roi is None or roi.size == 0:
            return {'face_detected': False, 'name': None, 'similarity': 0.0}

        # Detect face
        faces = self.face_detector.detect(roi)

        if not faces:
            return {'face_detected': False, 'name': None, 'similarity': 0.0}

        face = faces[0]
        embedding = face.embedding / np.linalg.norm(face.embedding)

        # Get face crop for unrecognized faces
        face_bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = face_bbox
        face_image = roi[max(0, y1):y2, max(0, x1):x2]

        # Match against database
        if len(self.face_recognizer.db_embs) == 0:
            return {
                'face_detected': True,
                'recognized': False,
                'name': None,
                'similarity': 0.0,
                'embedding': embedding,
                'face_image': face_image
            }

        similarities = self.face_recognizer.compute_similarities(np.array([embedding]))
        best_idx, best_similarity = self.face_recognizer.get_best_match(similarities)

        if best_similarity >= self.match_threshold:
            name = self.face_recognizer.db_names[best_idx]
            return {
                'face_detected': True,
                'recognized': True,
                'name': name,
                'similarity': best_similarity,
                'embedding': embedding,
                'face_image': face_image
            }
        else:
            return {
                'face_detected': True,
                'recognized': False,
                'name': None,
                'similarity': best_similarity,
                'embedding': embedding,
                'face_image': face_image
            }

    def _get_best_person_image(self, track_id: int) -> Optional[np.ndarray]:
        """Get the best quality person image from track history.

        Uses face quality criteria to select best frame, but returns full person crop.

        Selection criteria (based on face):
        1. Face size >= min_face_size
        2. Not blurry (Laplacian variance check)
        3. Most frontal (largest face area = closest to camera)

        Returns:
            Best quality person crop from bbox, or None if no suitable frames found
        """
        crops = self.track_manager.track_crop_history.get(track_id, {})
        if not crops:
            return None

        min_face_size = getattr(self, 'min_face_size', 150)
        blur_threshold = 100.0  # Laplacian variance threshold

        best_frame_num = None
        best_score = -1

        for frame_num, crop_data in crops.items():
            # Handle both old format (direct face crop) and new format (dict)
            if isinstance(crop_data, dict):
                face_crop = crop_data.get('face')
            else:
                face_crop = crop_data  # Backward compatibility

            if face_crop is None or face_crop.size == 0:
                continue

            # Check 1: Face size (height and width must be >= min_face_size)
            h, w = face_crop.shape[:2]
            if h < min_face_size or w < min_face_size:
                continue

            # Check 2: Blur detection using Laplacian variance
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian_var < blur_threshold:
                continue  # Too blurry

            # Score: Combine face size (frontal indicator) and sharpness
            # Larger face = more frontal, higher Laplacian = sharper
            size_score = (h * w) / (min_face_size ** 2)  # Normalized by min size
            sharpness_score = laplacian_var / blur_threshold

            total_score = size_score * 0.6 + sharpness_score * 0.4

            if total_score > best_score:
                best_score = total_score
                best_frame_num = frame_num

        # If we found a best frame, extract person crop from that frame
        if best_frame_num is not None:
            crop_data = crops[best_frame_num]
            if isinstance(crop_data, dict):
                frame = crop_data.get('frame')
                bbox = crop_data.get('bbox')

                if frame is not None and bbox is not None:
                    # Crop person from frame using bbox
                    x1, y1, x2, y2 = map(int, bbox)
                    person_crop = frame[y1:y2, x1:x2]
                    return person_crop
                else:
                    # Fallback to face crop if frame/bbox not available
                    return crop_data.get('face')
            else:
                # Old format - return face crop
                return crop_data

        return None

    def _find_track_with_identity(self, identity_name: str, exclude_track_id: Optional[int] = None) -> Optional[int]:
        """Find an existing track with the given identity.

        Args:
            identity_name: Name of the identity to search for
            exclude_track_id: Track ID to exclude from search

        Returns:
            Track ID with matching identity, or None if not found
        """
        all_states = self.state_manager.get_all_states()

        for state in all_states:
            # Skip the excluded track
            if exclude_track_id is not None and state.track_id == exclude_track_id:
                continue

            # Check if this track has the same locked identity
            if state.identity_locked and state.identity == identity_name:
                # Verify track is still active (not just in state history)
                track_info = self.person_tracker.get_track_info(state.track_id)
                if track_info is not None:
                    return state.track_id

        return None

    def _merge_tracks(self, source_track_id: int, target_track_id: int) -> None:
        """Merge source track into target track.

        Transfers all data from source track to target track and removes source.

        Args:
            source_track_id: Track to merge (will be deleted)
            target_track_id: Target track to merge into (will be kept)
        """
        # Transfer track data from source to target
        source_data = self.track_manager.get_track_data(source_track_id)
        if source_data:
            target_data = self.track_manager.get_track_data(target_track_id)
            if target_data:
                # Merge detection history
                if 'detections' in source_data and 'detections' in target_data:
                    target_data['detections'].extend(source_data['detections'])

                # Merge keypoints history (handle both dict and list formats)
                if 'keypoints' in source_data and 'keypoints' in target_data:
                    source_kp = source_data['keypoints']
                    target_kp = target_data['keypoints']
                    # If both are dicts, merge them
                    if isinstance(source_kp, dict) and isinstance(target_kp, dict):
                        target_kp.update(source_kp)
                    # If both are lists, extend
                    elif isinstance(source_kp, list) and isinstance(target_kp, list):
                        target_kp.extend(source_kp)

        # Transfer crop history
        if source_track_id in self.track_manager.track_crop_history:
            source_crops = self.track_manager.track_crop_history[source_track_id]
            if target_track_id not in self.track_manager.track_crop_history:
                self.track_manager.track_crop_history[target_track_id] = {}
            self.track_manager.track_crop_history[target_track_id].update(source_crops)

        # Remove source track from all managers
        self.track_manager.remove_track(source_track_id)
        self.state_manager.remove_person(source_track_id)
        self.identity_manager.reset_track(source_track_id)
        if self.enable_phone_detection:
            self.phone_usage_filter.reset_track(source_track_id)

        # Remove from person_tracker's active_tracks
        if source_track_id in self.person_tracker.active_tracks:
            del self.person_tracker.active_tracks[source_track_id]

    def reset(self) -> None:
        """Reset all tracking state."""
        self.person_tracker.reset()
        self.track_manager.reset()
        self.identity_manager.reset()
        if self.enable_phone_detection:
            self.phone_usage_filter.reset()
        self.state_manager.reset()
        self.frame_count = 0


class SmartOfficeEngine:
    """Unified engine for Smart Office person tracking and face recognition.

    This class replaces HBFace with an improved architecture that:
    - Tracks persons instead of faces for stable IDs
    - Uses temporal voting for identity locking
    - Supports phone usage detection
    - Maintains backward compatibility with EntryLogger and API
    """

    def __init__(
        self,
        email: str,
        password: str,
        client_slug: str,
        api_host: str,
        applications: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize SmartOfficeEngine.

        Args:
            email: API authentication email
            password: API authentication password
            client_slug: Organization slug
            api_host: API base URL
            applications: List of application types to fetch
                         Default: ['FaceRecognision', 'PhoneUsageDetection']
            **kwargs: Additional configuration
        """
        self.client_slug = client_slug
        self.api_host = api_host
        self.applications = applications or ['FaceRecognision', 'PhoneUsageDetection']

        # Store credentials for entry logger
        self._email = email
        self._password = password

        # Initialize API client
        self.api_client = APIClient(
            api_host=api_host,
            email=email,
            password=password,
            client_slug=client_slug
        )

        # Fetch camera configurations
        self.camera_configs = self._fetch_camera_configs()

        if not self.camera_configs:
            raise ValueError(f"No cameras found for applications: {self.applications}")

        # Initialize shared detection models (GPU efficiency)
        logger.info("Initializing shared detection models...")
        self.face_detector = FaceDetector(gpu_id=0, model_name='buffalo_l')
        self.face_recognizer = self._init_face_recognizer()

        # Shared person detector (YOLOv8-Pose)
        self.shared_person_detector = PersonDetector(
            model_size='s',
            confidence_threshold=0.5
        )

        # Shared phone detector (YOLOv8n) - only if any camera needs it
        needs_phone_detection = any(
            c.get('application') == 'PhoneUsageDetection'
            for c in self.camera_configs
        )

        # Load phone detection config from config file
        phone_config = self._load_phone_detection_config()

        self.shared_phone_detector = PhoneDetector(
            model_size=phone_config.get('model_size', 'n'),
            confidence_threshold=phone_config.get('confidence_threshold', 0.4)
        ) if needs_phone_detection else None

        if self.shared_phone_detector:
            logger.info("Phone detection enabled for PhoneUsageDetection cameras")

        # Initialize streams and engines
        self.streams: List[StreamHandler] = []
        self.camera_engines: List[CameraEngine] = []
        self._init_cameras()

        # Initialize entry logger
        self.entry_logger = self._init_entry_logger(email, password)

        # Visualization
        self.visualize = Visualization()
        self.frame_annotator = FrameAnnotator()

        # CSV logger for phone events
        self.csv_logger = CSVLogger(
            output_dir=kwargs.get('output_dir', 'volumes/storage/person-tracking'),
            client_slug=client_slug,
            camera_id=0  # Will be updated per event
        )

        # Initialize camera processor for dashboard streaming
        try:
            init_camera_processor()
            self.camera_processor = get_camera_processor()
            logger.info("Camera processor initialized for dashboard streaming")
        except Exception as e:
            logger.warning(f"Dashboard streaming unavailable: {e}")
            self.camera_processor = None

        # Video writers for saving output
        self.save_video = kwargs.get('save_video', False)
        self.video_writers: List[Optional[cv2.VideoWriter]] = []
        if self.save_video:
            self._init_video_writers(kwargs.get('output_dir', 'volumes/storage/person-tracking'))

        # Display settings
        self.show_display = kwargs.get('show', False)

        # Performance tracking
        self.running = False
        self.total_frames = 0
        self.start_time = 0

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"SmartOfficeEngine initialized with {len(self.camera_configs)} camera(s)")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.warning(f"Received signal {signum}, shutting down...")
        self.running = False

    def _fetch_camera_configs(self) -> List[Dict]:
        """Fetch camera configurations from API for all applications."""
        all_configs = []

        for application in self.applications:
            cameras = self.api_client.get_cameras(application=application)

            for cam in cameras:
                config = {
                    'camera_id': cam.get('id'),
                    'camera_name': cam.get('name', 'Unknown'),
                    'cam_type': cam.get('camera_type', 'IN').upper(),
                    'stream_url': cam.get('stream_url', ''),
                    'application': application,
                    'match_threshold': float(cam.get('matching_threshold', 0.3)),
                    'roi': self._parse_roi(cam.get('roi_points')),
                    'line_points': self._parse_line_points(cam.get('virtual_line_points'))
                }
                all_configs.append(config)

                logger.info(
                    f"Camera: {config['camera_name']} | "
                    f"Type: {config['cam_type']} | "
                    f"App: {application}"
                )

        return all_configs

    def _parse_roi(self, roi_points) -> Optional[Tuple[int, int, int, int]]:
        """Parse ROI points from API format."""
        if roi_points and len(roi_points) >= 2:
            return tuple(roi_points[0] + roi_points[1])
        return None

    def _parse_line_points(self, line_points) -> Optional[List[Tuple[int, int]]]:
        """Parse virtual line points from API format."""
        if line_points and len(line_points) >= 2:
            return [tuple(line_points[0]), tuple(line_points[1])]
        return None

    def _load_phone_detection_config(self) -> Dict[str, Any]:
        """Load phone detection configuration from config file.

        Returns:
            Dictionary with phone detection config (model_size, confidence_threshold)
        """
        config_path = Path('configs/person_tracking/config.yaml')

        # Default values
        default_config = {
            'model_size': 'n',
            'confidence_threshold': 0.4
        }

        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Get phone detection config from first camera (shared across all)
                if config and 'cameras' in config and len(config['cameras']) > 0:
                    camera_config = config['cameras'][0]
                    phone_config = camera_config.get('phone_detection', {})

                    result = {
                        'model_size': phone_config.get('model_size', default_config['model_size']),
                        'confidence_threshold': phone_config.get('confidence_threshold', default_config['confidence_threshold'])
                    }

                    logger.info(
                        f"Loaded phone detection config from {config_path}: "
                        f"model_size={result['model_size']}, threshold={result['confidence_threshold']}"
                    )
                    return result
            else:
                logger.warning(f"Config file not found: {config_path}, using defaults")

        except Exception as e:
            logger.error(f"Error loading phone detection config: {e}, using defaults")

        return default_config

    def _init_face_recognizer(self) -> FaceRecognition:
        """Initialize face recognizer with pgvector."""
        # Create args object for FaceRecognition
        args = type('Args', (), {})()
        args.use_pgvector = os.getenv('USE_PGVECTOR', 'true').lower() == 'true'
        args.client_slug = self.client_slug
        args.match_threshold = 0.3
        args.logger = logger
        args.db_path = None

        return FaceRecognition(args)

    def _init_cameras(self) -> None:
        """Initialize stream handlers and camera engines."""
        for config in self.camera_configs:
            # Create stream handler
            stream = StreamHandler(
                src=config['stream_url'],
                logger=logger
            )
            self.streams.append(stream)

            # Create camera engine with shared detectors
            engine = CameraEngine(
                camera_config=config,
                face_detector=self.face_detector,
                face_recognizer=self.face_recognizer,
                person_detector=self.shared_person_detector,
                phone_detector=self.shared_phone_detector,
                client_slug=self.client_slug
            )
            self.camera_engines.append(engine)

    def _init_video_writers(self, output_dir: str) -> None:
        """Initialize video writers for saving output."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)

        for config in self.camera_configs:
            camera_name = config['camera_name'].replace(' ', '_')
            filename = f"{output_dir}/{camera_name}_{timestamp}.avi"

            # Get frame dimensions from stream
            stream_idx = len(self.video_writers)
            if stream_idx < len(self.streams):
                stream = self.streams[stream_idx]
                if hasattr(stream, 'frame') and stream.frame is not None:
                    h, w = stream.frame.shape[:2]
                else:
                    w, h = 1920, 1080  # Default
            else:
                w, h = 1920, 1080

            writer = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*'XVID'),
                20,
                (w, h)
            )
            self.video_writers.append(writer)
            logger.info(f"Video writer initialized: {filename}")

    def _init_entry_logger(self, email: str, password: str) -> EntryLogger:
        """Initialize entry logger."""
        # Create args object for EntryLogger
        args = type('Args', (), {})()
        args.client_slug = self.client_slug
        args.api_host = self.api_host
        args.email = email
        args.password = password
        args.logger = logger
        args.db_names = self.face_recognizer.db_names
        args.production = True  # Enable API submissions

        return EntryLogger(args=args)

    def run(self) -> None:
        """Run the main processing loop."""
        self.running = True
        self.start_time = time.time()
        frame_nums = [0] * len(self.streams)

        # Start streams
        for stream in self.streams:
            if not stream.is_video:
                stream.start()

        logger.info("SmartOfficeEngine started")

        try:
            while self.running:
                frames = []

                # Read frames from all cameras
                for i, stream in enumerate(self.streams):
                    ret, frame = stream.read()
                    if not ret:
                        logger.warning(f"Camera {i}: Failed to read frame")
                        continue

                    frame_nums[i] += 1
                    self.total_frames += 1
                    frames.append((i, frame, frame_nums[i]))

                if not frames:
                    continue

                # Process each camera
                annotated_frames = []
                for camera_idx, frame, frame_num in frames:
                    engine = self.camera_engines[camera_idx]

                    # Process frame
                    recognized, phone_events, processed, phones = engine.process_frame(frame, frame_num)

                    # Handle recognized persons
                    for person in recognized:
                        self._handle_recognized_person(person)

                    # Handle phone events
                    for event in phone_events:
                        self._handle_phone_event(event)

                    # Annotate frame (pass phones to avoid duplicate detection)
                    annotated = self._annotate_frame(processed, engine, phones)
                    annotated_frames.append((camera_idx, annotated))

                # Output annotated frames
                for camera_idx, annotated in annotated_frames:
                    camera_name = self.camera_configs[camera_idx]['camera_name']

                    # Stream to dashboard
                    if self.camera_processor:
                        try:
                            self.camera_processor.update_frame(camera_name, annotated)
                        except Exception as e:
                            logger.debug(f"Dashboard streaming error: {e}")

                    # Save to video file
                    if self.save_video and camera_idx < len(self.video_writers):
                        writer = self.video_writers[camera_idx]
                        if writer:
                            writer.write(annotated)

                    # Display window
                    if self.show_display:
                        cv2.imshow(f"SmartOffice - {camera_name}", annotated)

                # Handle display window events
                if self.show_display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit requested via keyboard")
                        self.running = False



        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise

        finally:
            self._cleanup()

    def _handle_recognized_person(self, person: Dict) -> None:
        """Handle a recognized/unrecognized person."""
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
            # Send unrecognized face
            if face_image is not None and face_image.size > 0:
                self.entry_logger.send_unrecognized_face(
                    face=face_image,
                    status=status
                )
                logger.info(f"UNRECOGNIZED | Sent face from {camera_name} ({status})")

    def _handle_phone_event(self, event: Dict) -> None:
        """Handle a phone usage event."""
        event_type = event['event_type']
        identity = event.get('identity', 'Unknown')
        camera_name = event['camera_name']

        # Log to CSV
        self.csv_logger.camera_id = event['camera_id']
        self.csv_logger.log_event(
            track_id=event['track_id'],
            event_type=event_type,
            person_name=identity,
            using_phone=event_type == 'phone_usage_started',
            confidence=0.0,
            duration=event.get('duration'),
            timestamp=event['timestamp']
        )

        # Log to console
        if event_type == 'phone_usage_started':
            logger.warning(f"PHONE | {identity} started using phone at {camera_name}")
        else:
            duration = event.get('duration', 0)
            logger.info(f"PHONE | {identity} stopped using phone at {camera_name} ({duration:.1f}s)")

    def _annotate_frame(self, frame: np.ndarray, engine: CameraEngine, phones: List[Dict]) -> np.ndarray:
        """Annotate frame with detections and status.

        Args:
            frame: Processed frame
            engine: Camera engine with state
            phones: Pre-detected phones (avoids duplicate detection)

        Returns:
            Annotated frame
        """
        # Get all person states
        states = engine.state_manager.get_all_states()

        person_states = []
        for state in states:
            # Get track info to check if person is currently visible
            track_info = engine.person_tracker.get_track_info(state.track_id)

            # Skip tracks that are not in active_tracks (person not currently detected)
            # track_info is None means the track is not in person_tracker.active_tracks
            if track_info is None:
                logger.debug(f"Track {state.track_id}: Skipping (not in active_tracks)")
                continue

            # Remove bbox immediately when person not detected in current frame (age > 0)
            age = track_info.get('age', 0)
            if age > 0:
                logger.debug(f"Track {state.track_id}: Skipping bbox (age={age}, not in current frame)")
                continue

            track_data = engine.track_manager.get_track_data(state.track_id)
            bbox = engine.track_manager.get_latest_bbox(state.track_id)
            keypoints = None
            if track_data:
                kp_history = track_data.get('keypoints', {})
                if kp_history:
                    # Handle both dict (frame_num -> keypoints) and list (keypoints history)
                    if isinstance(kp_history, dict):
                        latest_frame = max(kp_history.keys())
                        keypoints = kp_history[latest_frame]
                    elif isinstance(kp_history, list) and len(kp_history) > 0:
                        keypoints = kp_history[-1]  # Use most recent keypoints

            person_states.append({
                'track_id': state.track_id,
                'bbox': bbox if bbox is not None else [0, 0, 0, 0],
                'keypoints': keypoints,
                'identity': state.identity,
                'identity_locked': state.identity_locked,
                'using_phone': state.using_phone
            })

        # Annotate (phones already passed, no duplicate detection)
        fps = self.total_frames / (time.time() - self.start_time) if self.start_time else 0
        annotated = self.frame_annotator.annotate_frame(
            frame=frame,
            person_states=person_states,
            phones=phones,
            fps=fps
        )

        return annotated

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down SmartOfficeEngine...")

        # Stop streams
        for stream in self.streams:
            stream.stop()

        # Close video writers
        for writer in self.video_writers:
            if writer:
                writer.release()
        logger.info("Video writers closed")

        # Close display windows
        if self.show_display:
            cv2.destroyAllWindows()

        # Save entry logger status
        self.entry_logger.save_status_info()

        # Log final stats
        elapsed = time.time() - self.start_time
        fps = self.total_frames / elapsed if elapsed > 0 else 0

        logger.info(
            f"Final Stats | Frames: {self.total_frames} | "
            f"Avg FPS: {fps:.1f} | "
            f"Total Runtime: {elapsed:.0f}s"
        )

        logger.info("SmartOfficeEngine shutdown complete")

    def reload_embeddings(self) -> None:
        """Reload face embeddings from database."""
        self.face_recognizer.reload_embeddings()
        logger.info("Reloaded face embeddings")
