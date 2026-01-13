"""SmartOfficeEngine - Unified person tracking and face recognition system.

This module provides a unified engine that combines:
- Person detection and tracking (YOLOv8-Pose + BoT-SORT)
- Face recognition within person ROIs (InsightFace)
- Attendance logging (IN/OUT status)

Replaces HBFace with improved architecture:
- Track persons instead of faces for stable IDs
- Temporal voting for identity locking
"""

import os
import sys
import time
import signal
import yaml
import threading
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
from person_tracking.core.identity_manager import IdentityManager
from person_tracking.core.state_manager import PersonStateManager
from person_tracking.core.face_adapter import crop_person_roi
from person_tracking.core.id_switch_corrector import IDSwitchCorrector
from person_tracking.core.global_track_manager import GlobalTrackManager
from person_tracking.video.frame_annotator import FrameAnnotator


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
        logger.info(f"GlobalTrackIDGenerator initialized (start_id={start_id})")

    def get_next_id(self) -> int:
        """Get next globally unique track ID (thread-safe).

        Returns:
            Next unique track ID
        """
        with self._lock:
            track_id = self._current_id
            self._current_id += 1
            return track_id

    def get_current_id(self) -> int:
        """Get current track ID without incrementing.

        Returns:
            Current track ID value
        """
        with self._lock:
            return self._current_id


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
        client_slug: str,
        global_id_generator: Optional[GlobalTrackIDGenerator] = None,
        api_client: Optional['APIClient'] = None,
        name_to_id_map: Optional[Dict[str, int]] = None,
        global_track_manager: Optional['GlobalTrackManager'] = None
    ):
        """Initialize camera engine.

        Args:
            camera_config: Camera configuration from API
            face_detector: Shared face detector instance
            face_recognizer: Shared face recognizer instance
            person_detector: Shared person detector instance
            client_slug: Client organization slug
            global_id_generator: Optional global track ID generator for cross-camera unique IDs
            api_client: API client for sending activities
            name_to_id_map: Dictionary mapping user names to IDs
            global_track_manager: Optional GlobalTrackManager for Phase 0 instrumentation
        """
        self.camera_id = camera_config['camera_id']
        self.camera_name = camera_config['camera_name']
        self.cam_type = camera_config['cam_type']  # IN or OUT
        self.stream_url = camera_config['stream_url']
        self.application = camera_config.get('application', ['attendance'])
        self.match_threshold = camera_config.get('match_threshold', 0.3)
        self.min_face_size = camera_config.get('min_face_size', 150)  # Minimum face size for quality check
        self.roi = camera_config.get('roi')
        self.line_points = camera_config.get('line_points')

        # Shared components (models)
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.client_slug = client_slug
        self.global_id_generator = global_id_generator
        self.global_track_manager = global_track_manager

        # API client and name mapping for activity tracking
        self.api_client = api_client
        self.name_to_id_map = name_to_id_map or {}

        # Initialize per-camera components (tracking, state)
        self._init_components()

        logger.info(
            f"CameraEngine initialized: {self.camera_name} (ID: {self.camera_id}) | "
            f"Type: {self.cam_type}"
        )

    def _init_components(self) -> None:
        """Initialize tracking and state management components.

        Note: Detection models (person_detector) are shared
        and passed in __init__, not created here.
        """
        # Person Tracking (per-camera for isolated state)
        # BoT-SORT uses external detections from PersonDetector (single YOLO pass)
        self.person_tracker = PersonTracker(
            tracker_type='botsort',  # Use BoT-SORT (IoU + Kalman filter)
            max_age=60,  # Keep tracks alive for ~2 seconds at 30fps
            min_hits=3,  # Require 3 consecutive detections before confirming track
            iou_threshold=0.8,  # IoU threshold for matching (BoT-SORT default)
            global_id_generator=self.global_id_generator,  # Enable global track IDs
            global_track_manager=self.global_track_manager,  # Phase 0: instrumentation
            camera_id=self.camera_id,
            confidence_threshold=self.person_detector.confidence_threshold,
            with_reid=False,  # Disable ReID (falls back to IoU + Kalman - still better than simple IoU)
            frame_rate=30,  # Assume 30 FPS for tracker buffer calculation
            device='cuda:0' if self.person_detector.device == 'cuda' else 'cpu'
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

        # ID Switch Corrector (face-based correction)
        self.id_corrector = IDSwitchCorrector(
            embedding_distance_threshold=0.4,  # Cosine distance for "same person"
            correction_interval_frames=5,  # Check every 5 frames
            min_embedding_samples=3  # Need 3+ embeddings for reliable comparison
        )

        # State Manager
        self.state_manager = PersonStateManager(
            camera_id=self.camera_id,
            api_client=self.api_client,
            name_to_id_map=self.name_to_id_map
        )

        # Frame counter
        self.frame_count = 0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> Tuple[List[Dict], np.ndarray]:
        """Process a single frame.

        Args:
            frame: Input video frame
            frame_num: Frame number

        Returns:
            Tuple of (recognized_persons, processed_frame)
        """
        # Apply ROI if configured
        if self.roi:
            x1, y1, x2, y2 = self.roi
            frame = frame[y1:y2, x1:x2]

        self.frame_count += 1
        recognized_persons = []

        # Step 1: Detect persons
        detections = self.person_detector.detect_persons(frame)

        # Step 2: Update tracker
        active_tracks, removed_tracks = self.person_tracker.update(detections, frame)

        # Step 4: Process each active track
        for track in active_tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            keypoints = track.get('keypoints')
            confidence = track.get('confidence', 0.0)

            # Check if this is a new track (person entered frame)
            prev_state = self.state_manager.get_state(track_id)
            # if prev_state is None:
            #     # Color based on camera type: Green for IN, Red for OUT
            #     if self.cam_type == "IN":
            #         logger.opt(colors=True).info(f"<green>→ Track {track_id} ENTERED frame [IN]</green>")
            #     else:
            #         logger.opt(colors=True).info(f"<red>→ Track {track_id} ENTERED frame [OUT]</red>")

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
                result = self._recognize_face(frame, bbox, track_id)

                if result['face_detected']:
                    # Store embedding for ID correction (all detected faces)
                    embedding = result.get('embedding')
                    if embedding is not None:
                        recognized_name = result.get('name') if result.get('recognized') else None
                        self.id_corrector.add_embedding(track_id, embedding, recognized_name)

                    # EARLY RE-ID: If this is a new track and face is recognized,
                    # check if this person already has an active track
                    if prev_state is None and result.get('recognized'):
                        recognized_name = result.get('name')
                        existing_track = self._find_track_with_identity(recognized_name, exclude_track_id=track_id)

                        if existing_track is not None:
                            # RE-IDENTIFICATION: This person already has an active track!
                            # Merge this new track into the existing one
                            logger.warning(
                                f"🔄 RE-ID: Track {existing_track} merged into Track {track_id} "
                                f"(returning person: {recognized_name})"
                            )
                            self._merge_tracks(source_track_id=existing_track, target_track_id=track_id)
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
                    logger.opt(colors=True).info(f"<blue> Track {track_id} recognized as '{identity}' [{self.cam_type}]</blue>")

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
                    # TIER 2: Identity consistency check (for already-locked identities)
                    # Continuously verify face matches locked identity
                    result = self._recognize_face(frame, bbox, track_id)
                    if result.get('face_detected') and result.get('embedding') is not None:
                        embedding = result['embedding']

                        # Add embedding to corrector
                        self.id_corrector.add_embedding(track_id, embedding, identity)

                        # Check consistency
                        is_consistent = self.id_corrector.check_identity_consistency(
                            track_id, embedding, identity
                        )

                        # if not is_consistent:
                        #     logger.error(
                        #         f"ID SWITCH DETECTED: Track {track_id} face doesn't match "
                        #         f"locked identity '{identity}'! Will be corrected in batch."
                        #     )
            else:
                voting = self.identity_manager.get_voting_status(track_id)
                if voting and voting.get('top_candidate'):
                    identity = voting['top_candidate']
                    identity_confidence = voting.get('top_avg_similarity', 0.0)

            # Get proof image (crop person from frame)
            proof_image = None
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                proof_image = frame[y1:y2, x1:x2]

            # Update state manager
            self.state_manager.update_person(
                track_id=track_id,
                identity=identity,
                identity_locked=identity_locked,
                identity_confidence=identity_confidence,
                proof_image=proof_image
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
                # if self.cam_type == "IN":
                #     logger.opt(colors=True).info(f"<green>← {identity_str} LEFT frame [IN]</green>")
                # else:
                #     logger.opt(colors=True).info(f"<magenta>← {identity_str} LEFT frame [OUT]</magenta>")

            # If person was NOT recognized (identity never locked), send unrecognized person image
            if state and not state.identity_locked:
                # logger.info(f"📸 Sending unrecognized person image for Track {track_id}")
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
            self.id_corrector.reset_track(track_id)  # Clean up embeddings

        # Delayed batch ID correction (every 5 frames)
        if self.id_corrector.should_run_correction():
            duplicates = self.id_corrector.find_duplicate_tracks()

            for track_id_1, track_id_2, distance in duplicates:
                # Determine which track to keep (lower ID = older track)
                if track_id_1 < track_id_2:
                    keep_track, merge_track = track_id_1, track_id_2
                else:
                    keep_track, merge_track = track_id_2, track_id_1

                # Check both tracks still exist
                if (self.person_tracker.get_track_info(keep_track) is not None and
                    self.person_tracker.get_track_info(merge_track) is not None):

                    logger.warning(
                        f"🔧 ID CORRECTION: Merging Track {merge_track} into Track {keep_track} "
                        f"(duplicate detected, face distance: {distance:.3f})"
                    )
                    self._merge_tracks(source_track_id=merge_track, target_track_id=keep_track)

        return recognized_persons, frame

    def _recognize_face(self, frame: np.ndarray, bbox: np.ndarray, track_id: int = 0) -> Dict:
        """Recognize face within person bounding box."""
        roi, offset = crop_person_roi(frame, bbox, expand=0.1)

        if roi is None or roi.size == 0:
            # Phase 0: Log face not visible
            if self.global_track_manager:
                self.global_track_manager.on_face_not_visible(
                    camera_id=self.camera_id,
                    local_track_id=track_id
                )
            return {'face_detected': False, 'name': None, 'similarity': 0.0}

        # Detect face
        faces = self.face_detector.detect(roi)

        if not faces:
            # Phase 0: Log face not visible
            if self.global_track_manager:
                self.global_track_manager.on_face_not_visible(
                    camera_id=self.camera_id,
                    local_track_id=track_id
                )
            return {'face_detected': False, 'name': None, 'similarity': 0.0}

        face = faces[0]
        embedding = face.embedding / np.linalg.norm(face.embedding)

        # Get face crop for unrecognized faces
        face_bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = face_bbox
        face_image = roi[max(0, y1):y2, max(0, x1):x2]

        # Match against database
        if len(self.face_recognizer.db_embs) == 0:
            # Phase 0: Log face detected but not recognized
            if self.global_track_manager:
                self.global_track_manager.on_face_detected(
                    camera_id=self.camera_id,
                    local_track_id=track_id,
                    quality=float(face.det_score) if hasattr(face, 'det_score') else 0.0,
                    recognized=False,
                    identity=None
                )
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
            # Phase 0: Log face detected and recognized
            if self.global_track_manager:
                self.global_track_manager.on_face_detected(
                    camera_id=self.camera_id,
                    local_track_id=track_id,
                    quality=float(face.det_score) if hasattr(face, 'det_score') else 0.0,
                    recognized=True,
                    identity=name
                )
            return {
                'face_detected': True,
                'recognized': True,
                'name': name,
                'similarity': best_similarity,
                'embedding': embedding,
                'face_image': face_image
            }
        else:
            # Phase 0: Log face detected but not recognized
            if self.global_track_manager:
                self.global_track_manager.on_face_detected(
                    camera_id=self.camera_id,
                    local_track_id=track_id,
                    quality=float(face.det_score) if hasattr(face, 'det_score') else 0.0,
                    recognized=False,
                    identity=None
                )
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

        # FALLBACK: If no "best" image found (all failed quality checks),
        # return ANY available image rather than None
        logger.debug(f"Track {track_id}: No high-quality image found, using fallback (any available image)")

        # Try to get the most recent frame (last in history)
        if crops:
            # Get most recent frame_num
            latest_frame_num = max(crops.keys())
            crop_data = crops[latest_frame_num]

            if isinstance(crop_data, dict):
                frame = crop_data.get('frame')
                bbox = crop_data.get('bbox')

                if frame is not None and bbox is not None:
                    # Return person crop from latest frame
                    x1, y1, x2, y2 = map(int, bbox)
                    person_crop = frame[y1:y2, x1:x2]
                    return person_crop
                else:
                    # Return face crop if available
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
        self.id_corrector.reset_track(source_track_id)  # Clean up embeddings

        # Remove from person_tracker's active_tracks
        if source_track_id in self.person_tracker.active_tracks:
            del self.person_tracker.active_tracks[source_track_id]

    def reset(self) -> None:
        """Reset all tracking state."""
        self.person_tracker.reset()
        self.track_manager.reset()
        self.identity_manager.reset()
        self.state_manager.reset()
        self.frame_count = 0


class SmartOfficeEngine:
    """Unified engine for Smart Office person tracking and face recognition.

    This class replaces HBFace with an improved architecture that:
    - Tracks persons instead of faces for stable IDs
    - Uses temporal voting for identity locking
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
                         Default: ['attendance']
                         Options: 'attendance', 'unrecognized', 'activity'
            **kwargs: Additional configuration
                     - person_detection_threshold: Person detection confidence (default: 0.5)
        """
        self.client_slug = client_slug
        self.api_host = api_host
        self.applications = applications or ['attendance']

        # Store credentials for entry logger
        self._email = email
        self._password = password

        # Store configuration
        self.config = kwargs

        # Initialize API client
        self.api_client = APIClient(
            api_host=api_host,
            email=email,
            password=password,
            client_slug=client_slug
        )

        # Load camera configurations from config file or environment
        # Fallback to API only if USE_API_FOR_CAMERAS env var is explicitly set
        use_api_for_cameras = os.getenv('USE_API_FOR_CAMERAS', 'false').lower() == 'true'

        if use_api_for_cameras:
            logger.info("Loading camera configs from API (USE_API_FOR_CAMERAS=true)")
            self.camera_configs = self._fetch_camera_configs_from_api()
        else:
            logger.info("Loading camera configs from config file or environment")
            self.camera_configs = self._load_camera_configs_from_file()

        if not self.camera_configs:
            raise ValueError(f"No cameras configured. Check config file or environment variables.")

        # Initialize global track ID generator for cross-camera unique IDs
        self.global_id_generator = GlobalTrackIDGenerator(start_id=1)
        logger.info("Global track ID generator enabled - track IDs will be unique across all cameras")

        self.global_track_manager = GlobalTrackManager()
        if self.global_track_manager.enabled:
            logger.info("GlobalTrackManager enabled - collecting baseline metrics")

        # Initialize shared detection models (GPU efficiency)
        logger.info("Initializing shared detection models...")
        self.face_detector = FaceDetector(gpu_id=0, model_name='buffalo_l')
        self.face_recognizer = self._init_face_recognizer()

        # Shared person detector (YOLOv8 - faster than YOLOv8-Pose)
        # Set use_pose=True if you want skeleton visualization (slower)
        person_conf_threshold = self.config.get('person_detection_threshold', 0.5)
        self.shared_person_detector = PersonDetector(
            model_size='s',
            confidence_threshold=person_conf_threshold,
            use_pose=False  # False = YOLOv8 (faster), True = YOLOv8-Pose (skeleton viz)
        )
        logger.info(f"Person detection threshold: {person_conf_threshold}")

        # Startup embedding sync - always runs to ensure database is up to date
        self._sync_embeddings_on_startup()

        # Create name-to-ID mapping for activity tracking
        # This requires fetching user data from API
        self.name_to_id_map = self._build_name_to_id_map()

        # Initialize streams and engines
        self.streams: List[StreamHandler] = []
        self.camera_engines: List[CameraEngine] = []
        self._init_cameras()

        # Initialize entry logger
        self.entry_logger = self._init_entry_logger(email, password)

        # Visualization
        self.visualize = Visualization()
        self.frame_annotator = FrameAnnotator()

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

    def _load_camera_configs_from_file(self) -> List[Dict]:
        """Load camera configurations from config file or environment variables."""
        from .config.camera_loader import get_camera_configs, convert_to_smart_office_format

        try:
            # Get camera configs (from file or env)
            cameras = get_camera_configs()

            # Convert to SmartOfficeEngine format
            configs = convert_to_smart_office_format(cameras)

            # Filter by applications if specified
            if self.applications:
                filtered_configs = []
                for config in configs:
                    if config['application'] in self.applications:
                        filtered_configs.append(config)
                configs = filtered_configs

            logger.info(f"Loaded {len(configs)} camera configuration(s)")
            for config in configs:
                logger.info(
                    f"Camera: {config['camera_name']} | "
                    f"Type: {config['cam_type']} | "
                    f"App: {config['application']}"
                )

            return configs
        except Exception as e:
            logger.error(f"Failed to load camera configs: {e}")
            raise

    def _fetch_camera_configs_from_api(self) -> List[Dict]:
        """Fetch camera configurations from API for all applications (legacy method)."""
        all_configs = []

        for application in self.applications:
            cameras = self.api_client.get_cameras(application=application)

            for cam in cameras:
                config = {
                    'camera_id': cam.get('id'),
                    'camera_name': cam.get('name', 'Unknown'),
                    'cam_type': cam.get('camera_type', 'IN').upper(),
                    'stream_url': cam.get('stream_url', ''),
                    'application': cam.get('application', application),  # Use camera's actual application field
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

    def _build_name_to_id_map(self) -> Dict[str, int]:
        """Build mapping of user names to IDs from API.

        Returns:
            Dictionary mapping name -> user_id
        """
        try:
            users = self.api_client.get_users()
            name_map = {}

            for user in users:
                name = user.get('name')
                user_id = user.get('id')
                if name and user_id:
                    name_map[name] = user_id

            logger.info(f"Built name-to-ID mapping for {len(name_map)} users")
            return name_map

        except Exception as e:
            logger.warning(f"Failed to build name-to-ID map: {e}")
            return {}

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

    def _sync_embeddings_on_startup(self) -> None:
        """Sync missing embeddings on startup.

        This method checks if there are any new users or images in the backend
        that don't have embeddings in pgvector, and calculates them automatically.
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTUP EMBEDDING SYNC")
            logger.info("=" * 80)

            # Import EmbeddingSyncService
            from .services.embedding_sync import EmbeddingSyncService

            # Initialize sync service
            sync_service = EmbeddingSyncService(
                client_slug=self.client_slug,
                gpu_id=0
            )

            # Run sync with authenticated API client
            result = sync_service.sync_missing_embeddings(api_client=self.api_client)

            if result.get('success'):
                users_processed = result.get('users_processed', 0)
                embeddings_added = result.get('embeddings_added', 0)

                if users_processed > 0:
                    logger.info(
                        f"✅ Startup sync complete: {users_processed} users processed, "
                        f"{embeddings_added} embeddings added"
                    )

                    # Reload embeddings into face recognizer
                    self.face_recognizer.reload_embeddings()
                    logger.info("Face recognizer reloaded with new embeddings")
                else:
                    logger.info("✅ No missing embeddings - database is up to date")
            else:
                logger.error(f"❌ Startup sync failed: {result.get('error')}")

            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Failed to sync embeddings on startup: {e}")
            logger.warning("Continuing with existing embeddings...")

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
                client_slug=self.client_slug,
                global_id_generator=self.global_id_generator,  # Enable global track IDs
                api_client=self.api_client,  # Pass API client for activity tracking
                name_to_id_map=self.name_to_id_map,  # Pass name-to-ID mapping
                global_track_manager=self.global_track_manager  # Phase 0: instrumentation
            )
            self.camera_engines.append(engine)

    def _init_video_writers(self, output_dir: str) -> None:
        """Initialize video writers for saving output."""
        from datetime import datetime

        now = datetime.now()
        date = now.strftime("%Y%m%d")
        time = now.strftime("%H%M%S")

        # Create output directory and check permissions
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Test if directory is writable
            test_file = os.path.join(output_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Output directory ready: {output_dir}")
        except Exception as e:
            logger.error(f"❌ Output directory not writable: {output_dir} - {e}")
            return

        for config in self.camera_configs:
            camera_name = config['camera_name'].replace(' ', '_')
            status = config.get('cam_type', 'IN').upper()  # IN or OUT
            # Format: status_cameraName_date_time.avi
            filename = f"{output_dir}/{status}_{camera_name}_{date}_{time}.avi"

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

            logger.debug(f"Attempting to create video writer: {filename} ({w}x{h})")

            try:
                # Use MJPEG codec - most reliable for OpenCV, no external dependencies
                # MJPEG = Motion JPEG, always available in OpenCV
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                logger.debug(f"FourCC code: {fourcc}")

                writer = cv2.VideoWriter(
                    filename,
                    fourcc,
                    20,  # FPS
                    (w, h)
                )

                if writer.isOpened():
                    self.video_writers.append(writer)
                    logger.info(f"✅ Video writer initialized: {filename} ({w}x{h}) [MJPEG codec]")
                else:
                    writer.release()
                    self.video_writers.append(None)
                    # More detailed error message
                    import subprocess
                    cv_build_info = cv2.getBuildInformation()
                    logger.error(f"❌ Failed to open video writer: {filename} ({w}x{h})")
                    logger.error(f"OpenCV version: {cv2.__version__}")
                    logger.debug(f"OpenCV build info:\n{cv_build_info}")
            except Exception as e:
                self.video_writers.append(None)
                logger.error(f"❌ Exception initializing video writer: {filename} - {e}")
                import traceback
                logger.error(traceback.format_exc())

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

        last_metrics_log_time = time.time()
        metrics_log_interval = 60.0
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
                    recognized, processed = engine.process_frame(frame, frame_num)

                    # Handle recognized persons
                    for person in recognized:
                        self._handle_recognized_person(person)

                    # Annotate frame
                    annotated = self._annotate_frame(processed, engine)
                    annotated_frames.append((camera_idx, annotated))

                # Output annotated frames
                for camera_idx, annotated in annotated_frames:
                    camera_name = self.camera_configs[camera_idx]['camera_name']

                    # Stream to dashboard
                    if self.camera_processor:
                        try:
                            self.camera_processor.update_frame(camera_name, annotated)
                        except Exception as e:
                            # logger.debug(f"Dashboard streaming error: {e}")
                            pass

                    # Save to video file
                    if self.save_video and camera_idx < len(self.video_writers):
                        writer = self.video_writers[camera_idx]
                        if writer:
                            writer.write(annotated)

                # Phase 0: Periodic baseline metrics logging
                current_time = time.time()
                if current_time - last_metrics_log_time >= metrics_log_interval:
                    if self.global_track_manager and self.global_track_manager.enabled:
                        self.global_track_manager.log_baseline_summary()
                    last_metrics_log_time = current_time

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

            # Skip tracks that are not in active_tracks (person not currently detected)
            # track_info is None means the track is not in person_tracker.active_tracks
            if track_info is None:
                # Verbose logging disabled to reduce log noise
                # logger.debug(f"Track {state.track_id}: Skipping (not in active_tracks)")
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
                'track_age': age,  # Add age for frame presence check
                'in_current_frame': (age == 0)  # Explicit flag: True only if detected in current frame
            })

        # Annotate frame
        fps = self.total_frames / (time.time() - self.start_time) if self.start_time else 0
        annotated = self.frame_annotator.annotate_frame(
            frame=frame,
            person_states=person_states,
            fps=fps
        )

        return annotated

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down SmartOfficeEngine...")

        # Phase 0: Log final baseline metrics
        if self.global_track_manager and self.global_track_manager.enabled:
            logger.info("=" * 80)
            logger.info("PHASE 0 - FINAL BASELINE METRICS")
            logger.info("=" * 80)
            self.global_track_manager.log_baseline_summary()
            metrics = self.global_track_manager.get_baseline_metrics()
            logger.info(f"Total tracks created: {metrics['total_tracks_created']}")
            logger.info(f"Total tracks removed: {metrics['total_tracks_removed']}")
            logger.info(f"Average track duration: {metrics['avg_track_duration_sec']:.1f}s")
            logger.info(f"Face visibility rate: {metrics['face_visibility_rate']:.1%}")
            logger.info(f"Faces detected: {metrics['total_faces_detected']}")
            logger.info(f"Faces not visible: {metrics['total_faces_not_visible']}")
            logger.info("=" * 80)

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
