"""CameraEngine - Single camera stream processing engine.

This module handles per-camera processing including:
- Person detection and tracking
- Face recognition within person ROIs
- Identity management and voting
- Track merging and ID correction
"""

import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# Face recognition components
from .core.detector import FaceDetector
from .core.recognizer import FaceRecognition

# Person tracking components
from person_tracking.core.person_detector import PersonDetector
from person_tracking.core.person_tracker import PersonTracker
from person_tracking.core.track_manager import PersonTrackManager
from person_tracking.core.identity_manager import IdentityManager
from person_tracking.core.state_manager import PersonStateManager
from person_tracking.core.face_adapter import crop_person_roi
from person_tracking.core.id_switch_corrector import IDSwitchCorrector
from person_tracking.core.global_track_manager import GlobalTrackManager

# GPU memory monitoring
from .utils.gpu_monitor import GPUMemoryMonitor

# Type hints for optional dependencies
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .api.client import APIClient


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
        person_detector: PersonDetector,
        client_slug: str,
        global_id_generator: Optional[GlobalTrackIDGenerator] = None,
        api_client: Optional['APIClient'] = None,
        name_to_id_map: Optional[Dict[str, int]] = None,
        global_track_manager: Optional[GlobalTrackManager] = None,
        action_recognizer: Optional[Any] = None,
        vlm_image_padding: float = 20.0,
        unrecognized_config: Optional[Dict[str, Any]] = None
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
            action_recognizer: Optional action recognizer for activity tracking
            vlm_image_padding: Padding percentage around person bbox for VLM (0-100)
            unrecognized_config: Config for unrecognized face quality filters
        """
        self.camera_id = camera_config['camera_id']
        self.camera_name = camera_config['camera_name']
        self.cam_type = camera_config['cam_type']  # IN or OUT
        self.stream_url = camera_config['stream_url']
        self.application = camera_config.get('application', ['attendance'])
        self.match_threshold = camera_config.get('match_threshold', 0.3)
        self.roi = camera_config.get('roi')
        self.line_points = camera_config.get('line_points')
        self.vlm_image_padding = vlm_image_padding  # Padding % for VLM images

        # Unrecognized face quality settings
        unrecognized_config = unrecognized_config or {}
        self.min_face_size = unrecognized_config.get('min_face_size', 150)
        self.blur_threshold = unrecognized_config.get('blur_threshold', 50)
        self.send_best_effort = unrecognized_config.get('send_best_effort', True)

        # Shared components (models)
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.action_recognizer = action_recognizer
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
            f"Type: {self.cam_type} | VLM: {'enabled' if self._activity_enabled else 'disabled'} | "
            f"Unrecognized: {'enabled' if self._unrecognized_enabled else 'disabled'}"
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
            max_age=45,  # Phase 3: Reduced from 60 to 45 frames (~1.5s @ 30fps) for faster cleanup
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
        # Phase 3: Reduced from 100 to 60 frames (2s @ 30fps) - saves 40% memory per track
        self.track_manager = PersonTrackManager(max_history_frames=60)

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

        # Action recognition timing (track per identity name, not track_id)
        self.last_action_check_per_identity: Dict[str, float] = {}
        self._action_cache_max_age_sec = 300.0  # 5 minutes
        self._action_cache_cleanup_interval = 500  # frames

        # Frame counter (wraps at 1M to prevent overflow, safe for modulo ops)
        self.frame_count = 0
        self._frame_count_max = 1_000_000

        # Cache application-based feature flags
        self._activity_enabled = 'activity' in self.application
        self._unrecognized_enabled = 'unrecognized' in self.application

        # GPU memory monitor (optional, only if CUDA available)
        self.gpu_monitor = GPUMemoryMonitor(
            device='cuda:0',
            warning_threshold_mb=2000.0  # Warn if GPU memory > 2GB
        )

    def _is_activity_enabled(self) -> bool:
        """Check if action recognition (VLM) is enabled for this camera.

        Returns:
            True if 'activity' is in camera's applications list
        """
        return self._activity_enabled

    def _is_unrecognized_enabled(self) -> bool:
        """Check if unrecognized face submission is enabled for this camera.

        Returns:
            True if 'unrecognized' is in camera's applications list
        """
        return self._unrecognized_enabled

    def _cleanup_action_check_cache(self) -> int:
        """Remove stale entries from action check cache to prevent memory leak.

        Entries older than _action_cache_max_age_sec are removed.

        Returns:
            Number of entries removed
        """
        if not self.last_action_check_per_identity:
            return 0

        current_time = time.time()
        cutoff_time = current_time - self._action_cache_max_age_sec

        stale_identities = [
            identity for identity, last_check in self.last_action_check_per_identity.items()
            if last_check < cutoff_time
        ]

        for identity in stale_identities:
            del self.last_action_check_per_identity[identity]

        if stale_identities:
            logger.debug(
                f"Action check cache cleanup: removed {len(stale_identities)} stale entries, "
                f"{len(self.last_action_check_per_identity)} remaining"
            )

        return len(stale_identities)

    def _crop_person_with_padding(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        padding_percent: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Crop person from frame with padding for better context.

        Args:
            frame: Input frame
            bbox: Person bounding box [x1, y1, x2, y2]
            padding_percent: Padding percentage (0-100). If None, uses self.vlm_image_padding

        Returns:
            Cropped image with padding, or None if invalid
        """
        if bbox is None:
            return None

        padding = padding_percent if padding_percent is not None else self.vlm_image_padding
        x1, y1, x2, y2 = map(int, bbox)

        # Calculate padding in pixels
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding / 100)
        pad_y = int(height * padding / 100)

        # Apply padding with frame bounds check
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(frame.shape[1], x2 + pad_x)
        y2_padded = min(frame.shape[0], y2 + pad_y)

        # Crop
        if x2_padded > x1_padded and y2_padded > y1_padded:
            return frame[y1_padded:y2_padded, x1_padded:x2_padded].copy()

        return None

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

        self.frame_count = (self.frame_count + 1) % self._frame_count_max
        recognized_persons = []

        # Periodic cleanup of action check cache to prevent memory leak
        if self.frame_count % self._action_cache_cleanup_interval == 0:
            self._cleanup_action_check_cache()

        # Phase 3: Periodic track cleanup to prevent unbounded growth
        # Run every 500 frames (~16 seconds @ 30fps) to enforce global track limit
        if self.frame_count % 500 == 0:
            removed = self.track_manager.cleanup_oldest_tracks(max_tracks=100)
            if removed > 0:
                logger.info(f"Camera {self.camera_name}: Cleaned up {removed} old tracks")

        # Periodic GPU memory cleanup to prevent CUDA memory accumulation
        # Run every 100 frames (~3 seconds @ 30fps) to defragment GPU memory
        if self.frame_count % 100 == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log GPU memory stats every 300 frames (~10 seconds @ 30fps)
            if self.frame_count % 300 == 0:
                self.gpu_monitor.log_memory_stats(context=f"Camera {self.camera_name}")
            
            # Also trigger garbage collection every 1000 frames (~33 seconds @ 30fps)
            # This catches any remaining CPU/GPU memory leaks
            if self.frame_count % 1000 == 0:
                import gc
                gc.collect()
                # Log detailed memory stats after GC
                self.gpu_monitor.log_memory_stats(context=f"Camera {self.camera_name} (after GC)", force=True)

        # Step 1: Detect persons
        detections = self.person_detector.detect_persons(frame)

        # Step 2: Update tracker
        active_tracks, removed_tracks = self.person_tracker.update(detections, frame)

        # Step 3: Assign global IDs (Phase 1 cross-camera tracking)
        # Priority: 1) Face identity (if locked), 2) Body ReID
        if self.global_track_manager and self.global_track_manager.enabled:
            for track in active_tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                confidence = track.get('confidence', 0.0)

                # Get identity from state manager (if already locked from previous frames)
                identity = None
                identity_locked = False
                if self.identity_manager.is_identity_locked(track_id):
                    locked_info = self.identity_manager.get_locked_identity(track_id)
                    if locked_info:
                        identity = locked_info.get('name')
                        identity_locked = True

                # Extract person crop for body ReID
                # Optimization: Use view instead of copy - ReID model creates copies during resize/cvtColor
                person_crop = None
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        person_crop = frame[y1:y2, x1:x2]  # View only, saves ~100KB per person

                # Assign global ID (identity-first, then body ReID fallback)
                global_id = self.global_track_manager.assign_global_id(
                    camera_id=self.camera_id,
                    local_track_id=track_id,
                    person_crop=person_crop,
                    face_embedding=None,
                    detection_confidence=confidence,
                    frame_num=frame_num,
                    identity=identity,
                    identity_locked=identity_locked
                )
                track['global_track_id'] = global_id

        # Step 4: Process each active track
        for track in active_tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            keypoints = track.get('keypoints')
            confidence = track.get('confidence', 0.0)

            # Check if this is a new track (person entered frame)
            prev_state = self.state_manager.get_state(track_id)

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
                                f"RE-ID: Track {existing_track} merged into Track {track_id} "
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
                        logger.warning(f"Track {track_id} merged into Track {existing_track} (same person: {identity})")
                        self._merge_tracks(source_track_id=track_id, target_track_id=existing_track)
                        # Skip further processing for this track (it's been merged)
                        continue

                    # GLOBAL ID REASSIGNMENT: Check if another global track has this identity
                    # and reassign this track's global ID to match
                    if self.global_track_manager and self.global_track_manager.enabled:
                        current_global_id = track.get('global_track_id')
                        existing_global = self.global_track_manager.find_global_track_by_identity(identity)

                        if existing_global is not None and existing_global.global_id != current_global_id:
                            # Reassign to existing global track with same identity
                            old_global_id = current_global_id
                            new_global_id = existing_global.global_id
                            self.global_track_manager.reassign_local_track(
                                camera_id=self.camera_id,
                                local_track_id=track_id,
                                new_global_id=new_global_id
                            )
                            track['global_track_id'] = new_global_id
                            logger.info(
                                f"GLOBAL_ID_REASSIGN | identity='{identity}' "
                                f"old_global={old_global_id} -> new_global={new_global_id} "
                                f"camera={self.camera_id} local_track={track_id}"
                            )
                        elif existing_global is None and current_global_id is not None:
                            # No existing global track with this identity - update current one
                            self.global_track_manager.update_global_track_identity(
                                current_global_id, identity, locked=True
                            )

                    # Identity just locked - emit recognition event for immediate logging
                    logger.opt(colors=True).info(f"<blue> Track {track_id} recognized as '{identity}' [{self.cam_type}]</blue>")

                    # Get best quality person image as proof (full frame with bbox)
                    proof_image = self._get_best_person_image(track_id)

                    recognized_persons.append({
                        'track_id': track_id,
                        'global_track_id': track.get('global_track_id'),  # Phase 1: cross-camera ID
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

            # Action recognition (only for locked identities on cameras with 'activity' application)
            if identity_locked and self.action_recognizer and self._is_activity_enabled():
                self._check_and_queue_action_recognition(
                    track_id=track_id,
                    identity=identity,
                    frame=frame,
                    bbox=bbox,
                    frame_num=frame_num
                )

            # Store face image and person crop for later use (unrecognized faces)
            # NOTE: We store person_crop instead of full frame to prevent memory explosion
            # Full frame at 1080p = ~6MB, person crop = ~100KB typically
            if face_image is not None and bbox is not None:
                # Extract person crop from bbox (much smaller than full frame)
                bx1, by1, bx2, by2 = map(int, bbox)
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(frame.shape[1], bx2), min(frame.shape[0], by2)
                person_crop = frame[by1:by2, bx1:bx2].copy() if bx2 > bx1 and by2 > by1 else None

                self.track_manager.track_crop_history.setdefault(track_id, {})[frame_num] = {
                    'face': face_image,
                    'bbox': bbox,
                    'person_crop': person_crop  # Store person crop, NOT full frame
                }

                # Enforce crop history limit (prevent memory leak)
                max_crops = 50
                crop_hist = self.track_manager.track_crop_history[track_id]
                if len(crop_hist) > max_crops:
                    oldest_frames = sorted(crop_hist.keys())[:-max_crops]
                    for old_frame in oldest_frames:
                        del crop_hist[old_frame]

        # Step 5: Process removed tracks (person left frame)
        for track in removed_tracks:
            track_id = track['track_id']

            # Get state before cleanup
            state = self.state_manager.get_state(track_id)

            # Log person exit
            if state:
                identity_str = f"'{state.identity}'" if state.identity_locked else f"Track {track_id}"

            # If person was NOT recognized (identity never locked), send unrecognized person image
            if state and not state.identity_locked:
                person_image = self._get_best_person_image(track_id)
                if person_image is not None and person_image.size > 0:
                    # Get global track ID if available
                    global_track_id = None
                    if self.global_track_manager and self.global_track_manager.enabled:
                        global_track_id = self.global_track_manager.get_global_id(
                            self.camera_id, track_id
                        )

                    # Add to recognized_persons list for API submission
                    recognized_persons.append({
                        'track_id': track_id,
                        'global_track_id': global_track_id,  # Phase 1: cross-camera ID
                        'name': None,
                        'recognized': False,
                        'confidence': 0.0,
                        'appear_time': state.first_seen,
                        'camera_name': self.camera_name,
                        'camera_id': self.camera_id,
                        'status': self.cam_type,
                        'face_image': person_image,  # Actually person image, but API expects this key
                        'send_unrecognized': self._is_unrecognized_enabled(),  # Per-camera flag
                    })

            # Notify GlobalTrackManager of track removal (Phase 1)
            if self.global_track_manager and self.global_track_manager.enabled:
                track_data = self.track_manager.get_track_data(track_id)
                self.global_track_manager.on_track_removed(
                    camera_id=self.camera_id,
                    local_track_id=track_id,
                    track_history=track_data.get('history') if track_data else None,
                    total_frames=track_data.get('total_frames', 0) if track_data else 0
                )

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
                        f"ID CORRECTION: Merging Track {merge_track} into Track {keep_track} "
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

        # Use configurable thresholds
        min_face_size = self.min_face_size
        blur_threshold = self.blur_threshold

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

        # If we found a best frame, return the stored person crop
        if best_frame_num is not None:
            crop_data = crops[best_frame_num]
            if isinstance(crop_data, dict):
                # Try person_crop first (new format), then frame+bbox (legacy)
                person_crop = crop_data.get('person_crop')
                if person_crop is not None:
                    return person_crop

                # Legacy fallback: crop from stored frame
                frame = crop_data.get('frame')
                bbox = crop_data.get('bbox')
                if frame is not None and bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    return frame[y1:y2, x1:x2]

                # Final fallback to face crop
                return crop_data.get('face')
            else:
                # Old format - return face crop
                return crop_data

        # FALLBACK: If no "best" image found (all failed quality checks),
        # optionally return ANY available image rather than None
        if not self.send_best_effort:
            logger.debug(f"Track {track_id}: No high-quality image found, send_best_effort=False, skipping")
            return None

        logger.debug(f"Track {track_id}: No high-quality image found, using fallback (best effort)")

        # Try to get the most recent frame (last in history)
        if crops:
            # Get most recent frame_num
            latest_frame_num = max(crops.keys())
            crop_data = crops[latest_frame_num]

            if isinstance(crop_data, dict):
                # Try person_crop first (new format), then frame+bbox (legacy)
                person_crop = crop_data.get('person_crop')
                if person_crop is not None:
                    return person_crop

                # Legacy fallback: crop from stored frame
                frame = crop_data.get('frame')
                bbox = crop_data.get('bbox')
                if frame is not None and bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    return frame[y1:y2, x1:x2]

                # Final fallback to face crop
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

    def _check_and_queue_action_recognition(
        self,
        track_id: int,
        identity: str,
        frame: np.ndarray,
        bbox: np.ndarray,
        frame_num: int
    ) -> None:
        """Check if action recognition is needed and queue request.

        Args:
            track_id: Track ID
            identity: Person identity (name)
            frame: Current video frame
            bbox: Person bounding box [x1, y1, x2, y2]
            frame_num: Current frame number
        """
        if bbox is None:
            return

        # Crop person with padding for better context (desk, computer, phone, etc.)
        vlm_image = self._crop_person_with_padding(frame, bbox)
        if vlm_image is None or vlm_image.size == 0:
            return

        # Check if enough time has passed since last action check (per identity, not per track_id)
        current_time = time.time()
        last_check_time = self.last_action_check_per_identity.get(identity, 0.0)
        time_since_last_check = current_time - last_check_time

        if time_since_last_check < self.action_recognizer.check_interval_seconds:
            return  # Too soon, skip

        # Update last check time for this identity
        self.last_action_check_per_identity[identity] = current_time

        # Get user ID from name
        user_id = self.name_to_id_map.get(identity)
        if user_id is None:
            logger.warning(f"Cannot find user_id for '{identity}', skipping action recognition")
            return

        # Create unique request ID
        request_id = f"cam{self.camera_id}_track{track_id}_frame{frame_num}"

        # Create callback function
        def action_result_callback(result: Dict):
            self._handle_action_result(
                track_id=track_id,
                user_id=user_id,
                identity=identity,
                result=result,
                timestamp=current_time,
                vlm_image=vlm_image
            )

        # Queue for async recognition
        queued = self.action_recognizer.recognize_async(
            image=vlm_image,
            request_id=request_id,
            callback=action_result_callback,
            metadata={
                'track_id': track_id,
                'identity': identity,
                'user_id': user_id,
                'camera_id': self.camera_id,
                'frame_num': frame_num
            }
        )

        if queued:
            logger.debug(
                f"Queued action recognition | {identity} (track id {track_id}) | "
                f"camera={self.camera_name}"
            )
        else:
            logger.warning(f"Failed to queue action recognition (queue full)")

    def _handle_action_result(
        self,
        track_id: int,
        user_id: int,
        identity: str,
        result: Dict,
        timestamp: float,
        vlm_image: np.ndarray
    ) -> None:
        """Handle action recognition result and send to API.

        Args:
            track_id: Track ID
            user_id: User ID in database
            identity: Person identity (name)
            result: Action recognition result
            timestamp: Unix timestamp
            vlm_image: Person crop image with padding (sent to VLM)
        """
        action = result.get('action')
        if not action:
            logger.debug(f"No action detected for {identity} (track {track_id})")
            return

        inference_time = result.get('inference_time', 0.0)
        raw_output = result.get('raw_output', '')

        logger.info(
            f"ACTION DETECTED | {identity}: {action} | "
            f"time={inference_time:.3f}s | camera={self.camera_name}"
        )

        # Update state
        state = self.state_manager.get_state(track_id)
        if state:
            state.last_detected_action = action

        # Note: Activity is already sent to backend by ActionRecognizer._post_activity_to_backend()
        # No need to send again here to avoid duplicate API calls

    def reset(self) -> None:
        """Reset all tracking state."""
        self.person_tracker.reset()
        self.track_manager.reset()
        self.identity_manager.reset()
        self.state_manager.reset()
        self.frame_count = 0
        self.last_action_check_per_identity.clear()

