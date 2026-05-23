"""CameraEngine - Single camera stream processing engine.

This module handles per-camera processing including:
- Person detection and tracking
- Face recognition within person ROIs
- Identity management and voting
- Track merging and ID correction
"""

import threading
import time
from collections import Counter
from datetime import datetime
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# Face recognition components
from domain.face_detection import FaceDetector
from domain.face_detection.recognizer import FaceRecognition

# Person tracking components
from domain.person_tracking import (
    PersonDetector,
    PersonTracker,
    PersonTrackManager,
    IdentityManager,
    PersonStateManager,
    GlobalTrackManager,
    IDSwitchCorrector,
)
from domain.person_tracking.face_adapter import crop_person_roi
from domain.action_recognition.recognizer import (
    PRIORITY_PHONE,
    PRIORITY_GENERAL,
    PRIORITY_UNKNOWN,
)

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
        logger.debug(f"GlobalTrackIDGenerator initialized (start_id={start_id})")

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
        name_to_id_map: Optional[Dict[str, int]] = None,
        global_track_manager: Optional[GlobalTrackManager] = None,
        action_recognizer: Optional[Any] = None
    ):
        """Initialize camera engine.

        Args:
            camera_config: Camera configuration from API
            face_detector: Shared face detector instance
            face_recognizer: Shared face recognizer instance
            person_detector: Shared person detector instance
            client_slug: Client organization slug
            global_id_generator: Optional global track ID generator for cross-camera unique IDs
            name_to_id_map: Dictionary mapping user names to IDs
            global_track_manager: Optional GlobalTrackManager for Phase 0 instrumentation
            action_recognizer: Optional action recognizer for activity tracking
        """
        self.camera_id = camera_config['camera_id']
        self.camera_name = camera_config['camera_name']
        self.cam_type = camera_config['cam_type']  # IN or OUT
        self.stream_url = camera_config['stream_url']
        self.application = camera_config.get('application', ['attendance'])
        self.match_threshold = camera_config.get('match_threshold', 0.3)
        self.match_margin = camera_config.get('match_margin', 0.10)
        self.min_face_size = camera_config.get('min_face_size', 60)
        self.blur_threshold = float(camera_config.get('blur_threshold', 30.0))
        self.identity_min_det_score = float(
            camera_config.get('identity_min_det_score', 0.65)
        )
        self.identity_lock_frames = int(camera_config.get('identity_lock_frames', 7))
        self.identity_consensus = float(camera_config.get('identity_consensus', 0.80))
        self.identity_min_window_duration_ms = int(
            camera_config.get('identity_min_window_duration_ms', 700)
        )
        self.identity_unlock_misses = int(camera_config.get('identity_unlock_misses', 4))
        self.id_switch_threshold = float(camera_config.get('id_switch_threshold', 0.55))
        # Hysteresis: only consider a locked identity mismatched if embedding distance
        # exceeds this (≥ id_switch_threshold). Stops lock/unlock flapping near the boundary.
        self.id_unlock_threshold = float(
            camera_config.get('id_unlock_threshold', max(0.70, self.id_switch_threshold + 0.15))
        )
        self.gender_age_min_face_size = int(
            camera_config.get('gender_age_min_face_size', self.min_face_size)
        )
        self.gender_age_min_det_score = float(
            camera_config.get('gender_age_min_det_score', 0.60)
        )
        self.gender_age_blur_threshold = float(
            camera_config.get('gender_age_blur_threshold', self.blur_threshold)
        )
        self.roi = camera_config.get('roi')
        self.virtual_lines = camera_config.get('virtual_lines', [])

        # Shared components (models)
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.action_recognizer = action_recognizer
        self.action_result_max_age_seconds = float(
            getattr(action_recognizer, 'max_queue_delay_seconds', 12.0)
        )
        self.client_slug = client_slug
        self.global_id_generator = global_id_generator
        self.global_track_manager = global_track_manager

        # Name mapping for activity tracking
        self.name_to_id_map = name_to_id_map or {}

        # Initialize per-camera components (tracking, state)
        self._init_components()

        logger.debug(
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
            max_age=20,  # Keep tracks alive for ~0.7s at 30fps before dropping
            min_hits=3,  # Require 3 consecutive detections before confirming track
            iou_threshold=0.45,  # IoU threshold for matching (lower = more robust re-association)
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
            identity_lock_frames=self.identity_lock_frames,
            identity_consensus=self.identity_consensus,
            min_window_duration_ms=self.identity_min_window_duration_ms,
            similarity_threshold=self.match_threshold
        )

        # ID Switch Corrector (face-based correction)
        self.id_corrector = IDSwitchCorrector(
            embedding_distance_threshold=self.id_switch_threshold,
            correction_interval_frames=5,
            min_embedding_samples=3,
            unlock_distance_threshold=self.id_unlock_threshold,
        )

        # State Manager
        self.state_manager = PersonStateManager(
            camera_id=self.camera_id,
            name_to_id_map=self.name_to_id_map
        )

        # Action recognition timing — split into two cadences:
        #   • last_phone_check_per_track: cheap YOLO pre-filter cadence (frequent)
        #   • last_action_check_per_identity: full VLM general-poll cadence (sparse)
        # Bounded: old entries pruned in _check_and_queue_action_recognition()
        self.last_action_check_per_identity: Dict[int, float] = {}
        self.last_phone_check_per_track: Dict[int, float] = {}
        self._max_action_identity_cache = 500
        self._last_queue_full_warn: float = 0.0

        # Best-scoring gender/age per track from InsightFace genderage model
        self.track_gender_age: Dict[int, Dict] = {}
        self._identity_mismatch_counts: Dict[int, int] = {}

        # Frame counter
        self.frame_count = 0

    # ── New parallel-pipeline API ─────────────────────────────────────────────

    def update_tracking(
        self,
        detections: List[Dict],
        frame: np.ndarray,
        frame_num: int,
    ) -> Tuple[List[Dict], List[Dict], List[Tuple[int, np.ndarray]]]:
        """CPU-only: update tracker and extract person ROIs for face detection.

        Args:
            detections: YOLO detections (from GPU worker or local detector)
            frame:      Current video frame (ROI already applied by caller)
            frame_num:  Frame number

        Returns:
            (active_tracks, removed_tracks, person_rois)
            person_rois = [(track_id, roi), …] — one ROI per active track
        """
        self.frame_count += 1

        # Update tracker
        active_tracks, removed_tracks = self.person_tracker.update(detections, frame)

        # GlobalTrackManager assignment (CPU-only)
        if self.global_track_manager and self.global_track_manager.enabled:
            for track in active_tracks:
                track_id = track["track_id"]
                bbox = track["bbox"]
                confidence = track.get("confidence", 0.0)

                identity = None
                identity_locked = False
                if self.identity_manager.is_identity_locked(track_id):
                    locked_info = self.identity_manager.get_locked_identity(track_id)
                    if locked_info:
                        identity = locked_info.get("name")
                        identity_locked = True

                person_crop = None
                if bbox is not None:
                    x1, y1, x2, y2 = self._clip_bbox(frame, bbox)
                    if x2 > x1 and y2 > y1:
                        person_crop = frame[y1:y2, x1:x2].copy()

                global_id = self.global_track_manager.assign_global_id(
                    camera_id=self.camera_id,
                    local_track_id=track_id,
                    person_crop=person_crop,
                    face_embedding=None,
                    detection_confidence=confidence,
                    frame_num=frame_num,
                    identity=identity,
                    identity_locked=identity_locked,
                )
                track["global_track_id"] = global_id

        # Store track detections and collect person ROIs for face detection
        person_rois: List[Tuple[int, np.ndarray]] = []
        for track in active_tracks:
            track_id = track["track_id"]
            bbox = track["bbox"]
            keypoints = track.get("keypoints")
            confidence = track.get("confidence", 0.0)

            self.track_manager.add_track_detection(
                track_id=track_id,
                frame_num=frame_num,
                bbox=bbox,
                keypoints=keypoints,
                confidence=confidence,
            )

            roi, roi_offset = crop_person_roi(frame, bbox, expand=0.1)
            person_rois.append((track_id, roi, roi_offset))

        return active_tracks, removed_tracks, person_rois

    def finalize_identities(
        self,
        active_tracks: List[Dict],
        removed_tracks: List[Dict],
        embeddings_map: Dict[int, Dict],
        frame: np.ndarray,
        frame_num: int,
    ) -> List[Dict]:
        """CPU-only: identity voting, ID correction, state updates.

        Args:
            active_tracks:  Tracks from update_tracking
            removed_tracks: Tracks from update_tracking
            embeddings_map: {track_id: {'embedding', 'face_image',
                                        'face_detected', 'det_score'}}
                            Result of GPU worker's ArcFace pass.
            frame:          Current video frame (for proof image crops)
            frame_num:      Frame number

        Returns:
            List of attendance event dicts.
        """
        recognized_persons: List[Dict] = []

        for track in active_tracks:
            track_id = track["track_id"]
            bbox = track["bbox"]

            # Compute recognition result from pre-computed embedding
            face_data = embeddings_map.get(track_id, {})
            result = self._compute_recognition_result(
                embedding=face_data.get("embedding"),
                face_image=face_data.get("face_image"),
                face_detected=face_data.get("face_detected", False),
                det_score=face_data.get("det_score", 0.0),
                face_width=face_data.get("face_width", 0),
                face_height=face_data.get("face_height", 0),
                track_id=track_id,
            )

            # Gender/age are person attributes, not employee-only data. Update
            # them for every active track with a usable face, including unknowns.
            self._update_gender_age(track_id, face_data)

            prev_state = self.state_manager.get_state(track_id)
            identity: Optional[str] = None
            identity_locked = False
            identity_confidence = 0.0
            face_image = result.get("face_image")

            if not self.identity_manager.is_identity_locked(track_id):
                if result["face_detected"]:
                    embedding = result.get("embedding")
                    if embedding is not None:
                        recognized_name = (
                            result.get("name") if result.get("recognized") else None
                        )
                        self.id_corrector.add_embedding(
                            track_id, embedding, recognized_name
                        )

                    # Early re-identification
                    if prev_state is None and result.get("recognized"):
                        recognized_name = result.get("name")
                        existing_track = self._find_track_with_identity(
                            recognized_name, exclude_track_id=track_id
                        )
                        if existing_track is not None:
                            logger.warning(
                                f"RE-ID: Track {existing_track} merged into Track {track_id} "
                                f"(returning person: {recognized_name})"
                            )
                            self._merge_tracks(
                                source_track_id=existing_track,
                                target_track_id=track_id,
                            )
                            continue

                    self.identity_manager.update_identity(track_id, result)

            if self.identity_manager.is_identity_locked(track_id):
                locked = self.identity_manager.get_locked_identity(track_id)
                identity = locked["name"]
                identity_locked = True
                identity_confidence = locked["confidence"]

                if prev_state and not prev_state.identity_locked:
                    # First time identity is locked — check for duplicate tracks
                    existing_track = self._find_track_with_identity(
                        identity, exclude_track_id=track_id
                    )
                    if existing_track is not None:
                        logger.warning(
                            f"Track {track_id} merged into Track {existing_track} "
                            f"(same person: {identity})"
                        )
                        self._merge_tracks(
                            source_track_id=track_id, target_track_id=existing_track
                        )
                        continue

                    # Global ID reassignment
                    if self.global_track_manager and self.global_track_manager.enabled:
                        current_global_id = track.get("global_track_id")
                        existing_global = (
                            self.global_track_manager.find_global_track_by_identity(
                                identity
                            )
                        )
                        if (
                            existing_global is not None
                            and existing_global.global_id != current_global_id
                        ):
                            new_global_id = existing_global.global_id
                            self.global_track_manager.reassign_local_track(
                                camera_id=self.camera_id,
                                local_track_id=track_id,
                                new_global_id=new_global_id,
                            )
                            track["global_track_id"] = new_global_id
                            logger.info(
                                f"GLOBAL_ID_REASSIGN | identity='{identity}' "
                                f"old_global={current_global_id} -> new_global={new_global_id} "
                                f"camera={self.camera_id} local_track={track_id}"
                            )
                        elif existing_global is None and current_global_id is not None:
                            self.global_track_manager.update_global_track_identity(
                                current_global_id, identity, locked=True
                            )

                    logger.opt(colors=True).info(
                        f"<blue> Track {track_id} recognized as '{identity}' [{self.cam_type}]</blue>"
                    )

                    proof_image = self._get_best_person_image(track_id)
                    _ga = self.track_gender_age.get(track_id, {})
                    recognized_persons.append(
                        {
                            "track_id": track_id,
                            "global_track_id": track.get("global_track_id"),
                            "name": identity,
                            "recognized": True,
                            "confidence": identity_confidence,
                            "appear_time": prev_state.first_seen,
                            "camera_name": self.camera_name,
                            "camera_id": self.camera_id,
                            "status": self.cam_type,
                            "face_image": face_image,
                            "proof_image": proof_image,
                            "application": self.application,
                            "gender": _ga.get("gender"),
                            "age": _ga.get("age"),
                        }
                    )
                else:
                    # TIER 2: Consistency check for already-locked identities
                    if result.get("face_detected") and result.get("embedding") is not None:
                        emb = result["embedding"]
                        consistent, mismatch_distance = self.id_corrector.check_identity_consistency(
                            track_id, emb, identity, return_distance=True
                        )

                        # TIER 2b: DB re-verification — guards against tracker ID swaps.
                        # The track-internal consistency check only verifies that the
                        # current embedding looks like the embeddings *previously*
                        # attached to this track_id. If BoT-SORT swapped this track to
                        # a different person, the internal check will pass while the
                        # locked identity is now wrong. So independently ask the DB:
                        # who does this face look most like? If a *different* enrolled
                        # identity wins by a clear margin, treat this frame as a
                        # mismatch — the regular unlock-misses counter then triggers
                        # a re-lock on the right identity within a few frames.
                        db_says_other = False
                        if (
                            len(self.face_recognizer.db_embs) > 0
                            and result.get("recognized") is True
                            and result.get("name") is not None
                            and result.get("name") != identity
                            and result.get("similarity", 0.0) >= self.match_threshold
                        ):
                            db_says_other = True
                            logger.warning(
                                f"IDENTITY_DB_OVERRIDE | camera={self.camera_id} "
                                f"track={track_id} locked='{identity}' "
                                f"db_says='{result.get('name')}' "
                                f"sim={result.get('similarity', 0.0):.3f}"
                            )

                        if consistent and not db_says_other:
                            self._identity_mismatch_counts.pop(track_id, None)
                            self.id_corrector.add_embedding(track_id, emb, identity)
                        else:
                            misses = self._identity_mismatch_counts.get(track_id, 0) + 1
                            self._identity_mismatch_counts[track_id] = misses
                            dist_str = (
                                f"{mismatch_distance:.3f}"
                                if mismatch_distance is not None else "n/a"
                            )
                            logger.warning(
                                f"IDENTITY_UNSTABLE | camera={self.camera_id} "
                                f"track={track_id} locked='{identity}' "
                                f"dist={dist_str} (unlock>{self.id_unlock_threshold}) "
                                f"mismatch={misses}/{self.identity_unlock_misses}"
                            )
                            if misses >= self.identity_unlock_misses:
                                self.identity_manager.reset_track(track_id)
                                self.id_corrector.reset_track(track_id)
                                if prev_state:
                                    prev_state.identity = None
                                    prev_state.identity_locked = False
                                    prev_state.identity_confidence = 0.0
                                identity = None
                                identity_locked = False
                                identity_confidence = 0.0
                                self._identity_mismatch_counts.pop(track_id, None)
                                logger.warning(
                                    f"IDENTITY_UNLOCKED | camera={self.camera_id} "
                                    f"track={track_id} due to repeated face mismatch"
                                )
            else:
                identity = None
                identity_confidence = 0.0

            # Proof image crop for state manager (tight bbox for face recognition)
            proof_image = None
            if bbox is not None:
                x1, y1, x2, y2 = self._clip_bbox(frame, bbox)
                proof_image = frame[y1:y2, x1:x2]

            # Padded context crop for activity recognition (VLM needs surrounding context)
            activity_image = None
            if bbox is not None:
                fh, fw = frame.shape[:2]
                bx1, by1, bx2, by2 = bbox[:4]
                pw = (bx2 - bx1) * 0.45
                ph = (by2 - by1) * 0.35
                ax1 = max(0, int(bx1 - pw))
                ay1 = max(0, int(by1 - ph))
                ax2 = min(fw, int(bx2 + pw))
                ay2 = min(fh, int(by2 + ph * 0.5))
                activity_image = frame[ay1:ay2, ax1:ax2]

            self.state_manager.update_person(
                track_id=track_id,
                identity=identity,
                identity_locked=identity_locked,
                identity_confidence=identity_confidence,
                proof_image=proof_image,
            )

            if (
                self.action_recognizer
                and self.action_recognizer.enabled
                and activity_image is not None
            ):
                self._check_and_queue_action_recognition(
                    track_id=track_id,
                    identity=identity if identity_locked else None,
                    proof_image=activity_image,
                    frame_num=frame_num,
                    person_bbox=bbox,
                    original_frame=frame,
                )

            if face_image is not None:
                _max_crop_frames = 30
                crop_history = self.track_manager.track_crop_history.setdefault(track_id, {})
                crop_history[frame_num] = {"face": face_image, "bbox": bbox, "frame": frame.copy()}
                if len(crop_history) > _max_crop_frames:
                    for old_key in sorted(crop_history)[:-_max_crop_frames]:
                        del crop_history[old_key]

        # Removed tracks
        for track in removed_tracks:
            track_id = track["track_id"]
            state = self.state_manager.get_state(track_id)

            if state and not state.identity_locked:
                person_image = self._get_best_person_image(track_id)
                if person_image is not None and person_image.size > 0:
                    global_track_id = None
                    if self.global_track_manager and self.global_track_manager.enabled:
                        global_track_id = self.global_track_manager.get_global_id(
                            self.camera_id, track_id
                        )
                    _ga = self.track_gender_age.get(track_id, {})
                    recognized_persons.append(
                        {
                            "track_id": track_id,
                            "global_track_id": global_track_id,
                            "name": None,
                            "recognized": False,
                            "confidence": 0.0,
                            "appear_time": state.first_seen,
                            "camera_name": self.camera_name,
                            "camera_id": self.camera_id,
                            "status": self.cam_type,
                            "face_image": person_image,
                            "application": self.application,
                            "gender": _ga.get("gender"),
                            "age": _ga.get("age"),
                        }
                    )

            # on_track_removed is already called internally by PersonTracker.update()
            # so we do NOT call it here to avoid double-logging TRACK_INACTIVE.

            self.track_manager.remove_track(track_id)
            self.state_manager.remove_person(track_id)
            self.identity_manager.reset_track(track_id)
            self.id_corrector.reset_track(track_id)
            self.track_gender_age.pop(track_id, None)
            self._identity_mismatch_counts.pop(track_id, None)

        # Batch ID correction
        if self.id_corrector.should_run_correction():
            duplicates = self.id_corrector.find_duplicate_tracks()
            for track_id_1, track_id_2, distance in duplicates:
                if track_id_1 < track_id_2:
                    keep_track, merge_track = track_id_1, track_id_2
                else:
                    keep_track, merge_track = track_id_2, track_id_1
                if (
                    self.person_tracker.get_track_info(keep_track) is not None
                    and self.person_tracker.get_track_info(merge_track) is not None
                ):
                    logger.debug(
                        f"ID CORRECTION: Merging Track {merge_track} into Track {keep_track} "
                        f"(duplicate detected, face distance: {distance:.3f})"
                    )
                    self._merge_tracks(
                        source_track_id=merge_track, target_track_id=keep_track
                    )

        return recognized_persons

    def _update_gender_age(self, track_id: int, face_data: Dict) -> None:
        """Update gender/age for any tracked person with a quality face sample."""
        if not face_data.get("face_detected"):
            return

        det_score = float(face_data.get("det_score", 0.0) or 0.0)
        if det_score < self.gender_age_min_det_score:
            return

        face_w = int(face_data.get("face_width") or 0)
        face_h = int(face_data.get("face_height") or 0)
        if face_w < self.gender_age_min_face_size or face_h < self.gender_age_min_face_size:
            return

        face_image = face_data.get("face_image")
        if not self._face_crop_is_sharp(face_image, self.gender_age_blur_threshold):
            return

        gender = self._normalize_gender(face_data.get("gender"))
        age = self._normalize_age(face_data.get("age"))
        if gender is None and age is None:
            return

        current = self.track_gender_age.setdefault(
            track_id,
            {
                "gender": None,
                "age": None,
                "age_samples": [],
                "gender_samples": [],
                "det_score": 0.0,
            },
        )

        if age is not None:
            current["age_samples"].append(age)
            current["age_samples"] = current["age_samples"][-20:]
            current["age"] = int(round(median(current["age_samples"])))

        if gender is not None:
            current["gender_samples"].append(gender)
            current["gender_samples"] = current["gender_samples"][-30:]
            if len(current["gender_samples"]) >= 3:
                current["gender"] = Counter(current["gender_samples"]).most_common(1)[0][0]

        if det_score > current.get("det_score", 0.0):
            current["det_score"] = det_score

    @staticmethod
    def _normalize_gender(gender) -> Optional[str]:
        """Normalize InsightFace gender output to stable string labels."""
        if gender is None:
            return None
        if isinstance(gender, str):
            value = gender.strip().lower()
            if value in {"m", "male"}:
                return "male"
            if value in {"f", "female"}:
                return "female"
            return None
        if isinstance(gender, (int, float, np.integer, np.floating)):
            return "male" if int(gender) == 1 else "female"
        return None

    @staticmethod
    def _normalize_age(age) -> Optional[int]:
        """Reject impossible/unstable age outputs before smoothing."""
        if age is None:
            return None
        try:
            value = int(round(float(age)))
        except (TypeError, ValueError):
            return None
        if value < 5 or value > 90:
            return None
        return value

    @staticmethod
    def _face_crop_is_sharp(face_image: Optional[np.ndarray], threshold: float) -> bool:
        if face_image is None or face_image.size == 0:
            return False
        gray = (
            cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            if len(face_image.shape) == 3 else face_image
        )
        return cv2.Laplacian(gray, cv2.CV_64F).var() >= threshold

    def _compute_recognition_result(
        self,
        embedding: Optional[np.ndarray],
        face_image: Optional[np.ndarray],
        face_detected: bool,
        det_score: float = 0.0,
        face_width: int = 0,
        face_height: int = 0,
        track_id: int = 0,
    ) -> Dict:
        """Build a recognition-result dict from a pre-computed embedding.

        Matches the format returned by _recognize_face() so that
        finalize_identities can use it unchanged.
        """
        if not face_detected or embedding is None:
            if self.global_track_manager:
                self.global_track_manager.on_face_not_visible(
                    camera_id=self.camera_id, local_track_id=track_id
                )
            return {"face_detected": face_detected, "name": None, "similarity": 0.0}

        if not self._identity_face_is_usable(face_image, det_score, face_width, face_height):
            if self.global_track_manager:
                self.global_track_manager.on_face_detected(
                    camera_id=self.camera_id,
                    local_track_id=track_id,
                    quality=det_score,
                    recognized=False,
                    identity=None,
                )
            return {
                "face_detected": True,
                "recognized": False,
                "name": None,
                "similarity": 0.0,
                "embedding": embedding,
                "face_image": face_image,
            }

        if len(self.face_recognizer.db_embs) == 0:
            if self.global_track_manager:
                self.global_track_manager.on_face_detected(
                    camera_id=self.camera_id,
                    local_track_id=track_id,
                    quality=det_score,
                    recognized=False,
                    identity=None,
                )
            return {
                "face_detected": True,
                "recognized": False,
                "name": None,
                "similarity": 0.0,
                "embedding": embedding,
                "face_image": face_image,
            }

        similarities = self.face_recognizer.compute_similarities(
            np.array([embedding])
        )
        best_idx, best_similarity, margin = self.face_recognizer.get_best_match(similarities)

        if best_similarity >= self.match_threshold and margin >= self.match_margin:
            name = self.face_recognizer.db_names[best_idx]
            if self.global_track_manager:
                self.global_track_manager.on_face_detected(
                    camera_id=self.camera_id,
                    local_track_id=track_id,
                    quality=det_score,
                    recognized=True,
                    identity=name,
                )
            return {
                "face_detected": True,
                "recognized": True,
                "name": name,
                "similarity": best_similarity,
                "embedding": embedding,
                "face_image": face_image,
            }
        else:
            if self.global_track_manager:
                self.global_track_manager.on_face_detected(
                    camera_id=self.camera_id,
                    local_track_id=track_id,
                    quality=det_score,
                    recognized=False,
                    identity=None,
                )
            return {
                "face_detected": True,
                "recognized": False,
                "name": None,
                "similarity": best_similarity,
                "embedding": embedding,
                "face_image": face_image,
            }

    def _identity_face_is_usable(
        self,
        face_image: Optional[np.ndarray],
        det_score: float,
        face_width: int,
        face_height: int,
    ) -> bool:
        """Gate identity matching harder than gender/age display."""
        if det_score < self.identity_min_det_score:
            return False
        if face_width < self.min_face_size or face_height < self.min_face_size:
            return False
        return self._face_crop_is_sharp(face_image, self.blur_threshold)

    @staticmethod
    def _clip_bbox(frame: np.ndarray, bbox) -> Tuple[int, int, int, int]:
        """Clamp a bbox to frame boundaries."""
        x1, y1, x2, y2 = map(int, bbox)
        return (
            max(0, x1), max(0, y1),
            min(frame.shape[1], x2), min(frame.shape[0], y2),
        )

    @staticmethod
    def _read_crop_image(crop_data) -> Optional[np.ndarray]:
        """Extract an image from a crop_data entry (dict or raw array)."""
        if isinstance(crop_data, dict):
            f = crop_data.get('frame')
            b = crop_data.get('bbox')
            if f is not None and b is not None:
                x1, y1, x2, y2 = map(int, b)
                return f[y1:y2, x1:x2]
            return crop_data.get('face')
        return crop_data

    def _get_best_person_image(self, track_id: int) -> Optional[np.ndarray]:
        """Get the best quality person image from track history."""
        crops = self.track_manager.track_crop_history.get(track_id, {})
        if not crops:
            return None

        min_face_size = self.min_face_size
        blur_threshold = self.blur_threshold

        best_frame_num = None
        best_score = -1

        for frame_num, crop_data in crops.items():
            face_crop = crop_data.get('face') if isinstance(crop_data, dict) else crop_data
            if face_crop is None or face_crop.size == 0:
                continue

            h, w = face_crop.shape[:2]
            if h < min_face_size or w < min_face_size:
                continue

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < blur_threshold:
                continue

            size_score = (h * w) / (min_face_size ** 2)
            sharpness_score = laplacian_var / blur_threshold
            total_score = size_score * 0.6 + sharpness_score * 0.4

            if total_score > best_score:
                best_score = total_score
                best_frame_num = frame_num

        key = best_frame_num if best_frame_num is not None else (max(crops.keys()) if crops else None)
        if key is not None:
            return self._read_crop_image(crops[key])

        return None

    def _find_track_with_identity(
        self, identity_name: str, exclude_track_id: Optional[int] = None
    ) -> Optional[int]:
        """Find an existing active track with the given locked identity."""
        for state in self.state_manager.get_all_states():
            if exclude_track_id is not None and state.track_id == exclude_track_id:
                continue
            if state.identity_locked and state.identity == identity_name:
                if self.person_tracker.get_track_info(state.track_id) is not None:
                    return state.track_id
        return None

    def _merge_tracks(self, source_track_id: int, target_track_id: int) -> None:
        """Merge source track into target track, transferring all data."""
        source_data = self.track_manager.get_track_data(source_track_id)
        if source_data:
            target_data = self.track_manager.get_track_data(target_track_id)
            if target_data:
                if 'detections' in source_data and 'detections' in target_data:
                    target_data['detections'].extend(source_data['detections'])
                if 'keypoints' in source_data and 'keypoints' in target_data:
                    source_kp = source_data['keypoints']
                    target_kp = target_data['keypoints']
                    if isinstance(source_kp, dict) and isinstance(target_kp, dict):
                        target_kp.update(source_kp)
                    elif isinstance(source_kp, list) and isinstance(target_kp, list):
                        target_kp.extend(source_kp)

        if source_track_id in self.track_manager.track_crop_history:
            source_crops = self.track_manager.track_crop_history[source_track_id]
            if target_track_id not in self.track_manager.track_crop_history:
                self.track_manager.track_crop_history[target_track_id] = {}
            self.track_manager.track_crop_history[target_track_id].update(source_crops)

        self.track_manager.remove_track(source_track_id)
        self.state_manager.remove_person(source_track_id)
        self.identity_manager.reset_track(source_track_id)
        self.id_corrector.reset_track(source_track_id)
        self._identity_mismatch_counts.pop(source_track_id, None)

        if source_track_id in self.person_tracker.active_tracks:
            del self.person_tracker.active_tracks[source_track_id]

    def _check_and_queue_action_recognition(
        self,
        track_id: int,
        identity: str,
        proof_image: Optional[np.ndarray],
        frame_num: int,
        person_bbox=None,
        original_frame=None,
    ) -> None:
        """Event-driven action recognition queueing.

        Two paths share the same VLM queue, ordered by priority:

          • Phone path (HIGH): cheap YOLO pre-filter runs every
            ``phone_precheck_interval_seconds``. If a phone-shaped object is
            detected in the face/hands region, enqueue a VLM request immediately.
            Most frames produce no detection → no GPU work.

          • General path (LOW): periodic activity sampling at
            ``general_poll_interval_seconds`` (much sparser). Only locked
            identities are polled; unidentified tracks burn no GPU on this path.

        This collapses VLM load from "every track every 8s" to "only when YOLO
        sees something worth classifying," matching Gemma-3n throughput.
        """
        if proof_image is None or proof_image.size == 0:
            return

        current_time = time.time()
        phone_interval = getattr(
            self.action_recognizer, 'phone_precheck_interval_seconds', 2.0
        )
        general_interval = getattr(
            self.action_recognizer, 'general_poll_interval_seconds', 30.0
        )
        unknown_interval = getattr(
            self.action_recognizer, 'unknown_poll_interval_seconds', 45.0
        )

        last_phone = self.last_phone_check_per_track.get(track_id, 0.0)
        last_general = self.last_action_check_per_identity.get(track_id, 0.0)
        do_phone_check = (current_time - last_phone) >= phone_interval
        # Poll EVERYONE for general activity (not just locked identities) so
        # workshop monitoring shows activity labels on visitors/unenrolled
        # workers too. Unrecognized polls use a longer cadence and lower
        # priority so locked identities still take precedence under load.
        active_general_interval = general_interval if identity is not None else unknown_interval
        do_general_poll = (current_time - last_general) >= active_general_interval

        if not do_phone_check and not do_general_poll:
            return

        # Prune stale entries to prevent unbounded growth
        if len(self.last_action_check_per_identity) > self._max_action_identity_cache:
            cutoff = current_time - general_interval * 2
            stale = [k for k, t in self.last_action_check_per_identity.items() if t < cutoff]
            for k in stale:
                del self.last_action_check_per_identity[k]
                self.last_phone_check_per_track.pop(k, None)

        user_id = self.name_to_id_map.get(identity) if identity else None

        # Native-resolution focus regions — used by both YOLO pre-filter and VLM.
        hand_region = None
        phone_regions: List[Dict] = []
        face_hands_crop = None
        if person_bbox is not None and original_frame is not None:
            try:
                bx1, by1, bx2, by2 = person_bbox[:4]
                fh, fw = original_frame.shape[:2]
                bw = bx2 - bx1
                bh = by2 - by1
                px = bw * 0.30

                def _crop_region(name: str, y_start: float, y_end: float):
                    rx1 = max(0, int(bx1 - px))
                    rx2 = min(fw, int(bx2 + px))
                    ry1 = max(0, int(by1 + bh * y_start))
                    ry2 = min(fh, int(by1 + bh * y_end))
                    if ry2 <= ry1 or rx2 <= rx1:
                        return None
                    crop = original_frame[ry1:ry2, rx1:rx2].copy()
                    phone_regions.append({'name': name, 'image': crop})
                    return crop

                face_hands_crop = _crop_region('face_hands', 0.00, 0.68)
                hand_region = _crop_region('hand', 0.28, 0.96)
                _crop_region('upper_body', 0.00, 0.90)
            except Exception:
                hand_region = None
                phone_regions = []
                face_hands_crop = None

        # ── YOLO phone pre-filter (cheap, ~20–50 ms on small crop) ──────────
        precomputed_phone_info = None
        phone_hit = False
        phone_detector = getattr(self.action_recognizer, 'phone_detector', None)
        if do_phone_check and phone_detector and phone_detector.available:
            self.last_phone_check_per_track[track_id] = current_time
            try:
                target = face_hands_crop if face_hands_crop is not None else proof_image
                precomputed_phone_info = phone_detector.detect(target)
                phone_hit = bool(precomputed_phone_info.get('phone_detected'))
            except Exception as e:
                logger.debug(f"phone pre-filter error track={track_id}: {e}")
                precomputed_phone_info = None
                phone_hit = False

        # Decide whether to enqueue and at what priority.
        if phone_hit:
            priority = PRIORITY_PHONE
            self.last_action_check_per_identity[track_id] = current_time
        elif do_general_poll:
            priority = PRIORITY_GENERAL if identity is not None else PRIORITY_UNKNOWN
            self.last_action_check_per_identity[track_id] = current_time
        else:
            return  # nothing worth a VLM call right now

        request_id = f"cam{self.camera_id}_track{track_id}_frame{frame_num}"

        def action_result_callback(result: Dict):
            self._handle_action_result(
                track_id=track_id,
                user_id=user_id,
                identity=identity,
                result=result,
                timestamp=current_time,
                proof_image=proof_image,
            )

        queued = self.action_recognizer.recognize_async(
            image=proof_image,
            request_id=request_id,
            callback=action_result_callback,
            priority=priority,
            metadata={
                'track_id': track_id,
                'identity': identity,
                'user_id': user_id,
                'user_name': identity,
                'camera_id': self.camera_id,
                'frame_num': frame_num,
                'hand_region': hand_region,
                'phone_regions': phone_regions,
                'precomputed_phone_info': precomputed_phone_info,
            },
        )

        if not queued:
            now = time.time()
            if now - self._last_queue_full_warn >= 30.0:
                logger.warning(
                    f"Failed to queue action recognition (queue full, priority={priority})"
                )
                self._last_queue_full_warn = now

    def _handle_action_result(
        self,
        track_id: int,
        user_id: int,
        identity: str,
        result: Dict,
        timestamp: float,
        proof_image: np.ndarray,
    ) -> None:
        """Handle action recognition result."""
        action = result.get('action')
        if not action:
            return

        state = self.state_manager.get_state(track_id)
        if not state:
            logger.warning(
                f"ACTION_STALE_RESULT_DROP | camera={self.camera_id} "
                f"track={track_id} action={action} reason=track_missing"
            )
            return

        completed_at = float(result.get('completed_at') or time.time())
        result_age = completed_at - timestamp
        if result_age > self.action_result_max_age_seconds:
            logger.warning(
                f"ACTION_STALE_RESULT_DROP | camera={self.camera_id} "
                f"track={track_id} action={action} age={result_age:.2f}s "
                f"> {self.action_result_max_age_seconds:.2f}s"
            )
            return

        display_identity = identity or (state.identity if state else None) or 'unknown'
        logger.info(
            f"ACTION DETECTED | {display_identity}: {action} | "
            f"time={result.get('inference_time', 0.0):.3f}s | "
            f"age={result_age:.2f}s | camera={self.camera_name}"
        )

        state.last_detected_action = action
        state.last_action_time = time.time()

    def reset(self) -> None:
        """Reset all tracking state."""
        self.person_tracker.reset()
        self.track_manager.reset()
        self.identity_manager.reset()
        self.state_manager.reset()
        self.last_action_check_per_identity.clear()
        self.last_phone_check_per_track.clear()
        self.track_gender_age.clear()
        self._identity_mismatch_counts.clear()
        self.frame_count = 0
