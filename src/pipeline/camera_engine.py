"""CameraEngine - Single camera stream processing engine.

This module handles per-camera processing including:
- Person detection and tracking
- Face recognition within person ROIs
- Identity management and voting
- Track merging and ID correction
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# Detection, tracking and recognition models
from lum_vision import (
    FaceDetector,
    FaceMatcher,
    GlobalTrackIDGenerator,
    GlobalTrackManager,
    IdentityManager,
    IDSwitchCorrector,
    PersonDetector,
    PersonStateManager,
    PersonTracker,
    PersonTrackManager,
    crop_person_roi,
)


class CameraEngine:
    """Engine for processing a single camera stream.

    Each camera has its own tracking and state management components.
    Detection models are shared across cameras for GPU efficiency.
    """

    def __init__(
        self,
        camera_config: Dict[str, Any],
        face_detector: FaceDetector,
        face_recognizer: FaceMatcher,
        person_detector: PersonDetector,
        client_slug: str,
        global_id_generator: Optional[GlobalTrackIDGenerator] = None,
        name_to_id_map: Optional[Dict[str, int]] = None,
        global_track_manager: Optional[GlobalTrackManager] = None,
        action_recognizer: Optional[Any] = None,
        homography_registry: Optional[Any] = None,
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
        self.min_face_size = camera_config.get('min_face_size', 150)  # Minimum face size for quality check
        # Unrecognized-case gate thresholds (LSO-7); injected from config.yaml by
        # the engine so _best_face_signals prefers frames that clear both.
        self.unrecognized_frontality_min = camera_config.get('unrecognized_frontality_min', 0.6)
        self.unrecognized_pitch_min = camera_config.get('unrecognized_pitch_min', 0.4)
        self.roi = camera_config.get('roi')
        self.line_points = camera_config.get('line_points')

        # Shared components (models)
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.person_detector = person_detector
        self.action_recognizer = action_recognizer
        self.client_slug = client_slug
        self.global_id_generator = global_id_generator
        self.global_track_manager = global_track_manager

        # Name mapping for activity tracking
        self.name_to_id_map = name_to_id_map or {}

        # Homography registry + per-track 5Hz throttle for real-time position emit
        self.homography_registry = homography_registry
        self._position_last_emit: Dict[int, float] = {}
        from messaging.publisher import MDAPublisher
        self._publisher = MDAPublisher(client_slug)

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
            identity_lock_frames=5,
            identity_consensus=0.60,
            min_window_duration_ms=333,
            similarity_threshold=self.match_threshold
        )

        # ID Switch Corrector (face-based correction)
        self.id_corrector = IDSwitchCorrector(
            embedding_distance_threshold=0.4,  # Cosine distance for "same person"
            correction_interval_frames=5,  # Check every 5 frames to avoid log spam
            min_embedding_samples=3  # Need 3+ embeddings for reliable comparison
        )

        # State Manager
        self.state_manager = PersonStateManager(
            camera_id=self.camera_id,
            name_to_id_map=self.name_to_id_map
        )

        # Action recognition throttling lives on the shared ActionRecognitionWorker,
        # not here — otherwise one person on N cameras is classified N times.

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
                track_id=track_id,
            )

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
                        }
                    )
                else:
                    # TIER 2: Consistency check for already-locked identities
                    if result.get("face_detected") and result.get("embedding") is not None:
                        emb = result["embedding"]
                        self.id_corrector.add_embedding(track_id, emb, identity)
                        self.id_corrector.check_identity_consistency(
                            track_id, emb, identity
                        )
            else:
                voting = self.identity_manager.get_voting_status(track_id)
                if voting and voting.get("top_candidate"):
                    identity = voting["top_candidate"]
                    identity_confidence = voting.get("top_avg_similarity", 0.0)

            # Proof image crop for state manager. Copied, not a view: this crop
            # outlives the frame (state manager, action queue) and a view would
            # pin the whole frame alive behind it.
            proof_image = None
            if bbox is not None:
                x1, y1, x2, y2 = self._clip_bbox(frame, bbox)
                if x2 > x1 and y2 > y1:
                    proof_image = frame[y1:y2, x1:x2].copy()

            self.state_manager.update_person(
                track_id=track_id,
                identity=identity,
                identity_locked=identity_locked,
                identity_confidence=identity_confidence,
                proof_image=proof_image,
            )

            if (
                identity_locked
                and self.action_recognizer
                and self.action_recognizer.enabled
                and "activity" in self.application
            ):
                self._check_and_queue_action_recognition(
                    track_id=track_id,
                    identity=identity,
                    proof_image=proof_image,
                    frame_num=frame_num,
                )

            if face_image is not None:
                _max_crop_frames = 30
                crop_history = self.track_manager.track_crop_history.setdefault(track_id, {})
                crop_history[frame_num] = {
                    "face": face_image, "bbox": bbox, "frame": frame.copy(),
                    # For the unrecognized-case frontality/pitch gate (LSO-7).
                    "landmarks": face_data.get("face_landmarks"),
                    "det_score": face_data.get("det_score", 0.0),
                    "frontality": face_data.get("frontality"),
                    "pitch": face_data.get("pitch"),
                }
                if len(crop_history) > _max_crop_frames:
                    for old_key in sorted(crop_history)[:-_max_crop_frames]:
                        del crop_history[old_key]

        # Removed tracks
        for track in removed_tracks:
            track_id = track["track_id"]
            state = self.state_manager.get_state(track_id)

            if state and not state.identity_locked:
                # Orientation gate signals for this unrecognized track (LSO-7).
                face_frontality, face_det_score, face_pitch, best_crop = (
                    self._best_face_signals(track_id)
                )
                # Card image = the gate frame, so it matches the face that
                # passed the gate (not _get_best_person_image's last-frame fallback).
                person_image = self._unrecognized_card_image(best_crop, track_id)
                if person_image is not None and person_image.size > 0:
                    global_track_id = None
                    if self.global_track_manager and self.global_track_manager.enabled:
                        global_track_id = self.global_track_manager.get_global_id(
                            self.camera_id, track_id
                        )
                    recognized_persons.append(
                        {
                            "track_id": track_id,
                            "global_track_id": global_track_id,
                            "name": None,
                            "recognized": False,
                            "confidence": 0.0,
                            "face_frontality": face_frontality,
                            "face_pitch": face_pitch,
                            "face_det_score": face_det_score,
                            "appear_time": state.first_seen,
                            "camera_name": self.camera_name,
                            "camera_id": self.camera_id,
                            "status": self.cam_type,
                            "face_image": person_image,
                            "application": self.application,
                        }
                    )

            # on_track_removed is already called internally by PersonTracker.update()
            # so we do NOT call it here to avoid double-logging TRACK_INACTIVE.

            self.track_manager.remove_track(track_id)
            self.state_manager.remove_person(track_id)
            self.identity_manager.reset_track(track_id)
            self.id_corrector.reset_track(track_id)

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

    def emit_positions(self, active_tracks: List[Dict]) -> None:
        """Publish floor-projected positions for active tracks (~5Hz per track).

        Per-frame call. Skips silently if no homography is cached for this camera
        (HomographyRegistry caches negative lookups and logs once).
        """
        if self.homography_registry is None or not active_tracks:
            return

        cached = self.homography_registry.get(self.client_slug, self.camera_id)
        if cached is None:
            return
        H, map_id = cached

        now = time.monotonic()
        cooldown = 0.2  # 5 Hz

        for track in active_tracks:
            track_id = track.get("track_id")
            bbox = track.get("bbox")
            if track_id is None or bbox is None:
                continue
            last = self._position_last_emit.get(track_id, 0.0)
            if now - last < cooldown:
                continue

            x1, y1, x2, y2 = bbox
            foot = np.array(
                [[[(float(x1) + float(x2)) / 2.0, float(y2)]]], dtype=np.float64
            )
            projected = cv2.perspectiveTransform(foot, H).reshape(2)

            user_id: Optional[int] = None
            user_name: Optional[str] = None
            if self.identity_manager.is_identity_locked(track_id):
                locked = self.identity_manager.get_locked_identity(track_id)
                if locked:
                    user_name = locked.get("name")
                    if user_name is not None:
                        user_id = self.name_to_id_map.get(user_name)

            try:
                self._publisher.publish_user_location_updated_position(
                    camera_id=self.camera_id,
                    map_id=map_id,
                    track_id=int(track_id),
                    user_id=user_id,
                    user_name=user_name,
                    x=float(projected[0]),
                    y=float(projected[1]),
                )
                self._position_last_emit[track_id] = now
            except Exception as e:
                logger.exception(
                    f"emit_positions publish failed for camera {self.camera_id} "
                    f"track {track_id}: {e}"
                )

    def _compute_recognition_result(
        self,
        embedding: Optional[np.ndarray],
        face_image: Optional[np.ndarray],
        face_detected: bool,
        det_score: float = 0.0,
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
        best_idx, best_similarity = self.face_recognizer.get_best_match(similarities)

        if best_similarity >= self.match_threshold:
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

        min_face_size = getattr(self, 'min_face_size', 150)
        blur_threshold = 100.0

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

    def _best_face_signals(self, track_id: int) -> Tuple[float, float, float, Optional[dict]]:
        """Signals for the best gate-passing frame of an unrecognized track.

        Returns (frontality, det_score, pitch, crop_data). See _pick_best_signals
        for the selection rule; frontality/pitch come from the lum-model-vision
        package (stored in crop_history at capture).
        """
        crops = self.track_manager.track_crop_history.get(track_id, {})
        return self._pick_best_signals(
            crops, self.unrecognized_frontality_min, self.unrecognized_pitch_min
        )

    @staticmethod
    def _pick_best_signals(
        crops: Dict, fr_min: float, pi_min: float
    ) -> Tuple[float, float, float, Optional[dict]]:
        """Pick the highest yaw×pitch frame that clears BOTH thresholds; only if
        none do, return the best-effort frame (which the gate then drops).

        Picking by product alone can return a frame that fails one threshold
        while another frame passes both — silently dropping a valid case. Returns
        (0.0, 0.0, 0.0, None) when no frame has usable orientation.
        """
        best_pass = (0.0, 0.0, 0.0, None)
        best_any = (0.0, 0.0, 0.0, None)
        best_pass_score = -1.0
        best_any_score = -1.0
        for crop_data in crops.values():
            if not isinstance(crop_data, dict):
                continue
            fr = crop_data.get("frontality")
            pi = crop_data.get("pitch")
            if fr is None or pi is None:
                continue
            score = fr * pi
            cand = (fr, float(crop_data.get("det_score", 0.0) or 0.0), pi, crop_data)
            if score > best_any_score:
                best_any_score = score
                best_any = cand
            if fr >= fr_min and pi >= pi_min and score > best_pass_score:
                best_pass_score = score
                best_pass = cand
        return best_pass if best_pass[3] is not None else best_any

    def _unrecognized_card_image(self, best_crop: Optional[dict], track_id: int) -> Optional[np.ndarray]:
        """Card image for an UNRECOGNIZED case: the person ROI of the gate frame
        (best yaw×pitch), so the card matches the frontal face that passed the
        gate. Returns None if there's no usable gate frame — the case is then
        skipped rather than falling back to _get_best_person_image's quality-blind
        last frame (which stays for the recognized/attendance path). A None
        best_crop means no orientation was scored, so the gate drops it anyway.
        """
        if best_crop is not None:
            frame, bbox = best_crop.get("frame"), best_crop.get("bbox")
            if frame is not None and bbox is not None:
                roi, _ = crop_person_roi(frame, np.asarray(bbox, dtype=float), expand=0.1)
                if roi is not None and roi.size:
                    return roi
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

        if source_track_id in self.person_tracker.active_tracks:
            del self.person_tracker.active_tracks[source_track_id]

    def _check_and_queue_action_recognition(
        self,
        track_id: int,
        identity: str,
        proof_image: Optional[np.ndarray],
        frame_num: int,
    ) -> None:
        """Check if action recognition is needed and queue request."""
        if proof_image is None or proof_image.size == 0:
            return

        # Engine-wide reservation: at most one inference per identity per
        # interval, however many cameras can see them right now.
        if not self.action_recognizer.reserve_check(identity):
            return

        user_id = self.name_to_id_map.get(identity)
        if user_id is None:
            logger.warning(f"Cannot find user_id for '{identity}', skipping action recognition")
            self.action_recognizer.cancel_check(identity)
            return

        request_id = f"cam{self.camera_id}_track{track_id}_frame{frame_num}"

        def action_result_callback(result: Dict):
            self._handle_action_result(
                track_id=track_id,
                identity=identity,
                result=result,
            )

        queued = self.action_recognizer.recognize_async(
            image=proof_image,
            request_id=request_id,
            callback=action_result_callback,
            metadata={
                'track_id': track_id,
                'identity': identity,
                'user_id': user_id,
                'camera_id': self.camera_id,
                'frame_num': frame_num,
            },
        )

        if not queued:
            # Nothing ran, so release the reservation rather than making this
            # person wait out a full interval.
            self.action_recognizer.cancel_check(identity)
            logger.warning("Failed to queue action recognition (queue full)")

    def _handle_action_result(
        self,
        track_id: int,
        identity: str,
        result: Dict,
    ) -> None:
        """Handle action recognition result."""
        action = result.get('action')
        if not action:
            return

        logger.info(
            f"ACTION DETECTED | {identity}: {action} | "
            f"time={result.get('inference_time', 0.0):.3f}s | camera={self.camera_name}"
        )

        state = self.state_manager.get_state(track_id)
        if state:
            state.last_detected_action = action

    def reset(self) -> None:
        """Reset all tracking state."""
        self.person_tracker.reset()
        self.track_manager.reset()
        self.identity_manager.reset()
        self.state_manager.reset()
        # The action throttle is engine-wide and deliberately NOT cleared here —
        # one camera resetting must not license a duplicate inference elsewhere.
        self.frame_count = 0

