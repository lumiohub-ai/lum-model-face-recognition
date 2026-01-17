"""
Global Track Manager - Phase 1 Implementation.

Cross-camera global track management using Body ReID.

Philosophy:
- Prefer duplicate global IDs over incorrect merges
- Conservative matching with hard rejection rules
- Body ReID as primary signal (face often not visible)
- Feature flag for safe rollout
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import time

import numpy as np
import torch
import cv2
import yaml
from loguru import logger

from .global_track import (
    GlobalTrack,
    EmbeddingQuality,
    CameraTrackInfo,
    CachedEmbedding
)


class GlobalTrackManager:
    """
    Assigns global IDs across cameras using Body ReID (post-processing layer).

    Phase 1: Conservative body-only matching with hard rejection rules.
    """

    def __init__(self, config_path: Optional[str] = None, app_config: Optional[Dict[str, Any]] = None):
        """
        Initialize GlobalTrackManager.

        Args:
            config_path: Path to global_tracking.yaml config file
            app_config: Optional app config dict (from config.yaml)
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Feature flag priority:
        # 1. app_config (config.yaml) - enable_global_tracking
        # 2. global_tracking.yaml - enabled
        # 3. Environment variable ENABLE_GLOBAL_TRACKING (backward compatibility)
        if app_config and 'enable_global_tracking' in app_config:
            self.enabled = app_config.get('enable_global_tracking', True)
        elif self.config.get('enabled') is not None:
            self.enabled = self.config.get('enabled', False)
        else:
            self.enabled = os.getenv('ENABLE_GLOBAL_TRACKING', 'true').lower() == 'true'

        # Global tracks
        self.global_tracks: Dict[int, GlobalTrack] = {}

        # Mapping: camera_id -> {local_track_id: global_track_id}
        self.local_to_global: Dict[int, Dict[int, int]] = defaultdict(dict)

        # ID generator (start at 1000 to distinguish from local IDs)
        self.global_id_counter = 1000

        # Embedding cache for interval-based extraction
        self.embedding_cache: Dict[Tuple[int, int], CachedEmbedding] = {}

        # ReID model (lazy initialization)
        self._body_reid_model = None
        self._device = None

        # Configuration values
        self._init_config_values()

        # Metrics for monitoring
        self.metrics = GlobalTrackingMetrics()

        # Baseline metrics (Phase 0 compatibility)
        self.track_stats: Dict[tuple, Dict[str, Any]] = {}
        self.total_tracks_created = 0
        self.total_tracks_removed = 0
        self.total_faces_detected = 0
        self.total_faces_not_visible = 0

        # Last validation time
        self._last_validation_time = time.time()

        if self.enabled:
            logger.info("GlobalTrackManager initialized (PHASE 1 - BODY ReID)")
            logger.info(f"Config: threshold={self.similarity_threshold}, "
                       f"temporal_window={self.temporal_window_sec}s")
        else:
            logger.debug("GlobalTrackManager disabled (enable_global_tracking=false)")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try default paths
            possible_paths = [
                Path("configs/global_tracking.yaml"),
                Path(__file__).parent.parent.parent.parent / "configs" / "global_tracking.yaml",
            ]
            for p in possible_paths:
                if p.exists():
                    config_path = str(p)
                    break

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('global_tracking', {})

        # Return defaults if no config found
        return {}

    def _init_config_values(self) -> None:
        """Initialize configuration values with defaults."""
        # Matching thresholds
        matching_cfg = self.config.get('matching', {})
        self.similarity_threshold = matching_cfg.get('threshold', 0.70)
        self.temporal_window_sec = matching_cfg.get('temporal_window_sec', 60)
        self.min_reentry_gap_sec = matching_cfg.get('min_reentry_gap_sec', 10)
        self.min_quality = matching_cfg.get('min_quality', 0.5)
        self.removal_window_sec = matching_cfg.get('removal_window_sec', 300)

        # Crop quality requirements
        crop_cfg = self.config.get('crop_quality', {})
        self.min_crop_height = crop_cfg.get('min_height', 80)
        self.min_crop_width = crop_cfg.get('min_width', 40)
        self.min_crop_area = crop_cfg.get('min_area', 3200)
        self.min_aspect_ratio = crop_cfg.get('min_aspect_ratio', 1.5)
        self.max_aspect_ratio = crop_cfg.get('max_aspect_ratio', 4.0)

        # Embedding management
        emb_cfg = self.config.get('embeddings', {})
        self.top_k_size = emb_cfg.get('top_k_size', 5)
        self.prototype_alpha = emb_cfg.get('prototype_alpha', 0.2)
        self.extract_interval_frames = emb_cfg.get('extract_interval_frames', 10)

        # Safety
        safety_cfg = self.config.get('safety', {})
        self.validation_interval_sec = safety_cfg.get('validation_interval_sec', 30)
        self.archive_after_inactive_min = safety_cfg.get('archive_after_inactive_min', 10)
        self.overlapping_camera_groups: List[List[int]] = safety_cfg.get(
            'overlapping_cameras', []
        )

        # Body ReID model config
        reid_cfg = self.config.get('body_reid', {})
        self.reid_model_name = reid_cfg.get('model', 'osnet_x0_25_msmt17')
        self.reid_weights_path = reid_cfg.get(
            'weights_path', 'volumes/models/weights/osnet_x0_25_msmt17.pt'
        )
        self.reid_device = reid_cfg.get('device', 'cuda:0')
        self.reid_half_precision = reid_cfg.get('half_precision', False)

    @property
    def body_reid_model(self):
        """Lazy initialization of body ReID model."""
        if self._body_reid_model is None:
            self._init_body_reid_model()
        return self._body_reid_model

    def _download_reid_weights(self, weights_path: Path) -> bool:
        """
        Download ReID weights if not present.

        Args:
            weights_path: Path to save the weights

        Returns:
            True if weights are available, False otherwise
        """
        if weights_path.exists():
            return True

        # Create parent directory if needed
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        # OSNet model URLs (multiple sources for redundancy)
        model_urls = {
            'osnet_x0_25_msmt17': [
                # Google Drive (primary)
                ('gdown', '1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e'),
                # Direct URL fallback (if available)
                ('url', 'https://github.com/mikel-brostrom/yolo_tracking/releases/download/v10.0.0/osnet_x0_25_msmt17.pt'),
            ],
            'osnet_x1_0_msmt17': [
                ('gdown', '1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x'),
            ],
        }

        model_name = self.reid_model_name
        if model_name not in model_urls:
            logger.error(f"Unknown ReID model: {model_name}")
            return False

        logger.info(f"Downloading ReID weights for {model_name}...")

        for source_type, source_id in model_urls[model_name]:
            try:
                if source_type == 'gdown':
                    # Try gdown first (Google Drive)
                    try:
                        import gdown
                        url = f'https://drive.google.com/uc?id={source_id}'
                        logger.info(f"Downloading from Google Drive: {source_id}")
                        gdown.download(url, str(weights_path), quiet=False)
                        if weights_path.exists():
                            logger.info(f"ReID weights downloaded to {weights_path}")
                            return True
                    except ImportError:
                        logger.debug("gdown not available, trying alternative...")
                        continue
                    except Exception as e:
                        logger.warning(f"gdown download failed: {e}")
                        continue

                elif source_type == 'url':
                    # Direct URL download
                    import urllib.request
                    logger.info(f"Downloading from URL: {source_id}")
                    urllib.request.urlretrieve(source_id, str(weights_path))
                    if weights_path.exists():
                        logger.info(f"ReID weights downloaded to {weights_path}")
                        return True

            except Exception as e:
                logger.warning(f"Download failed from {source_type}: {e}")
                continue

        logger.error(f"Failed to download ReID weights for {model_name}")
        return False

    def _init_body_reid_model(self) -> None:
        """Initialize OSNet ReID model."""
        try:
            from boxmot.appearance.reid_auto_backend import ReidAutoBackend

            weights_path = Path(self.reid_weights_path)

            # Auto-download weights if not present
            if not weights_path.exists():
                logger.warning(f"ReID weights not found at {weights_path}. Attempting download...")
                if not self._download_reid_weights(weights_path):
                    raise FileNotFoundError(
                        f"Could not download ReID weights. Please manually download "
                        f"{self.reid_model_name} weights to {weights_path}"
                    )

            # Determine device
            if torch.cuda.is_available() and 'cuda' in self.reid_device:
                self._device = torch.device(self.reid_device)
            else:
                self._device = torch.device('cpu')
                logger.warning("CUDA not available, using CPU for ReID")

            backend = ReidAutoBackend(
                weights=weights_path,
                device=self._device,
                half=self.reid_half_precision
            )

            self._body_reid_model = backend.get_backend()
            logger.info(f"Body ReID model loaded: {self.reid_model_name} on {self._device}")

        except Exception as e:
            logger.error(f"Failed to initialize body ReID model: {e}")
            self._body_reid_model = None
            raise

    def assign_global_id(
        self,
        camera_id: int,
        local_track_id: int,
        person_crop: Optional[np.ndarray],
        face_embedding: Optional[np.ndarray] = None,
        detection_confidence: float = 0.0,
        frame_num: int = 0,
        identity: Optional[str] = None,
        identity_locked: bool = False
    ) -> int:
        """
        Assign global ID to local track (called per frame for active tracks).

        Matching priority:
        1. Face identity (if locked) - most reliable
        2. Body ReID (if identity not locked) - fallback

        Args:
            camera_id: Camera identifier
            local_track_id: Local track ID from PersonTracker
            person_crop: Cropped person image (H, W, C)
            face_embedding: Face embedding if available (not used in Phase 1)
            detection_confidence: Detection confidence score
            frame_num: Current frame number
            identity: Face recognition identity name (if available)
            identity_locked: Whether face identity is locked (confirmed)

        Returns:
            Global track ID
        """
        if not self.enabled:
            return local_track_id  # Return local ID if disabled

        start_time = time.time()

        # Check if already assigned
        if local_track_id in self.local_to_global[camera_id]:
            global_id = self.local_to_global[camera_id][local_track_id]

            # Update cached embedding periodically
            if person_crop is not None:
                self._maybe_update_embedding(
                    camera_id, local_track_id, person_crop,
                    detection_confidence, frame_num
                )

            # Update last seen time
            if global_id in self.global_tracks:
                track = self.global_tracks[global_id]
                track.last_seen = datetime.now()
                if camera_id in track.camera_tracks:
                    track.camera_tracks[camera_id].last_seen = datetime.now()

                # Update identity if now locked (and track doesn't have one)
                if identity_locked and identity and not track.identity:
                    track.set_identity(identity, locked=True)
                    logger.info(
                        f"GLOBAL_IDENTITY_UPDATE | global_id={global_id} "
                        f"identity='{identity}' (locked)"
                    )

            return global_id

        # New track - need to match or create
        self.metrics.total_assignments += 1

        # STEP 0: Identity-based matching (PRIMARY - if identity is locked)
        if identity_locked and identity:
            existing_track = self.find_global_track_by_identity(identity)
            if existing_track is not None:
                # Found existing global track with same identity - reuse it
                global_id = existing_track.global_id
                self._associate_track(
                    global_id, camera_id, local_track_id,
                    None, detection_confidence  # No embedding needed for identity match
                )
                logger.info(
                    f"GLOBAL_IDENTITY_MATCH | global_id={global_id} camera={camera_id} "
                    f"local_id={local_track_id} identity='{identity}'"
                )
                self.metrics.matched_to_existing += 1
                self._record_matching_time(start_time)
                return global_id

        # STEP 1: Validate crop quality
        if person_crop is None or not self._validate_crop_quality(person_crop):
            logger.debug(
                f"Poor crop quality for camera={camera_id} track={local_track_id}, "
                f"skipping ReID match"
            )
            global_id = self._create_new_global_track(
                camera_id, local_track_id, None, 0.0,
                identity=identity, identity_locked=identity_locked
            )
            self._record_matching_time(start_time)
            return global_id

        # STEP 2: Extract body embedding
        body_embedding = self._extract_body_embedding(person_crop)
        if body_embedding is None:
            global_id = self._create_new_global_track(
                camera_id, local_track_id, None, 0.0,
                identity=identity, identity_locked=identity_locked
            )
            self._record_matching_time(start_time)
            return global_id

        body_quality = detection_confidence

        # Cache the embedding
        self.embedding_cache[(camera_id, local_track_id)] = CachedEmbedding(
            embedding=body_embedding,
            frame_num=frame_num,
            quality=body_quality,
            timestamp=datetime.now()
        )

        # STEP 3: Hard rejection - low quality
        if body_quality < self.min_quality:
            logger.debug(
                f"Low quality ({body_quality:.2f}) for camera={camera_id} "
                f"track={local_track_id}, creating new global ID"
            )
            global_id = self._create_new_global_track(
                camera_id, local_track_id, body_embedding, body_quality,
                identity=identity, identity_locked=identity_locked
            )
            self._record_matching_time(start_time)
            return global_id

        # STEP 4: Get candidate tracks (temporal gating)
        candidates = self._get_candidate_tracks(
            camera_id=camera_id,
            time_window=self.temporal_window_sec
        )

        if not candidates:
            global_id = self._create_new_global_track(
                camera_id, local_track_id, body_embedding, body_quality,
                identity=identity, identity_locked=identity_locked
            )
            self._record_matching_time(start_time)
            return global_id

        # STEP 5: Match against candidates (body ReID only in Phase 1)
        best_match = None
        best_similarity = 0.0

        for candidate in candidates:
            # Hard rejection: same camera too recently
            if camera_id in candidate.camera_tracks:
                last_seen = candidate.camera_tracks[camera_id].last_seen
                time_since = (datetime.now() - last_seen).total_seconds()
                if time_since < self.min_reentry_gap_sec:
                    continue  # Skip: too recent on this camera

            # Compute body similarity
            if candidate.body_prototype is not None:
                similarity = self._cosine_similarity(
                    body_embedding, candidate.body_prototype
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = candidate

        # STEP 6: Conservative decision
        if best_match and best_similarity >= self.similarity_threshold:
            # MATCH: Assign existing global ID
            global_id = best_match.global_id
            self._associate_track(
                global_id, camera_id, local_track_id,
                body_embedding, body_quality
            )

            # Update identity on matched track if we have one and it doesn't
            if identity_locked and identity and not best_match.identity:
                best_match.set_identity(identity, locked=True)

            logger.info(
                f"GLOBAL_MATCH | global_id={global_id} camera={camera_id} "
                f"local_id={local_track_id} similarity={best_similarity:.3f}"
            )
            self.metrics.matched_to_existing += 1
            self.metrics.avg_similarity_matched = (
                (self.metrics.avg_similarity_matched *
                 (self.metrics.matched_to_existing - 1) + best_similarity) /
                self.metrics.matched_to_existing
            )
        else:
            # NO MATCH: Create new global ID (prefer false negatives)
            global_id = self._create_new_global_track(
                camera_id, local_track_id, body_embedding, body_quality,
                identity=identity, identity_locked=identity_locked
            )

            if best_similarity > 0:
                self.metrics.avg_similarity_rejected = (
                    (self.metrics.avg_similarity_rejected *
                     self.metrics.created_new + best_similarity) /
                    (self.metrics.created_new + 1)
                )

            logger.info(
                f"GLOBAL_NEW | global_id={global_id} camera={camera_id} "
                f"local_id={local_track_id} best_sim={best_similarity:.3f}"
            )

        self._record_matching_time(start_time)
        return global_id

    def _get_candidate_tracks(
        self,
        camera_id: int,
        time_window: float,
        include_same_camera: bool = False
    ) -> List[GlobalTrack]:
        """
        Get candidate global tracks for matching (with gating).

        Args:
            camera_id: Current camera ID
            time_window: Temporal window in seconds
            include_same_camera: Whether to include same-camera candidates

        Returns:
            List of candidate GlobalTrack objects
        """
        current_time = datetime.now()
        candidates = []

        for global_id, track in self.global_tracks.items():
            # Gate 1: Temporal - only recent tracks
            time_since_last_seen = (current_time - track.last_seen).total_seconds()
            if time_since_last_seen > time_window:
                continue  # Too old

            # Gate 2: Same-camera re-entry cooldown
            if not include_same_camera and camera_id in track.camera_tracks:
                camera_track = track.camera_tracks[camera_id]
                time_since_on_camera = (
                    current_time - camera_track.last_seen
                ).total_seconds()
                if time_since_on_camera < self.min_reentry_gap_sec:
                    continue  # Too recent on this camera

            # Gate 3: Quality - only tracks with good embeddings
            if track.body_prototype is None:
                continue  # No embedding yet

            candidates.append(track)

        return candidates

    def _create_new_global_track(
        self,
        camera_id: int,
        local_track_id: int,
        body_embedding: Optional[np.ndarray],
        body_quality: float,
        identity: Optional[str] = None,
        identity_locked: bool = False
    ) -> int:
        """Create a new global track."""
        global_id = self.global_id_counter
        self.global_id_counter += 1

        track = GlobalTrack(global_id)
        track.add_camera_track(camera_id, local_track_id)

        if body_embedding is not None:
            track.add_embedding(
                body_embedding, body_quality,
                self.top_k_size, self.prototype_alpha
            )

        # Set identity if provided
        if identity and identity_locked:
            track.set_identity(identity, locked=True)

        self.global_tracks[global_id] = track
        self.local_to_global[camera_id][local_track_id] = global_id

        self.metrics.created_new += 1
        return global_id

    def _associate_track(
        self,
        global_id: int,
        camera_id: int,
        local_track_id: int,
        body_embedding: np.ndarray,
        body_quality: float
    ) -> None:
        """Associate local track with existing global track."""
        track = self.global_tracks[global_id]

        # Update camera presence
        track.add_camera_track(camera_id, local_track_id)

        # Update embeddings (top-K + prototype)
        track.add_embedding(
            body_embedding, body_quality,
            self.top_k_size, self.prototype_alpha
        )

        # Update mapping
        self.local_to_global[camera_id][local_track_id] = global_id

    def _maybe_update_embedding(
        self,
        camera_id: int,
        local_track_id: int,
        person_crop: np.ndarray,
        quality: float,
        frame_num: int
    ) -> None:
        """Update embedding if interval passed."""
        key = (camera_id, local_track_id)

        if key not in self.embedding_cache:
            return

        cached = self.embedding_cache[key]
        frames_since_last = frame_num - cached.frame_num

        # Only extract if interval passed
        if frames_since_last >= self.extract_interval_frames:
            if not self._validate_crop_quality(person_crop):
                return

            new_embedding = self._extract_body_embedding(person_crop)
            if new_embedding is None:
                return

            # Update cache
            self.embedding_cache[key] = CachedEmbedding(
                embedding=new_embedding,
                frame_num=frame_num,
                quality=quality,
                timestamp=datetime.now()
            )
            self.metrics.cache_hit_rate = 0  # Reset on update

            # Update global track prototype
            global_id = self.local_to_global[camera_id].get(local_track_id)
            if global_id and global_id in self.global_tracks:
                self.global_tracks[global_id].add_embedding(
                    new_embedding, quality,
                    self.top_k_size, self.prototype_alpha
                )

    def _extract_body_embedding(
        self,
        person_crop: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract body ReID embedding from person crop.

        Args:
            person_crop: Person crop image (H, W, C) in BGR format

        Returns:
            Normalized embedding vector or None on failure
        """
        if self._body_reid_model is None:
            try:
                self._init_body_reid_model()
            except Exception:
                return None

        if self._body_reid_model is None:
            return None

        try:
            start_time = time.time()

            # Preprocess for OSNet (128x256)
            crop_resized = cv2.resize(person_crop, (128, 256))

            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

            # Convert to tensor
            crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float()
            crop_tensor = crop_tensor.unsqueeze(0) / 255.0
            crop_tensor = crop_tensor.to(self._device)

            # Extract features
            with torch.no_grad():
                embedding = self._body_reid_model(crop_tensor)

            # Convert to numpy and normalize
            embedding = embedding.cpu().numpy()[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Record extraction time
            extraction_time_ms = (time.time() - start_time) * 1000
            self.metrics.avg_extraction_time_ms = (
                self.metrics.avg_extraction_time_ms * 0.9 +
                extraction_time_ms * 0.1
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to extract body embedding: {e}")
            return None

    def _validate_crop_quality(self, person_crop: np.ndarray) -> bool:
        """
        Validate crop quality before ReID extraction.

        Args:
            person_crop: Person crop image (H, W, C)

        Returns:
            True if crop meets quality requirements
        """
        if person_crop is None or person_crop.size == 0:
            return False

        h, w = person_crop.shape[:2]

        # Check minimum dimensions
        if h < self.min_crop_height or w < self.min_crop_width:
            return False

        # Check aspect ratio (person should be roughly vertical)
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False

        # Check minimum area
        area = h * w
        if area < self.min_crop_area:
            return False

        return True

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b))

    def _record_matching_time(self, start_time: float) -> None:
        """Record matching time for metrics."""
        matching_time_ms = (time.time() - start_time) * 1000
        self.metrics.avg_matching_time_ms = (
            self.metrics.avg_matching_time_ms * 0.9 +
            matching_time_ms * 0.1
        )

    # =========================================================================
    # Track Removal Handling
    # =========================================================================

    def on_track_removed(
        self,
        camera_id: int,
        local_track_id: int,
        track_history: Optional[List[Dict]] = None,
        total_frames: int = 0
    ) -> None:
        """
        Handle track removal.

        Args:
            camera_id: Camera identifier
            local_track_id: Local track ID
            track_history: Historical detections (optional)
            total_frames: Total frames this track was active
        """
        if not self.enabled:
            return

        # Get current global ID
        global_id = self.local_to_global[camera_id].get(local_track_id)

        if global_id is None:
            return

        # Mark camera track as inactive
        if global_id in self.global_tracks:
            track = self.global_tracks[global_id]
            track.mark_camera_inactive(camera_id)

            logger.info(
                f"TRACK_INACTIVE | global_id={global_id} camera={camera_id} "
                f"local_id={local_track_id}"
            )

        # Clean up mapping (after some delay to allow re-association)
        # For now, keep mapping - will be cleaned in periodic_validation

        # Clean up embedding cache
        cache_key = (camera_id, local_track_id)
        if cache_key in self.embedding_cache:
            del self.embedding_cache[cache_key]

        # Log baseline stats (Phase 0 compatibility)
        track_key = (camera_id, local_track_id)
        if track_key in self.track_stats:
            stats = self.track_stats[track_key]
            duration = (datetime.now() - stats['created_at']).total_seconds()
            face_rate = stats['faces_detected'] / max(stats['total_frames'], 1)

            logger.info(
                f"TRACK_REMOVED | camera={camera_id} local_id={local_track_id} "
                f"global_id={global_id} duration={duration:.1f}s "
                f"face_rate={face_rate:.2%}"
            )
            del self.track_stats[track_key]

        self.total_tracks_removed += 1

    # =========================================================================
    # Safety Mechanisms
    # =========================================================================

    def detect_impossible_merges(self) -> List[Dict]:
        """
        Detect if same global ID appears on multiple non-overlapping cameras.

        Returns:
            List of conflict dictionaries
        """
        conflicts = []

        for global_id, track in self.global_tracks.items():
            active_cameras = track.get_active_cameras(recency_threshold_sec=5.0)

            if len(active_cameras) > 1:
                # Check if all active cameras can overlap
                if not self._cameras_can_overlap(active_cameras):
                    conflicts.append({
                        'global_id': global_id,
                        'cameras': active_cameras,
                        'severity': 'HIGH'
                    })
                    logger.error(
                        f"IMPOSSIBLE_MERGE | global_id={global_id} "
                        f"active_cameras={active_cameras} - person can't be "
                        f"in two non-overlapping places!"
                    )
                    self.metrics.conflicts_detected += 1

        return conflicts

    def _cameras_can_overlap(self, camera_ids: List[int]) -> bool:
        """Check if given cameras are allowed to see same person simultaneously."""
        for overlap_group in self.overlapping_camera_groups:
            if all(cam_id in overlap_group for cam_id in camera_ids):
                return True
        return False

    def split_global_track(self, global_id: int) -> None:
        """Split incorrectly merged global track."""
        if global_id not in self.global_tracks:
            return

        track = self.global_tracks[global_id]

        # Find camera with longest presence (keep as primary)
        primary_camera = None
        max_duration = 0

        for cam_id, cam_track in track.camera_tracks.items():
            duration = (cam_track.last_seen - cam_track.first_seen).total_seconds()
            if duration > max_duration:
                max_duration = duration
                primary_camera = cam_id

        if primary_camera is None:
            return

        # Create new global IDs for other cameras
        cameras_to_split = [
            cam_id for cam_id in track.camera_tracks.keys()
            if cam_id != primary_camera
        ]

        for cam_id in cameras_to_split:
            cam_track = track.camera_tracks[cam_id]

            # Create new global track
            new_global_id = self.global_id_counter
            self.global_id_counter += 1

            new_track = GlobalTrack(new_global_id)
            new_track.camera_tracks[cam_id] = CameraTrackInfo(
                camera_id=cam_id,
                local_track_id=cam_track.local_track_id,
                first_seen=cam_track.first_seen,
                last_seen=cam_track.last_seen,
                active=cam_track.active
            )

            # Copy prototype (will rebuild independently)
            if track.body_prototype is not None:
                new_track.body_prototype = track.body_prototype.copy()

            self.global_tracks[new_global_id] = new_track

            # Update mapping
            local_id = cam_track.local_track_id
            self.local_to_global[cam_id][local_id] = new_global_id

            logger.info(
                f"SPLIT_TRACK | old_global={global_id} new_global={new_global_id} "
                f"camera={cam_id} local_id={local_id}"
            )
            self.metrics.tracks_split += 1

        # Update original track (keep only primary camera)
        track.camera_tracks = {
            primary_camera: track.camera_tracks[primary_camera]
        }

    def periodic_validation(self) -> None:
        """Run periodic checks for data consistency."""
        current_time = time.time()

        # Rate limit validation
        if current_time - self._last_validation_time < self.validation_interval_sec:
            return

        self._last_validation_time = current_time

        # Check 1: Detect and fix conflicts
        conflicts = self.detect_impossible_merges()
        for conflict in conflicts:
            if conflict['severity'] == 'HIGH':
                self.split_global_track(conflict['global_id'])

        # Check 2: Archive old global tracks
        cutoff_time = datetime.now() - timedelta(
            minutes=self.archive_after_inactive_min
        )
        inactive_tracks = []

        for global_id, track in self.global_tracks.items():
            if track.last_seen < cutoff_time:
                # All camera tracks inactive
                if all(not ct.active for ct in track.camera_tracks.values()):
                    inactive_tracks.append(global_id)

        for global_id in inactive_tracks:
            track = self.global_tracks.pop(global_id)
            duration = track.get_duration_seconds()
            logger.info(
                f"ARCHIVE_TRACK | global_id={global_id} duration={duration:.1f}s "
                f"cameras={list(track.camera_tracks.keys())}"
            )

            # Clean up mappings
            for cam_id, cam_track in track.camera_tracks.items():
                local_id = cam_track.local_track_id
                if (cam_id in self.local_to_global and
                    local_id in self.local_to_global[cam_id]):
                    del self.local_to_global[cam_id][local_id]

        # Check 3: Cache cleanup
        self._cleanup_embedding_cache()

        # Log metrics
        self.metrics.log_summary()

    def _cleanup_embedding_cache(self) -> None:
        """Remove stale cache entries."""
        cutoff_time = datetime.now() - timedelta(minutes=5)
        stale_keys = [
            key for key, cached in self.embedding_cache.items()
            if cached.timestamp < cutoff_time
        ]

        for key in stale_keys:
            del self.embedding_cache[key]

    # =========================================================================
    # Phase 0 Compatibility Methods
    # =========================================================================

    def on_track_created(
        self,
        camera_id: int,
        local_track_id: int,
        bbox: Optional[np.ndarray] = None,
        frame_num: int = 0
    ) -> None:
        """Log when a new track is created (Phase 0 compatibility)."""
        if not self.enabled:
            return

        track_key = (camera_id, local_track_id)
        self.track_stats[track_key] = {
            'camera_id': camera_id,
            'local_track_id': local_track_id,
            'created_at': datetime.now(),
            'first_frame': frame_num,
            'total_frames': 0,
            'faces_detected': 0,
            'faces_not_visible': 0
        }

        self.total_tracks_created += 1

        logger.debug(
            f"TRACK_CREATED | camera={camera_id} local_id={local_track_id} "
            f"frame={frame_num}"
        )

    def on_face_detected(
        self,
        camera_id: int,
        local_track_id: int,
        quality: float = 0.0,
        recognized: bool = False,
        identity: Optional[str] = None
    ) -> None:
        """Log when a face is detected (Phase 0 compatibility)."""
        if not self.enabled:
            return

        track_key = (camera_id, local_track_id)
        if track_key in self.track_stats:
            self.track_stats[track_key]['faces_detected'] += 1

        self.total_faces_detected += 1

    def on_face_not_visible(
        self,
        camera_id: int,
        local_track_id: int
    ) -> None:
        """Log when face is not visible (Phase 0 compatibility)."""
        if not self.enabled:
            return

        track_key = (camera_id, local_track_id)
        if track_key in self.track_stats:
            self.track_stats[track_key]['faces_not_visible'] += 1

        self.total_faces_not_visible += 1

    def on_track_update(
        self,
        camera_id: int,
        local_track_id: int
    ) -> None:
        """Update track frame counter (Phase 0 compatibility)."""
        if not self.enabled:
            return

        track_key = (camera_id, local_track_id)
        if track_key in self.track_stats:
            self.track_stats[track_key]['total_frames'] += 1

    def get_baseline_metrics(self) -> Dict[str, Any]:
        """Get baseline metrics for analysis."""
        if not self.enabled:
            return {}

        total_frames = sum(
            stats['total_frames']
            for stats in self.track_stats.values()
        )

        avg_track_duration = 0.0
        if self.track_stats:
            durations = [
                (datetime.now() - stats['created_at']).total_seconds()
                for stats in self.track_stats.values()
            ]
            avg_track_duration = sum(durations) / len(durations)

        face_visibility_rate = 0.0
        if self.total_faces_detected + self.total_faces_not_visible > 0:
            face_visibility_rate = (
                self.total_faces_detected /
                (self.total_faces_detected + self.total_faces_not_visible)
            )

        return {
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_removed': self.total_tracks_removed,
            'active_tracks': len(self.track_stats),
            'active_global_tracks': len(self.global_tracks),
            'avg_track_duration_sec': avg_track_duration,
            'total_faces_detected': self.total_faces_detected,
            'total_faces_not_visible': self.total_faces_not_visible,
            'face_visibility_rate': face_visibility_rate,
            'match_rate': self.metrics.get_match_rate(),
            'conflicts_detected': self.metrics.conflicts_detected
        }

    def log_baseline_summary(self) -> None:
        """Log a summary of baseline metrics."""
        if not self.enabled:
            return

        metrics = self.get_baseline_metrics()

        logger.info(
            f"GLOBAL_TRACKING_METRICS | "
            f"global_tracks={metrics['active_global_tracks']} "
            f"match_rate={metrics['match_rate']:.1%} "
            f"conflicts={metrics['conflicts_detected']} "
            f"face_visibility={metrics['face_visibility_rate']:.1%}"
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_global_id(self, camera_id: int, local_track_id: int) -> Optional[int]:
        """Get global ID for a local track."""
        return self.local_to_global[camera_id].get(local_track_id)

    def get_global_track(self, global_id: int) -> Optional[GlobalTrack]:
        """Get GlobalTrack by ID."""
        return self.global_tracks.get(global_id)

    def find_global_track_by_identity(self, identity: str) -> Optional[GlobalTrack]:
        """
        Find an active global track by face identity.

        Args:
            identity: Face recognition identity name

        Returns:
            GlobalTrack if found, None otherwise
        """
        if not identity:
            return None

        for global_id, track in self.global_tracks.items():
            if track.active and track.identity == identity:
                return track
        return None

    def update_global_track_identity(
        self,
        global_id: int,
        identity: str,
        locked: bool = True
    ) -> bool:
        """
        Update identity for a global track.

        Args:
            global_id: Global track ID
            identity: Face recognition identity name
            locked: Whether identity is locked

        Returns:
            True if updated, False if track not found
        """
        track = self.global_tracks.get(global_id)
        if track is None:
            return False
        track.set_identity(identity, locked)
        logger.info(f"GLOBAL_IDENTITY | global_id={global_id} identity='{identity}' locked={locked}")
        return True

    def reassign_local_track(
        self,
        camera_id: int,
        local_track_id: int,
        new_global_id: int
    ) -> bool:
        """
        Reassign a local track to a different global track.

        Used when face identity reveals that a track belongs to an existing
        global track (identity-based correction).

        Args:
            camera_id: Camera identifier
            local_track_id: Local track ID to reassign
            new_global_id: Target global track ID

        Returns:
            True if reassigned, False if failed
        """
        # Get current assignment
        old_global_id = self.local_to_global[camera_id].get(local_track_id)

        if old_global_id is None:
            logger.warning(f"Cannot reassign: local track {local_track_id} not found on camera {camera_id}")
            return False

        if old_global_id == new_global_id:
            return True  # Already assigned correctly

        # Get target global track
        target_track = self.global_tracks.get(new_global_id)
        if target_track is None:
            logger.warning(f"Cannot reassign: target global track {new_global_id} not found")
            return False

        # Update mapping
        self.local_to_global[camera_id][local_track_id] = new_global_id

        # Add camera track info to target
        target_track.add_camera_track(camera_id, local_track_id)

        # Transfer embedding if available
        cache_key = (camera_id, local_track_id)
        if cache_key in self.embedding_cache:
            cached = self.embedding_cache[cache_key]
            target_track.add_embedding(
                cached.embedding, cached.quality,
                self.top_k_size, self.prototype_alpha
            )

        logger.info(
            f"REASSIGN | camera={camera_id} local={local_track_id} "
            f"old_global={old_global_id} -> new_global={new_global_id}"
        )
        return True

    def get_all_camera_appearances(
        self,
        global_id: int
    ) -> List[Dict[str, Any]]:
        """Get all camera appearances for a global track."""
        track = self.global_tracks.get(global_id)
        if track is None:
            return []

        return [
            {
                'camera_id': cam_id,
                'local_track_id': cam_track.local_track_id,
                'first_seen': cam_track.first_seen.isoformat(),
                'last_seen': cam_track.last_seen.isoformat(),
                'active': cam_track.active
            }
            for cam_id, cam_track in track.camera_tracks.items()
        ]


class GlobalTrackingMetrics:
    """Metrics for monitoring GlobalTrackManager."""

    def __init__(self):
        # Matching metrics
        self.total_assignments: int = 0
        self.matched_to_existing: int = 0
        self.created_new: int = 0

        # Quality metrics
        self.avg_similarity_matched: float = 0.0
        self.avg_similarity_rejected: float = 0.0

        # Conflict metrics
        self.conflicts_detected: int = 0
        self.tracks_split: int = 0

        # Performance metrics
        self.avg_extraction_time_ms: float = 0.0
        self.avg_matching_time_ms: float = 0.0
        self.cache_hit_rate: float = 0.0

    def get_match_rate(self) -> float:
        """Get match rate as fraction."""
        if self.total_assignments == 0:
            return 0.0
        return self.matched_to_existing / self.total_assignments

    def log_summary(self) -> None:
        """Log metrics summary."""
        logger.info(
            f"GLOBAL_TRACKING_PERF | "
            f"match_rate={self.get_match_rate():.1%} "
            f"conflicts={self.conflicts_detected} "
            f"splits={self.tracks_split} "
            f"avg_match_time={self.avg_matching_time_ms:.1f}ms "
            f"avg_extract_time={self.avg_extraction_time_ms:.1f}ms"
        )
