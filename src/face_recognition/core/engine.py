"""Refactored face recognition engine - orchestrates detection, tracking, and recognition."""

import json
import os
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytz
import requests
from shapely.geometry import LineString
from loguru import logger

from .detector import FaceDetector
from .tracker import FaceTracker
from .track_manager import TrackManager
from .recognizer import FaceRecognition  # FaceRecognition is the class name in recognizer.py
from .deduplicator import UnknownDeduplicator
from .filter_metrics import FilterMetrics


class FaceEngine:
    """Main face recognition engine that orchestrates detection, tracking and recognition.

    This refactored engine delegates responsibilities to specialized components:
    - FaceDetector: Face detection and embedding computation
    - FaceTracker: Face tracking across frames
    - TrackManager: Track lifecycle and history management
    - FaceRecognition: Face recognition and matching
    """

    def __init__(self, args) -> None:
        """Initialize the face recognition engine.

        Args:
            args: Configuration arguments containing parameters for the face recognition system
        """
        self.args = args
        self.client_slug = args.client_slug
        self.FR_SLUG = os.getenv("FR_SLUG")
        self.timezone = args.timezone
        self.max_track_lifetime_seconds = getattr(args, 'max_track_lifetime_seconds', 120)

        # Get face detection padding from args or environment (default: 20%)
        padding_percent = getattr(args, 'face_padding', None) or \
                         float(os.getenv('FACE_DETECTION_PADDING', '20.0'))

        # Initialize specialized components
        self.detector = FaceDetector(gpu_id=args.gpu_id, padding_percent=padding_percent)
        self.tracker = FaceTracker(gpu_id=args.gpu_id)
        self.track_manager = TrackManager(
            timezone=self.timezone,
            max_track_lifetime_seconds=self.max_track_lifetime_seconds
        )
        self.face_recognition = FaceRecognition(args)

        # Setup data collection folder
        self._setup_data_collection_folder()

        # Evaluation results storage
        self.mot_results: List[Dict[str, Any]] = []

        args.logger.info(
            f"Loaded {len(self.face_recognition.db_embs)} embeddings from "
            f"{self.face_recognition.args.db_path}"
        )

        # Initialize temporal deduplicator for unknown faces
        cache_ttl = getattr(args, 'dedup_cache_ttl_seconds', 300)
        similarity_threshold = getattr(args, 'dedup_similarity_threshold', 0.85)
        cross_camera_window = getattr(args, 'dedup_cross_camera_window', 60)

        self.unknown_deduplicator = UnknownDeduplicator(
            cache_ttl_seconds=cache_ttl,
            similarity_threshold=similarity_threshold,
            cross_camera_window_seconds=cross_camera_window
        )

        # Initialize filter metrics for tracking performance
        self.filter_metrics = FilterMetrics()

        # Rate limiting tracker for unknowns
        self._recent_unknowns: Dict[str, deque] = {}

        # Shadow mode flag (log decisions but don't actually filter)
        self.shadow_mode = getattr(args, 'shadow_mode', False)
        if self.shadow_mode:
            args.logger.info("🔍 SHADOW MODE ENABLED - Logging filter decisions without filtering")

        # Setup filter decision logging
        self._setup_filter_logging()

    def _setup_data_collection_folder(self) -> None:
        """Set up folder for storing collected face data and recognition results."""
        self.data_collection_path = os.path.join(
            os.getcwd(),
            f'/app/volumes/storage/{self.FR_SLUG}/data/{self.client_slug}/collection'
        )
        if not os.path.exists(self.data_collection_path):
            os.makedirs(self.data_collection_path)

    def _setup_filter_logging(self) -> None:
        """Setup logging for filter decisions."""
        log_dir = getattr(self.args, 'filter_log_dir', '/app/logs')
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                self.args.logger.warning(f"Could not create filter log directory: {e}")
                return

        self.filter_log_path = os.path.join(log_dir, 'filter_decisions.jsonl')
        self.args.logger.info(f"Filter decisions will be logged to: {self.filter_log_path}")

    def _log_filter_decision(
        self,
        track_id: int,
        decision: str,
        reason: str,
        quality_score: float,
        recognition_info: Dict[str, Any],
        track_data: Dict[str, Any]
    ) -> None:
        """Log detailed filter decision.

        Args:
            track_id: Track ID
            decision: 'SEND' or 'FILTER'
            reason: Reason for the decision
            quality_score: Overall quality score
            recognition_info: Recognition information
            track_data: Track data
        """
        similarity = recognition_info.get('similarity', 0.0)
        lifetime = track_data.get('lifetime_seconds', 0.0)

        # Update metrics
        self.filter_metrics.log_decision(
            decision=decision,
            reason=reason,
            quality=quality_score,
            similarity=similarity,
            lifetime=lifetime
        )

        # Log to file
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'track_id': track_id,
            'camera': self.args.camera_name,
            'camera_type': self.args.cam_type,
            'decision': decision,
            'reason': reason,
            'shadow_mode': self.shadow_mode,

            # Quality metrics
            'quality_score': round(quality_score, 3),
            'frontality_score': round(recognition_info.get('frontality_score', 0), 3),

            # Recognition metrics
            'similarity': round(similarity, 3),
            'best_match_name': recognition_info.get('name', 'unknown'),
            'status': recognition_info.get('status', 'unknown'),
            'confidence': round(recognition_info.get('confidence', 0), 3),
            'top3_similarities': [round(s, 3) for s in recognition_info.get('top3_similarities', [])],

            # Track metrics
            'track_lifetime': round(lifetime, 2),
            'num_frames': track_data.get('num_frames', 0),
        }

        try:
            with open(self.filter_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.args.logger.warning(f"Could not write filter log: {e}")

        # Log summary every 100 tracks
        if self.filter_metrics.counters['total_tracks'] % 100 == 0:
            summary = self.filter_metrics.get_summary()
            self.args.logger.info(
                f"Filter summary (last 100 tracks): "
                f"Reduction: {summary['reduction_rate']}, "
                f"Sent: {summary['sent_count']}, "
                f"Filtered: {summary['filtered_count']}"
            )

    def compute_embeddings(self, image: np.ndarray, alpha: float = 0.9) -> Optional[np.ndarray]:
        """Compute face embeddings for a given image.

        Args:
            image: Input image array
            alpha: Normalization factor for the embedding

        Returns:
            Face embedding vector or None if no face is detected
        """
        return self.detector.compute_embedding(image, alpha)

    def get_emb(self, main_url: str) -> Optional[List[np.ndarray]]:
        """Get face embeddings from images stored in cloud storage.

        Args:
            main_url: URL or JSON string containing URLs to face images

        Returns:
            List of face embeddings, or None if no faces are found
        """
        try:
            # Check if main_url is None or empty
            if not main_url:
                self.args.logger.warning("No image URL provided (None or empty)")
                return None

            # Try to parse as JSON (for array of URLs), otherwise treat as single URL
            try:
                url_list = json.loads(main_url)
                if not isinstance(url_list, list):
                    url_list = [main_url]
            except (json.JSONDecodeError, TypeError):
                # If it's not JSON, treat it as a plain URL string
                url_list = [main_url]

            embeddings = []
            for url in url_list:
                prefix = "https://storage.googleapis.com/"
                if url.startswith(prefix):
                    # Use HTTP request for signed URLs with retry logic
                    max_retries = 3
                    img_bytes = None

                    for attempt in range(max_retries):
                        try:
                            response = requests.get(url, timeout=30, verify=True)
                            response.raise_for_status()
                            img_bytes = response.content
                            break  # Success, exit retry loop
                        except requests.exceptions.SSLError as e:
                            if attempt < max_retries - 1:
                                self.args.logger.warning(
                                    f"SSL error (attempt {attempt + 1}/{max_retries}), retrying..."
                                )
                                # Exponential backoff: 0.5s, 1s, 1.5s
                                import time
                                time.sleep(0.5 * (attempt + 1))
                                continue
                            else:
                                self.args.logger.warning(
                                    f"Failed to fetch image after {max_retries} attempts: {e}"
                                )
                                break
                        except requests.exceptions.RequestException as e:
                            self.args.logger.warning(f"Failed to fetch image: {e}")
                            break

                    if img_bytes is None:
                        continue

                    # Decode image from bytes to OpenCV image
                    img_array = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if image is None:
                        self.args.logger.warning("Failed to decode image")
                        continue

                    embedding = self.compute_embeddings(image)

                    if embedding is not None:
                        embeddings.append(embedding)

            if len(embeddings) <= 0:
                self.args.logger.warning(f"No face found in the image")
                return None
            else:
                return embeddings

        except Exception as e:
            self.args.logger.warning(f"Error processing image: {e}")
            return None

    def update_database(self, new_users: List[dict], deleted_users: List[str]) -> None:
        """Update the face recognition database with new users and delete old ones.

        Args:
            new_users: List of new users to add to the database
            deleted_users: List of users to remove from the database
        """
        for user in new_users:
            embedding = self.get_emb(user['image_path'])

            if embedding is None:
                self.args.logger.warning(f"{user['name']} has no face in the image.")
                continue

            for emb in embedding:
                # Add to in-memory cache
                self.face_recognition.db_names.append(user['name'])
                self.face_recognition.db_embs = np.append(
                    self.face_recognition.db_embs, [emb], axis=0
                )

            self.args.logger.info(
                f"Added {user['name']} to the database with {len(embedding)} images."
            )

        # Rebuild index after adding new users
        if new_users:
            self.face_recognition._rebuild_name_index()

        for name in deleted_users:
            index = self.face_recognition.get_index_by_name(name)
            if index is not None:
                # Delete from in-memory cache
                self.face_recognition.db_names.pop(index)
                self.face_recognition.db_embs = np.delete(
                    self.face_recognition.db_embs, index, axis=0
                )

                # Rebuild index after deletion
                self.face_recognition._rebuild_name_index()
                self.args.logger.info(f"Deleted {name} from the database.")
            else:
                self.args.logger.warning(f"{name} not found in the database.")

        self.face_recognition.update_pkl()

    def track(self, frame: np.ndarray) -> Tuple[List, List]:
        """Track faces in a given frame using the detection and tracking pipeline.

        Args:
            frame: Input frame to process

        Returns:
            Tuple containing lists of active tracks and removed tracks
        """
        # Detect faces and extract features
        face_features = self.detector.extract_face_features(frame)

        if not face_features:
            # No faces detected
            active_tracks, removed_tracks = self.tracker.update(
                np.empty((0, 6)), frame, np.empty((0, 512))
            )
            return active_tracks, removed_tracks

        # Prepare data for tracker
        boxes = np.array([f['bbox'] for f in face_features])
        features = np.array([f['embedding'] for f in face_features])

        # Store landmarks temporarily for this frame (for later retrieval)
        self.current_frame_landmarks = {
            idx: f['landmarks'] for idx, f in enumerate(face_features)
        }

        # Update tracker
        active_tracks, removed_tracks = self.tracker.update(boxes, frame, features)

        return active_tracks, removed_tracks

    def visualize_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Visualize tracked faces on the input frame.

        Args:
            frame: Input frame to visualize tracks on

        Returns:
            Frame with visualization of tracked faces
        """
        visualization_frame = self.tracker.visualize(frame, show_trajectories=True)

        # Put cam_type on the top right corner
        cam_type = self.args.cam_type
        cv2.putText(
            visualization_frame,
            cam_type,
            (visualization_frame.shape[1] - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return visualization_frame

    def process_active_tracks(
        self,
        tracks: List,
        frame: np.ndarray,
        frame_num: int
    ) -> None:
        """Process active tracks to extract and store face data.

        Args:
            tracks: List of active tracks
            frame: Current frame being processed
            frame_num: Frame number in the sequence
        """
        for idx, track in enumerate(tracks):
            emb, track_id = track.emb, track.id

            # Register track
            self.track_manager.register_track(track_id)

            if track.history_observations and len(track.history_observations) > 2:
                box = track.history_observations[-1]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                face_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

                h, w = face_crop.shape[:2]
                if h < self.args.minimum_face_size or w < self.args.minimum_face_size:
                    continue

                # Get landmarks if available
                landmarks = None
                if hasattr(self, 'current_frame_landmarks') and idx in self.current_frame_landmarks:
                    landmarks = self.current_frame_landmarks[idx]

                # Store track data (embeddings, crops, landmarks, etc.)
                self.track_manager.store_track_data(
                    track_id=track_id,
                    frame_num=frame_num,
                    embedding=emb,
                    box=[x1, y1, x2, y2, track.conf, 0],
                    center=center,
                    face_crop=face_crop,
                    landmarks=landmarks
                )

                # Calculate quality score and update best frame using two-tier strategy
                quality_score = 0.0
                if landmarks is not None and len(landmarks) == 5:
                    bbox = [x1, y1, x2, y2, track.conf, 0]
                    is_valid, quality_score, _ = self.face_recognition.enhanced_frame_quality_check(
                        face_crop, landmarks, bbox
                    )
                    # Only update if frame passes basic quality checks
                    if is_valid:
                        self.track_manager.update_best_frame(
                            track_id=track_id,
                            frame_num=frame_num,
                            full_frame=frame,
                            landmarks=landmarks,
                            bbox=bbox,
                            quality_score=quality_score,
                            frontality_threshold=0.65
                        )
                else:
                    # No landmarks - use bbox size only as fallback
                    # Set very low quality score so it's only used if no frontal frames exist
                    self.track_manager.update_best_frame(
                        track_id=track_id,
                        frame_num=frame_num,
                        full_frame=frame,
                        landmarks=None,
                        bbox=[x1, y1, x2, y2, track.conf, 0],
                        quality_score=0.1,  # Low score = fallback only
                        frontality_threshold=0.65
                    )

    def prune_long_lived_tracks(self) -> List[int]:
        """Identify and prune tracks that have exceeded their maximum lifetime.

        Returns:
            A list of track IDs that were pruned.
        """
        active_tracks = self.tracker.get_active_tracks()
        expired_track_ids = self.track_manager.find_expired_tracks(active_tracks)

        if expired_track_ids:
            self.tracker.remove_tracks(expired_track_ids)
            # Clean up track_manager cache to prevent memory leaks
            for track_id in expired_track_ids:
                self.track_manager.delete_track_cache(track_id)

        return expired_track_ids

    def _get_recent_unknown_count(self, camera_name: str, window_seconds: int = 60) -> int:
        """Count unknowns sent in recent time window.

        Args:
            camera_name: Camera name
            window_seconds: Time window in seconds

        Returns:
            Count of unknowns sent in the window
        """
        if camera_name not in self._recent_unknowns:
            self._recent_unknowns[camera_name] = deque(maxlen=20)

        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)

        # Clean old entries
        self._recent_unknowns[camera_name] = deque(
            [t for t in self._recent_unknowns[camera_name] if t > cutoff],
            maxlen=20
        )

        return len(self._recent_unknowns[camera_name])

    def _record_unknown_sent(self, camera_name: str) -> None:
        """Record that an unknown was sent to dashboard.

        Args:
            camera_name: Camera name
        """
        if camera_name not in self._recent_unknowns:
            self._recent_unknowns[camera_name] = deque(maxlen=20)

        self._recent_unknowns[camera_name].append(datetime.now())

    def should_send_unrecognized_to_dashboard(
        self,
        track_id: int,
        recognition_info: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[np.ndarray], float]:
        """Multi-stage decision for unrecognized faces.

        Args:
            track_id: Track ID
            recognition_info: Recognition information

        Returns:
            Tuple of (should_send, reason, image, quality_score)
        """
        # Get track data
        appear_time = self.track_manager.get_track_appear_time(track_id)
        track_embeddings = self.track_manager.get_track_embeddings(track_id)
        track_landmarks = self.track_manager.get_track_landmarks(track_id)

        # Calculate track lifetime
        track_lifetime = 0.0
        if appear_time:
            now = datetime.now(pytz.timezone(self.timezone))
            track_lifetime = (now - appear_time).total_seconds()

        # Prepare track data for logging
        track_data = {
            'lifetime_seconds': track_lifetime,
            'num_frames': len(track_embeddings),
        }

        # === STAGE 1: Track Lifetime Gating ===
        min_lifetime = getattr(self.args, 'min_unrecognized_track_lifetime', 1.5)
        if track_lifetime < min_lifetime:
            self._log_filter_decision(
                track_id, 'FILTER', f'short_track_{track_lifetime:.1f}s',
                0.0, recognition_info, track_data
            )
            return False, f"short_track_{track_lifetime:.1f}s", None, 0.0

        # === STAGE 2: Check UNCERTAIN Status ===
        status = recognition_info.get('status', 'UNKNOWN')
        if status == 'UNCERTAIN':
            # Option A: Wait for better frames (check if track has improved)
            # For now, we'll filter UNCERTAIN cases but log them
            self._log_filter_decision(
                track_id, 'FILTER', 'uncertain_match',
                0.0, recognition_info, track_data
            )
            return False, "uncertain_match", None, 0.0

        # === STAGE 3: Quality Score Validation ===
        matched_frame_num = recognition_info.get('matched_frame_num')
        if matched_frame_num is None:
            self._log_filter_decision(
                track_id, 'FILTER', 'no_frontal_frame',
                0.0, recognition_info, track_data
            )
            return False, "no_frontal_frame", None, 0.0

        # Calculate average quality score from track
        quality_scores = []
        for frame_num in track_embeddings.keys():
            if frame_num in track_landmarks:
                face_crop = self.track_manager.get_track_crop(track_id, frame_num)
                landmarks = track_landmarks[frame_num]

                if face_crop is not None and landmarks is not None:
                    # Get bbox for this frame (simplified - use frame dimensions)
                    h, w = face_crop.shape[:2]
                    bbox = np.array([0, 0, w, h, 1.0, 0])

                    is_valid, quality, _ = self.face_recognition.enhanced_frame_quality_check(
                        face_crop, landmarks, bbox
                    )
                    if is_valid:
                        quality_scores.append(quality)

        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        min_quality = getattr(self.args, 'min_track_quality_score', 0.50)

        if avg_quality < min_quality:
            self._log_filter_decision(
                track_id, 'FILTER', f'low_quality_{avg_quality:.2f}',
                avg_quality, recognition_info, track_data
            )
            return False, f"low_quality_{avg_quality:.2f}", None, avg_quality

        # Require minimum quality frames
        min_quality_frames = getattr(self.args, 'min_quality_frames', 2)
        if len(quality_scores) < min_quality_frames:
            self._log_filter_decision(
                track_id, 'FILTER', f'insufficient_quality_frames_{len(quality_scores)}',
                avg_quality, recognition_info, track_data
            )
            return False, f"insufficient_quality_frames_{len(quality_scores)}", None, avg_quality

        # === STAGE 4: Similarity Check ===
        # Ensure similarity is low enough (confident unknown)
        similarity = recognition_info.get('similarity', 0.0)
        max_unknown_similarity = getattr(self.args, 'max_unknown_similarity', 0.28)

        if similarity > max_unknown_similarity:
            self._log_filter_decision(
                track_id, 'FILTER', f'similarity_too_high_{similarity:.3f}',
                avg_quality, recognition_info, track_data
            )
            return False, f"similarity_too_high_{similarity:.3f}", None, avg_quality

        # === STAGE 5: Temporal Deduplication ===
        matched_embedding = track_embeddings.get(matched_frame_num)
        if matched_embedding is None:
            self._log_filter_decision(
                track_id, 'FILTER', 'no_embedding_for_matched_frame',
                avg_quality, recognition_info, track_data
            )
            return False, "no_embedding_for_matched_frame", None, avg_quality

        camera_name = self.args.camera_name
        should_send, dedup_reason = self.unknown_deduplicator.should_send(
            embedding=matched_embedding,
            camera_name=camera_name,
            quality_score=avg_quality,
            track_id=track_id
        )

        if not should_send:
            self._log_filter_decision(
                track_id, 'FILTER', dedup_reason,
                avg_quality, recognition_info, track_data
            )
            return False, dedup_reason, None, avg_quality

        # === STAGE 6: Rate Limiting ===
        max_per_minute = getattr(self.args, 'max_unknowns_per_camera_per_minute', 12)
        recent_count = self._get_recent_unknown_count(camera_name, window_seconds=60)

        if recent_count >= max_per_minute:
            self._log_filter_decision(
                track_id, 'FILTER', f'rate_limited_{recent_count}',
                avg_quality, recognition_info, track_data
            )
            return False, f"rate_limited_{recent_count}", None, avg_quality

        # === ALL CHECKS PASSED ===
        # Send full frame for unrecognized faces (not just crop)
        image = self.track_manager.get_track_frame(track_id, matched_frame_num)

        self._log_filter_decision(
            track_id, 'SEND', dedup_reason,
            avg_quality, recognition_info, track_data
        )

        # Record that we sent this unknown
        self._record_unknown_sent(camera_name)

        return True, dedup_reason, image, avg_quality

    def recognize_removed_tracks(
        self,
        removed_tracks: List[int],
        last_frame: bool = False
    ) -> Dict[str, List]:
        """Recognize faces in tracks that are no longer active.

        Args:
            removed_tracks: List of track IDs that are no longer active
            last_frame: Whether this is the last frame of the video

        Returns:
            Dictionary mapping recognized person names to their track information
        """
        persons_logged = {}

        # Determine which tracks to process
        if last_frame:
            tracks_to_process = self.track_manager.get_unprocessed_tracks(last_frame=True)
        else:
            tracks_to_process = removed_tracks

        # Filter out already processed tracks
        tracks_to_process = [
            tid for tid in tracks_to_process
            if not self.track_manager.is_track_passed(tid)
        ]

        for track_id in tracks_to_process:
            self.track_manager.mark_track_passed(track_id)

            # Validate track has embeddings
            if not self.track_manager.has_embeddings(track_id):
                self.track_manager.delete_track_cache(track_id)
                continue

            # Validate track passed counting line
            if not self._count_line_passing(track_id):
                self.args.logger.debug(f"Track {track_id} skipped - did not pass counting line")
                self.track_manager.delete_track_cache(track_id)
                continue

            # Perform recognition
            recognition_info = self._perform_recognition(track_id)
            if recognition_info is None:
                self.args.logger.debug(f"Track {track_id} skipped - recognition failed")
                self.track_manager.delete_track_cache(track_id)
                continue

            # No need to cache best frame - we have all frames in track_frame_history
            # matched_frame_num from recognition will be used directly to retrieve frame

            # Log recognition result
            self._log_recognition_result(track_id, recognition_info)

            # Get appropriate image for this recognition result
            image = self._get_recognition_image(track_id, recognition_info)

            # Gate 1: skip image-less unrecognized tracks only in non-eval mode
            if not self.args.eval:
                if image is None and recognition_info['recognized'] == 'unrecognized':
                    # Unrecognized face failed validation
                    self.track_manager.delete_track_cache(track_id)
                    continue

            # Enhanced filtering for unrecognized faces
            if recognition_info['recognized'] == 'unrecognized':
                if not self.args.eval:
                    # Gate 2: production filter — use comprehensive multi-stage filtering
                    should_send, reason, filtered_image, quality_score = self.should_send_unrecognized_to_dashboard(
                        track_id, recognition_info
                    )

                    # In shadow mode, log but always send
                    if self.shadow_mode:
                        if not should_send:
                            self.args.logger.debug(
                                f"SHADOW MODE: Would have filtered {track_id} - {reason} (quality: {quality_score:.2f})"
                            )
                        # Continue with original logic in shadow mode
                        image = self._get_recognition_image(track_id, recognition_info)
                    else:
                        # Production mode: actually filter
                        if not should_send:
                            self.args.logger.debug(
                                f"Filtered unrecognized {track_id}: {reason} (quality: {quality_score:.2f})"
                            )
                            self.track_manager.delete_track_cache(track_id)
                            continue

                        # Use the filtered image
                        image = filtered_image
                else:
                    # Eval mode: no image needed, just recognition_info
                    image = None

            # Store result
            name = recognition_info['name']
            appear_time = self.track_manager.get_track_appear_time(track_id)
            persons_logged[name] = [
                track_id,
                appear_time,
                recognition_info['recognized'],
                image,
                recognition_info
            ]

            if self.args.eval:
                self._record_evaluation_results(track_id, name)

            self.track_manager.delete_track_cache(track_id)

        return persons_logged

    def _perform_recognition(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Perform face recognition for a given track.

        Args:
            track_id: Track ID to recognize

        Returns:
            Recognition information dictionary, or None if recognition fails
        """
        track_embeddings = self.track_manager.get_track_embeddings(track_id)
        frame_nums = list(track_embeddings.keys())
        emb_array = np.array(list(track_embeddings.values()))
        track_landmarks = self.track_manager.get_track_landmarks(track_id)

        return self.face_recognition.recognize_face(emb_array, frame_nums, track_landmarks)

    def _log_recognition_result(self, track_id: int, recognition_info: Dict[str, Any]) -> None:
        """Log the recognition result with structured logging.

        Args:
            track_id: Track ID
            recognition_info: Recognition information dictionary
        """
        name = recognition_info['name']
        sim = recognition_info['similarity']
        recognized_status = recognition_info['recognized']
        camera_name = getattr(self.args, 'camera_name', 'Unknown')

        self.args.logger.debug(
            f"{self.args.camera_name}: {self.args.cam_type} -> {track_id} -> "
            f"{name} -> {sim:.2f}."
        )

        # Log with timestamp for console output
        tz = pytz.timezone(self.args.timezone)
        timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

        if recognized_status == 'unrecognized':
            self.args.logger.info(
                f"[{timestamp}] UNRECOGNIZED | Track ID: {track_id} | "
                f"Best Match: {name} ({sim:.2f}) | Camera: {camera_name}"
            )

    def _get_recognition_image(
        self,
        track_id: int,
        recognition_info: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Get the appropriate image for a recognition result.

        Args:
            track_id: Track ID
            recognition_info: Recognition information dictionary

        Returns:
            Image array, or None if validation fails for unrecognized faces
        """
        recognized_status = recognition_info['recognized']
        matched_frame_num = recognition_info['matched_frame_num']

        if recognized_status == 'recognized':
            return self.track_manager.get_track_frame(track_id, matched_frame_num)

        elif recognized_status == 'unrecognized':
            # If no valid frontal frame was found, skip sending
            if matched_frame_num is None:
                self.args.logger.debug(
                    f"Skipping unrecognized face {track_id} - no valid frontal frames found"
                )
                return None

            # Send full frame for unrecognized faces (not just crop)
            image = self.track_manager.get_track_frame(track_id, matched_frame_num)

            # Validate landmarks for unrecognized faces
            if not self._validate_unrecognized_face_landmarks(track_id, matched_frame_num):
                return None

            return image

        return None

    def _validate_unrecognized_face_landmarks(
        self,
        track_id: int,
        frame_num: int
    ) -> bool:
        """Validate that an unrecognized face has valid frontal landmarks.

        Args:
            track_id: Track ID
            frame_num: Frame number to validate

        Returns:
            True if landmarks are valid, False otherwise
        """
        track_landmarks = self.track_manager.get_track_landmarks(track_id)

        if frame_num not in track_landmarks:
            self.args.logger.debug(
                f"Skipping API submission for unrecognized face {track_id} - "
                f"no landmarks available"
            )
            return False

        landmarks = track_landmarks[frame_num]
        if not self.face_recognition.is_face_frontal_and_valid(landmarks):
            self.args.logger.debug(
                f"Skipping API submission for unrecognized face {track_id} - "
                f"not frontal or missing keypoints"
            )
            return False

        return True

    def _count_line_passing(self, track_id: int) -> bool:
        """Check if a tracked face has crossed a configured counting line.

        Args:
            track_id: ID of the track to check

        Returns:
            True if the track has crossed the counting line, False otherwise
        """
        if self.args.line_points is None:
            return True

        road_points = self.track_manager.get_track_trajectory(track_id)

        if len(road_points) < 2:
            return False

        first_point, last_point = road_points[0], road_points[-1]
        track_line = LineString([first_point, last_point])
        counting_line = LineString(self.args.line_points)

        return track_line.intersects(counting_line)

    def _record_evaluation_results(self, track_id: int, name: str) -> None:
        """Record evaluation results for a recognized face.

        Args:
            track_id: ID of the track
            name: Recognized name of the person
        """
        if not self.args.eval:
            return

        # Get first appeared frame number
        track_embeddings = self.track_manager.get_track_embeddings(track_id)
        frame_num = list(track_embeddings.keys())[0]

        save_path = self.args.txt_path

        if not os.path.exists(save_path):
            with open(save_path, 'w') as f:
                f.write("time,name,cam_type\n")

        with open(save_path, 'a') as f:
            total_seconds = frame_num / self.args.fps

            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)

            time_str = f"{minutes}:{seconds:02d}"
            f.write(f"{time_str},{name},{self.args.cam_type}\n")

        appear_time = self.track_manager.get_track_appear_time(track_id)
        self.mot_results.append({
            'name': name,
            'cam_type': self.args.cam_type,
            'time': appear_time.strftime('%Y-%m-%d %H:%M:%S') if appear_time else 'Unknown',
        })
