"""Refactored face recognition engine - orchestrates detection, tracking, and recognition."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gcsfs
import numpy as np
import pytz
import requests
from shapely.geometry import LineString
from loguru import logger

from .detector import FaceDetector
from .tracker import FaceTracker
from .track_manager import TrackManager
from .recognizer import FaceRecognition  # FaceRecognition is the class name in recognizer.py
from ..services.redis_pubsub import RedisSubscriber


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

        # Initialize cloud storage
        self.fs = gcsfs.GCSFileSystem(token=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))

        # Get face detection padding from args or environment (default: 20%)
        # Priority: args.face_detection_padding > args.face_padding > env var
        padding_percent = getattr(args, 'face_detection_padding', None) or \
                         getattr(args, 'face_padding', None) or \
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

        # Start Redis subscriber for real-time embedding updates
        self.redis_subscriber = None
        if getattr(args, 'use_pgvector', False):
            try:
                self.redis_subscriber = RedisSubscriber(
                    client_slug=self.client_slug,
                    on_update=self._on_embedding_update
                )
                self.redis_subscriber.start()
            except Exception as e:
                args.logger.warning(f"Failed to start Redis subscriber: {e}")

    def _setup_data_collection_folder(self) -> None:
        """Set up folder for storing collected face data and recognition results."""
        self.data_collection_path = os.path.join(
            os.getcwd(),
            f'/app/volumes/storage/{self.FR_SLUG}/data/{self.client_slug}/collection'
        )
        if not os.path.exists(self.data_collection_path):
            os.makedirs(self.data_collection_path)

    def _on_embedding_update(self, action: str, user_name: str) -> None:
        """Callback for Redis embedding update events.

        Args:
            action: Action type ('add_user', 'update_user', 'delete_user')
            user_name: Name of the user affected
        """
        try:
            self.args.logger.info(f"🔄 Reloading embeddings due to {action} for {user_name}")

            # Reload embeddings from pgvector database
            self.face_recognition.reload_embeddings()

            self.args.logger.info(
                f"✅ Embeddings reloaded successfully. "
                f"Total: {len(self.face_recognition.db_embs)}"
            )
        except Exception as e:
            self.args.logger.error(f"❌ Failed to reload embeddings: {e}")

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

                # Save to pgvector database if enabled
                if self.face_recognition.use_pgvector and self.face_recognition.pgvector_store:
                    try:
                        self.face_recognition.pgvector_store.add_embedding(
                            user_id=user.get('id', user['name']),
                            user_name=user['name'],
                            image_url=user.get('image_path', ''),
                            embedding=emb,
                            external_id=user.get('external_id'),
                            metadata={'source': 'backend_sync'}
                        )
                    except Exception as e:
                        self.args.logger.error(f"Failed to save embedding to pgvector: {e}")

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

                # Delete from pgvector database if enabled
                if self.face_recognition.use_pgvector and self.face_recognition.pgvector_store:
                    try:
                        self.face_recognition.pgvector_store.delete_user_embeddings(name)
                    except Exception as e:
                        self.args.logger.error(f"Failed to delete from pgvector: {e}")

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

                # Store track data
                self.track_manager.store_track_data(
                    track_id=track_id,
                    frame_num=frame_num,
                    embedding=emb,
                    box=[x1, y1, x2, y2, track.conf, 0],
                    center=center,
                    face_crop=face_crop,
                    full_frame=frame,
                    landmarks=landmarks
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

        return expired_track_ids

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
                continue

            # Log recognition result
            self._log_recognition_result(track_id, recognition_info)

            # Get appropriate image for this recognition result
            image = self._get_recognition_image(track_id, recognition_info)
            if image is None and recognition_info['recognized'] == 'unrecognized':
                # Unrecognized face failed validation
                self.track_manager.delete_track_cache(track_id)
                continue

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

            image = self.track_manager.get_track_crop(track_id, matched_frame_num)

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
