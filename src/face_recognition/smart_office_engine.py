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
import time
import signal
from typing import Dict, List, Optional, Tuple

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
from .camera_engine import CameraEngine, GlobalTrackIDGenerator

# Person tracking components
from person_tracking.core.person_detector import PersonDetector
from person_tracking.core.global_track_manager import GlobalTrackManager
from person_tracking.video.frame_annotator import FrameAnnotator


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

        # Load camera configurations from config file or API
        # Use config setting with fallback to env var for backward compatibility
        use_api_for_cameras = self.config.get(
            'use_api_for_cameras',
            os.getenv('USE_API_FOR_CAMERAS', 'true').lower() == 'true'
        )

        if use_api_for_cameras:
            logger.info("Loading camera configs from API (use_api_for_cameras=true)")
            self.camera_configs = self._fetch_camera_configs_from_api()
        else:
            logger.info("Loading camera configs from config file")
            self.camera_configs = self._load_camera_configs_from_file()

        if not self.camera_configs:
            raise ValueError(f"No cameras configured. Check config file or environment variables.")

        # Initialize global track ID generator for cross-camera unique IDs
        self.global_id_generator = GlobalTrackIDGenerator(start_id=1)
        logger.info("Global track ID generator enabled - track IDs will be unique across all cameras")

        self.global_track_manager = GlobalTrackManager(app_config=self.config)
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
        # Use config setting with fallback to env var for backward compatibility
        args.use_pgvector = self.config.get(
            'use_pgvector',
            os.getenv('USE_PGVECTOR', 'true').lower() == 'true'
        )
        args.client_slug = self.client_slug
        args.match_threshold = self.config.get('match_threshold', 0.3)
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
                gpu_id=0,
                config=self.config
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
                    logger.info(f"✅ Video writer initialized: {filename} ({w}x{h}) [MJPEG]")
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
                    # Save to video file
                    if self.save_video and camera_idx < len(self.video_writers):
                        writer = self.video_writers[camera_idx]
                        if writer:
                            writer.write(annotated)

                # Phase 1: Periodic validation and metrics logging
                current_time = time.time()
                if self.global_track_manager and self.global_track_manager.enabled:
                    # Run periodic validation (conflict detection, track archiving)
                    self.global_track_manager.periodic_validation()

                    # Log metrics periodically
                    if current_time - last_metrics_log_time >= metrics_log_interval:
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

            # Get global ID for display (if global tracking is enabled)
            display_id = state.track_id  # Default to local track ID
            if self.global_track_manager and self.global_track_manager.enabled:
                global_id = self.global_track_manager.get_global_id(
                    engine.camera_id, state.track_id
                )
                if global_id is not None:
                    display_id = global_id

            person_states.append({
                'track_id': display_id,
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
