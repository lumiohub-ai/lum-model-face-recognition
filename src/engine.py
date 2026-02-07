"""SmartOfficeEngine - Unified person tracking and face recognition system.

This module provides a unified engine that combines:
- Person detection and tracking (YOLOv8-Pose + BoT-SORT)
- Face recognition within person ROIs (InsightFace)
- Attendance logging (IN/OUT status)

Architecture:
- Uses ModelFactory for ML model initialization
- Uses StreamManager for video I/O
- Uses FrameProcessor for processing logic
- Uses EngineLifecycle for startup/shutdown
"""

import os
import time
from typing import List, Optional

from loguru import logger

# Infrastructure
from infrastructure.video import StreamManager, FrameAnnotator
from infrastructure import EngineLifecycle, EntryLogger
from infrastructure.storage import Repository

# Domain
from domain.face_detection import ModelFactory, FrameProcessor

# Config
from config import load_cameras

# Local
from camera_engine import CameraEngine


class SmartOfficeEngine:
    """Unified engine for Smart Office person tracking and face recognition.

    Orchestrates components for:
    - Person tracking with stable IDs
    - Face recognition with temporal voting
    - Attendance logging via API
    """

    def __init__(
        self,
        client_slug: str,
        applications: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize SmartOfficeEngine.

        Args:
            client_slug: Organization slug (tenant identifier)
            applications: List of application types (default: ['attendance'])
            **kwargs: Additional configuration from config.yaml
        """
        self.client_slug = client_slug
        self.applications = applications or ['attendance']
        self.config = kwargs

        # Lifecycle manager
        self.lifecycle = EngineLifecycle()

        # Initialize repository for database access
        self.repository = Repository(client_slug)

        # Load camera configurations
        use_db = self.config.get(
            'use_api_for_cameras',
            os.getenv('USE_API_FOR_CAMERAS', 'true').lower() == 'true'
        )
        self.camera_configs = load_cameras(
            client_slug=client_slug,
            use_db=use_db,
            applications=self.applications
        )

        if not self.camera_configs:
            raise ValueError("No cameras configured. Check config file or database.")

        # Initialize models via factory
        self.models = ModelFactory(self.config, client_slug)
        self.models.initialize_all()

        # Sync embeddings on startup
        self.lifecycle.sync_embeddings_on_startup(
            client_slug=client_slug,
            face_recognizer=self.models.face_recognizer,
            config=self.config
        )

        # Build name-to-ID mapping for activity tracking
        self.name_to_id_map = self.lifecycle.build_name_to_id_map(client_slug)

        # Initialize stream manager
        self.stream_manager = StreamManager(self.camera_configs)
        self.stream_manager.init_streams()

        # Initialize video writers if saving
        self.save_video = kwargs.get('save_video', False)
        if self.save_video:
            output_dir = kwargs.get('output_dir', 'volumes/storage/person-tracking')
            self.stream_manager.init_video_writers(output_dir)

        # Initialize camera engines
        self.camera_engines = self._init_camera_engines()

        # Initialize entry logger
        self.entry_logger = self._init_entry_logger()

        # Initialize frame processor
        self.frame_processor = FrameProcessor(
            camera_engines=self.camera_engines,
            camera_configs=self.camera_configs,
            entry_logger=self.entry_logger,
            frame_annotator=FrameAnnotator(),
            global_track_manager=self.models.global_track_manager
        )

        # Display settings
        self.show_display = kwargs.get('show', False)

        # Set up signal handlers
        self.lifecycle.setup_signal_handlers(self._stop)

        logger.info(f"SmartOfficeEngine initialized with {len(self.camera_configs)} camera(s)")

    def _init_camera_engines(self) -> List[CameraEngine]:
        """Initialize camera engines with shared models."""
        engines = []

        for config in self.camera_configs:
            engine = CameraEngine(
                camera_config=config,
                face_detector=self.models.face_detector,
                face_recognizer=self.models.face_recognizer,
                person_detector=self.models.person_detector,
                client_slug=self.client_slug,
                global_id_generator=self.models.global_id_generator,
                name_to_id_map=self.name_to_id_map,
                global_track_manager=self.models.global_track_manager,
                action_recognizer=self.models.action_recognizer
            )
            engines.append(engine)

        return engines

    def _init_entry_logger(self) -> EntryLogger:
        """Initialize entry logger."""
        args = type('Args', (), {})()
        args.client_slug = self.client_slug
        args.logger = logger
        args.db_names = self.models.face_recognizer.db_names
        args.production = True

        return EntryLogger(args=args)

    def _stop(self) -> None:
        """Stop the engine (signal handler callback)."""
        self.lifecycle.mark_stopped()

    def run(self) -> None:
        """Run the main processing loop."""
        self.lifecycle.mark_started()
        last_metrics_log_time = time.time()
        last_validation_time = time.time()
        metrics_log_interval = 60.0
        validation_interval = 30.0

        # Start streams
        self.stream_manager.start_streams()
        logger.info("SmartOfficeEngine started")

        try:
            while self.lifecycle.running:
                # Read frames from all cameras
                frames = self.stream_manager.read_all_frames()

                if not frames:
                    continue

                # Process frames
                annotated_frames = self.frame_processor.process_all_frames(frames)

                # Write to video files
                if self.save_video:
                    self.stream_manager.write_frames(annotated_frames)

                # Periodic tasks
                current_time = time.time()

                # Periodic validation
                if current_time - last_validation_time >= validation_interval:
                    if self.models.global_track_manager:
                        self.models.global_track_manager.periodic_validation()
                    last_validation_time = current_time

                # Periodic metrics logging
                if current_time - last_metrics_log_time >= metrics_log_interval:
                    self.frame_processor.log_metrics()
                    last_metrics_log_time = current_time

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise

        finally:
            self._cleanup()

    def reload_camera_configs(self) -> bool:
        """
        Reload camera configurations from database.

        This is called when camera config changes are received via MDA.
        Returns True if reload was successful, False otherwise.
        """
        try:
            logger.info("Reloading camera configurations...")

            use_db = self.config.get(
                'use_api_for_cameras',
                os.getenv('USE_API_FOR_CAMERAS', 'true').lower() == 'true'
            )

            new_configs = load_cameras(
                client_slug=self.client_slug,
                use_db=use_db,
                applications=self.applications
            )

            if not new_configs:
                logger.warning("No cameras found after reload - keeping existing config")
                return False

            # Check if configs actually changed
            old_ids = set(c.get('camera_id') for c in self.camera_configs)
            new_ids = set(c.get('camera_id') for c in new_configs)

            if old_ids == new_ids:
                # Same cameras, update applications for existing configs
                for new_config in new_configs:
                    for i, old_config in enumerate(self.camera_configs):
                        if old_config.get('camera_id') == new_config.get('camera_id'):
                            self.camera_configs[i] = new_config
                            # Update camera engine application
                            for engine in self.camera_engines:
                                if engine.camera_id == new_config.get('camera_id'):
                                    engine.application = new_config.get('application', ['attendance'])
                                    logger.info(f"Updated camera {engine.camera_id} applications: {engine.application}")
                            break
                logger.info(f"Camera configurations updated (same {len(new_configs)} cameras)")
            else:
                # Different cameras - need full reinit
                logger.warning("Camera set changed - requires engine restart for full reinit")
                self.camera_configs = new_configs
                logger.info(f"Camera configurations reloaded: {len(new_configs)} cameras")

            return True

        except Exception as e:
            logger.error(f"Failed to reload camera configs: {e}")
            return False

    def reload_embeddings(self) -> bool:
        """
        Reload face embeddings from pgvector database.

        This is called when new user embeddings are created/updated/deleted via MDA.
        Returns True if reload was successful, False otherwise.
        """
        try:
            logger.info("Reloading face embeddings...")

            # Reload embeddings in face recognizer
            self.models.face_recognizer.reload_embeddings()

            # Update entry logger's db_names
            self.entry_logger.current_users = self.models.face_recognizer.db_names

            # Reload person status from API (fixes cache/database mismatch)
            self.entry_logger.reload_status()

            logger.info(f"Face embeddings reloaded: {len(self.models.face_recognizer.db_names)} users")
            return True

        except Exception as e:
            logger.error(f"Failed to reload embeddings: {e}")
            return False

    def _cleanup(self) -> None:
        """Clean up resources."""
        # Stop action recognizer workers
        self.models.cleanup()

        self.lifecycle.cleanup(
            stream_manager=self.stream_manager,
            global_track_manager=self.models.global_track_manager,
            entry_logger=self.entry_logger,
            show_display=self.show_display,
            total_frames=self.frame_processor.total_frames
        )
