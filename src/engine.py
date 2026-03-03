"""SmartOfficeEngine - Unified person tracking and face recognition system.

This module provides a unified engine that combines:
- Person detection and tracking (YOLOv8-Pose + BoT-SORT)
- Face recognition within person ROIs (InsightFace)
- Attendance logging (IN/OUT status)

Architecture:
- Uses ModelFactory for ML model initialization
- Uses StreamManager for video I/O
- Uses FrameProcessor for processing logic
"""

import time
import cv2
from typing import List, Optional

from loguru import logger

# Infrastructure
from infrastructure.video import StreamManager, FrameAnnotator
from infrastructure import EntryLogger
from infrastructure.storage import Repository, EmbeddingSyncService

# Domain
from domain.face_detection import ModelFactory, FrameProcessor

# Config
from config import load_cameras_from_db

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

        # Lifecycle state
        self._running = False
        self._start_time = 0.0
        self.needs_reinit = False

        # Initialize repository for database access
        self.repository = Repository(client_slug)

        # Load camera configurations from database
        self.camera_configs = load_cameras_from_db(
            client_slug=client_slug,
            applications=self.applications
        )

        if not self.camera_configs:
            raise ValueError("No cameras configured. Check config file or database.")

        # Initialize models via factory
        self.models = ModelFactory(self.config, client_slug)
        self.models.initialize_all()

        # Sync embeddings on startup
        self._sync_embeddings_on_startup()

        # Build name-to-ID mapping for activity tracking
        self.name_to_id_map = self._build_name_to_id_map()

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

    def stop(self) -> None:
        """Stop the engine (signal handler callback)."""
        self._running = False

    def run(self) -> None:
        """Run the main processing loop."""
        self._running = True
        self._start_time = time.time()
        last_metrics_log_time = time.time()
        last_validation_time = time.time()
        metrics_log_interval = 60.0
        validation_interval = 30.0

        # Start streams
        self.stream_manager.start_streams()
        logger.info("SmartOfficeEngine started")

        try:
            while self._running:
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

            new_configs = load_cameras_from_db(
                client_slug=self.client_slug,
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
                # Camera set changed — restart engine to pick up new cameras
                logger.info(f"Camera set changed: {old_ids} → {new_ids}. Restarting engine...")
                self.needs_reinit = True
                self.camera_configs = new_configs
                self.stop()

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

            # Refresh name-to-ID mapping for new/updated users
            self.name_to_id_map = self._build_name_to_id_map()

            # Update camera engines with new mapping
            for engine in self.camera_engines:
                engine.name_to_id_map = self.name_to_id_map

            # Update entry logger's name_to_id
            self.entry_logger.name_to_id = [
                {'name': name, 'id': user_id}
                for name, user_id in self.name_to_id_map.items()
            ]

            logger.info(f"Face embeddings reloaded: {len(self.models.face_recognizer.db_names)} users")
            return True

        except Exception as e:
            logger.error(f"Failed to reload embeddings: {e}")
            return False

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down SmartOfficeEngine...")

        # Log final global tracking metrics
        if self.models.global_track_manager and self.models.global_track_manager.enabled:
            self._log_final_metrics()

        # Stop action recognizer workers
        self.models.cleanup()

        # Stop streams and writers
        self.stream_manager.cleanup()

        # Close display windows
        if self.show_display:
            cv2.destroyAllWindows()



        # Log final stats
        self._log_final_stats()

        logger.info("SmartOfficeEngine shutdown complete")

    def _sync_embeddings_on_startup(self) -> bool:
        """Sync missing embeddings on startup.

        Returns:
            True if sync successful, False otherwise
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTUP EMBEDDING SYNC")
            logger.info("=" * 80)

            # Initialize sync service
            sync_service = EmbeddingSyncService(
                client_slug=self.client_slug,
                gpu_id=0,
                config=self.config
            )

            # Run sync
            result = sync_service.sync_missing_embeddings()

            if result.get('success'):
                users_processed = result.get('users_processed', 0)
                embeddings_added = result.get('embeddings_added', 0)

                if users_processed > 0:
                    logger.info(
                        f"Startup sync complete: {users_processed} users processed, "
                        f"{embeddings_added} embeddings added"
                    )
                    self.models.face_recognizer.reload_embeddings()
                    logger.info("Face recognizer reloaded with new embeddings")
                else:
                    logger.info("No missing embeddings - database is up to date")

                logger.info("=" * 80)
                return True
            else:
                logger.error(f"Startup sync failed: {result.get('error')}")
                logger.info("=" * 80)
                return False

        except Exception as e:
            logger.error(f"Failed to sync embeddings on startup: {e}")
            logger.warning("Continuing with existing embeddings...")
            return False

    def _build_name_to_id_map(self) -> dict:
        """Build mapping of user names to IDs from database.

        Returns:
            Dictionary mapping name -> user_id
        """
        try:
            users = self.repository.get_user_name_to_id()
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

    def _log_final_metrics(self) -> None:
        """Log final global tracking metrics."""
        logger.info("=" * 80)
        logger.info("PHASE 0 - FINAL BASELINE METRICS")
        logger.info("=" * 80)

        self.models.global_track_manager.log_baseline_summary()

        metrics = self.models.global_track_manager.get_baseline_metrics()
        logger.info(f"Total tracks created: {metrics['total_tracks_created']}")
        logger.info(f"Total tracks removed: {metrics['total_tracks_removed']}")
        logger.info(f"Average track duration: {metrics['avg_track_duration_sec']:.1f}s")
        logger.info(f"Face visibility rate: {metrics['face_visibility_rate']:.1%}")
        logger.info(f"Faces detected: {metrics['total_faces_detected']}")
        logger.info(f"Faces not visible: {metrics['total_faces_not_visible']}")
        logger.info("=" * 80)

    def _log_final_stats(self) -> None:
        """Log final processing statistics."""
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        total_frames = self.frame_processor.total_frames
        fps = total_frames / elapsed if elapsed > 0 else 0

        logger.info(
            f"Final Stats | Frames: {total_frames} | "
            f"Avg FPS: {fps:.1f} | "
            f"Total Runtime: {elapsed:.0f}s"
        )
