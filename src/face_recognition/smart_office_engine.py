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
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Core components
from .api.client import APIClient
from .camera_engine import CameraEngine
from .config.camera_loader import load_cameras
from .core.model_factory import ModelFactory
from .core.processing import FrameProcessor
from .logging.entry_logger import EntryLogger
from .services.lifecycle import EngineLifecycle
from .video.stream_manager import StreamManager

# Person tracking
from person_tracking.video.frame_annotator import FrameAnnotator


class SmartOfficeEngine:
    """Unified engine for Smart Office person tracking and face recognition.

    Orchestrates components for:
    - Person tracking with stable IDs
    - Face recognition with temporal voting
    - Attendance logging via API
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
            applications: List of application types (default: ['attendance'])
            **kwargs: Additional configuration
        """
        self.client_slug = client_slug
        self.api_host = api_host
        self.applications = applications or ['attendance']
        self.config = kwargs

        # Lifecycle manager
        self.lifecycle = EngineLifecycle()

        # Initialize API client
        self.api_client = APIClient(
            api_host=api_host,
            email=email,
            password=password,
            client_slug=client_slug
        )

        # Load camera configurations
        use_api = self.config.get(
            'use_api_for_cameras',
            os.getenv('USE_API_FOR_CAMERAS', 'true').lower() == 'true'
        )
        self.camera_configs = load_cameras(
            api_client=self.api_client,
            use_api=use_api,
            applications=self.applications
        )

        if not self.camera_configs:
            raise ValueError("No cameras configured. Check config file or API.")

        # Initialize models via factory
        self.models = ModelFactory(self.config, client_slug, self.api_client)
        self.models.initialize_all()

        # Sync embeddings on startup
        self.lifecycle.sync_embeddings_on_startup(
            client_slug=client_slug,
            api_client=self.api_client,
            face_recognizer=self.models.face_recognizer,
            config=self.config
        )

        # Build name-to-ID mapping for activity tracking
        self.name_to_id_map = self.lifecycle.build_name_to_id_map(self.api_client)

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
        self.entry_logger = self._init_entry_logger(email, password)

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

        # Get VLM image padding from config
        action_config = self.config.get('action_recognition', {})
        vlm_image_padding = action_config.get('image_padding', 20.0)

        for config in self.camera_configs:
            engine = CameraEngine(
                camera_config=config,
                face_detector=self.models.face_detector,
                face_recognizer=self.models.face_recognizer,
                person_detector=self.models.person_detector,
                client_slug=self.client_slug,
                global_id_generator=self.models.global_id_generator,
                api_client=self.api_client,
                name_to_id_map=self.name_to_id_map,
                global_track_manager=self.models.global_track_manager,
                action_recognizer=self.models.action_recognizer,
                vlm_image_padding=vlm_image_padding
            )
            engines.append(engine)

        return engines

    def _init_entry_logger(self, email: str, password: str) -> EntryLogger:
        """Initialize entry logger."""
        args = type('Args', (), {})()
        args.client_slug = self.client_slug
        args.api_host = self.api_host
        args.email = email
        args.password = password
        args.logger = logger
        args.db_names = self.models.face_recognizer.db_names
        args.production = True

        return EntryLogger(args=args)

    def _stop(self) -> None:
        """Stop the engine (signal handler callback)."""
        self.lifecycle.mark_stopped()

    def _refresh_name_to_id_map(self) -> bool:
        """Refresh name-to-ID mapping from API and propagate to camera engines.

        This ensures new users added to the backend are recognized for activity tracking.

        Returns:
            True if refresh succeeded, False otherwise
        """
        try:
            new_map = self.lifecycle.build_name_to_id_map(self.api_client)

            if not new_map:
                logger.warning("Name-to-ID refresh returned empty map, keeping existing")
                return False

            # Check if there are changes
            if new_map != self.name_to_id_map:
                added = set(new_map.keys()) - set(self.name_to_id_map.keys())
                removed = set(self.name_to_id_map.keys()) - set(new_map.keys())

                if added or removed:
                    logger.info(
                        f"Name-to-ID map updated: +{len(added)} added, -{len(removed)} removed"
                    )

                # Update main map
                self.name_to_id_map = new_map

                # Propagate to all camera engines
                for engine in self.camera_engines:
                    engine.name_to_id_map = new_map
                    # Also update state manager if it has a reference
                    if hasattr(engine, 'state_manager') and engine.state_manager:
                        engine.state_manager.name_to_id_map = new_map

            return True

        except Exception as e:
            logger.warning(f"Failed to refresh name-to-ID map: {e}")
            return False

    def run(self) -> None:
        """Run the main processing loop."""
        self.lifecycle.mark_started()
        last_metrics_log_time = time.time()
        last_validation_time = time.time()
        last_user_refresh_time = time.time()
        metrics_log_interval = 60.0
        validation_interval = 30.0
        user_refresh_interval = 300.0  # 5 minutes

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

                # Periodic validation (Phase 4)
                if current_time - last_validation_time >= validation_interval:
                    if self.models.global_track_manager:
                        self.models.global_track_manager.periodic_validation()
                    last_validation_time = current_time

                # Periodic metrics logging
                if current_time - last_metrics_log_time >= metrics_log_interval:
                    self.frame_processor.log_metrics()
                    last_metrics_log_time = current_time

                # Periodic user refresh (for new users added to backend)
                if current_time - last_user_refresh_time >= user_refresh_interval:
                    self._refresh_name_to_id_map()
                    last_user_refresh_time = current_time

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise

        finally:
            self._cleanup()

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
