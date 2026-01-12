#!/usr/bin/env python3
"""Main entry point for the Person Tracking system.

This script initializes and runs the person tracking pipeline for
detecting persons and recognizing faces.

Configuration is loaded from:
1. YAML config file: configs/person_tracking/config.yaml
2. Environment variables (override YAML values)

Environment Variables:
    PERSON_TRACKING_CONFIG: Path to config file (default: configs/person_tracking/config.yaml)
    HB_CLIENTSLUG: Client organization slug
    HB_IN: Primary camera RTSP URL
    POSTGRES_HOST, POSTGRES_PORT, etc.: Database connection
    LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
"""

import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
from loguru import logger

# Import person tracking modules
from person_tracking.engine import PersonTrackingEngine
from person_tracking.config.manager import ConfigurationManager
from person_tracking.config.models import PersonTrackingAppConfig
from person_tracking.logging.setup import setup_logging

# Import StreamHandler from face_recognition module
from face_recognition.video.stream_handler import StreamHandler


class PersonTrackingApp:
    """Main application class for person tracking.

    This class manages the lifecycle of the person tracking system,
    including initialization, processing, and shutdown.
    """

    def __init__(self):
        """Initialize the PersonTrackingApp."""
        self.config: Optional[PersonTrackingAppConfig] = None
        self.engines: Dict[int, PersonTrackingEngine] = {}
        self.streams: Dict[int, StreamHandler] = {}
        self.video_writers: Dict[int, Optional[cv2.VideoWriter]] = {}
        self.running = False
        self.start_time = time.time()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def _load_configuration(self) -> PersonTrackingAppConfig:
        """Load configuration from file and environment.

        Returns:
            Loaded configuration object

        Raises:
            FileNotFoundError: If config file not found
            ValidationError: If configuration is invalid
        """
        # Determine config file path
        default_config_path = Path(__file__).parent.parent.parent.parent / "configs" / "person_tracking" / "config.yaml"
        config_path = os.getenv("PERSON_TRACKING_CONFIG", str(default_config_path))

        if Path(config_path).exists():
            logger.info(f"Loading configuration from: {config_path}")
            config_manager = ConfigurationManager.load_config(config_path)
        else:
            logger.info("Config file not found, loading from environment variables")
            config_manager = ConfigurationManager.load_config_from_env()

        return config_manager.config

    def initialize(self) -> bool:
        """Initialize the application.

        Returns:
            True if initialization successful
        """
        try:
            # Setup initial logging
            log_level = os.getenv("LOG_LEVEL", "INFO")
            setup_logging(level=log_level)

            logger.info("=" * 60)
            logger.info("Person Tracking System - Initializing")
            logger.info("=" * 60)

            # Load configuration
            self.config = self._load_configuration()

            # Reconfigure logging with loaded settings
            setup_logging(
                level=self.config.logging.level,
                log_file=self.config.logging.file_path if self.config.logging.file else None,
                console=self.config.logging.console,
                file_logging=self.config.logging.file
            )

            logger.info(f"Client: {self.config.client_slug}")
            logger.info(f"Cameras configured: {len(self.config.cameras)}")
            logger.info(f"Database: pgvector={'enabled' if self.config.database.use_pgvector else 'disabled'}")

            # Initialize engines and streams for each camera
            for camera_config in self.config.cameras:
                camera_id = camera_config.camera_id
                video_path = camera_config.video_path

                logger.info(f"Initializing Camera {camera_id}: {camera_config.camera_name}")
                logger.info(f"  Source: {video_path}")
                logger.info(f"  Person Model: YOLOv8{camera_config.person_detection.model_size}-pose")

                # Create stream handler
                stream = StreamHandler(
                    src=video_path,
                    logger=logger
                )
                stream.start()
                self.streams[camera_id] = stream

                # Create engine
                engine = PersonTrackingEngine(
                    config=self.config,
                    camera_config=camera_config
                )
                self.engines[camera_id] = engine

                # Initialize video writer if save_video is enabled
                if camera_config.storage.save_annotated_frames:
                    output_dir = Path(camera_config.storage.output_dir) / self.config.client_slug / f"camera_{camera_id}"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    video_filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    video_path = output_dir / video_filename

                    # Get first frame to determine video dimensions
                    first_frame = stream.get_first_frame()
                    if first_frame is not None:
                        h, w = first_frame.shape[:2]

                        # Adjust dimensions if resize is enabled
                        if camera_config.performance.resize_width:
                            target_w = camera_config.performance.resize_width
                            target_h = int(h * target_w / w)
                            w, h = target_w, target_h

                        # Use MP4V codec (compatible with most players)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = camera_config.performance.target_fps

                        video_writer = cv2.VideoWriter(
                            str(video_path),
                            fourcc,
                            fps,
                            (w, h)
                        )

                        if video_writer.isOpened():
                            self.video_writers[camera_id] = video_writer
                            logger.info(f"Video output: {video_path} ({w}x{h} @ {fps} FPS)")
                        else:
                            logger.error(f"Failed to initialize video writer for camera {camera_id}")
                            self.video_writers[camera_id] = None
                    else:
                        logger.warning(f"Cannot initialize video writer - no first frame for camera {camera_id}")
                        self.video_writers[camera_id] = None
                else:
                    self.video_writers[camera_id] = None

                logger.info(f"Camera {camera_id} initialized successfully")

            logger.info("=" * 60)
            logger.info("Initialization complete")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.exception(e)
            return False

    def run(self) -> None:
        """Run the main processing loop."""
        if not self.config:
            logger.error("Configuration not loaded. Call initialize() first.")
            return

        self.running = True
        frame_num = 0
        last_stats_time = time.time()
        last_frame_log_time = time.time()
        stats_interval = 10.0  # Log stats every 10 seconds
        frame_log_interval = 5.0  # Log frame processing every 5 seconds

        logger.info("Starting person tracking pipeline...")

        try:
            while self.running:
                frame_processed = False

                for camera_id, stream in self.streams.items():
                    engine = self.engines[camera_id]
                    camera_config = self._get_camera_config(camera_id)

                    # Read frame from stream
                    ret, frame = stream.read()

                    if not ret or frame is None:
                        logger.warning(f"Camera {camera_id}: Failed to read frame (ret={ret}, frame={'None' if frame is None else 'valid'})")
                        # If this is a video file and we can't read, the video might have ended
                        if stream.is_video:
                            logger.error(f"Camera {camera_id}: Video file ended or cannot be read")
                            self.running = False
                            break
                        continue

                    frame_processed = True

                    # Resize frame if configured
                    if camera_config and camera_config.performance.resize_width:
                        h, w = frame.shape[:2]
                        target_w = camera_config.performance.resize_width
                        target_h = int(h * target_w / w)
                        frame = cv2.resize(frame, (target_w, target_h))

                    # Process frame through pipeline
                    annotated_frame, events = engine.process_frame(frame, frame_num)

                    # Write to video file
                    if camera_id in self.video_writers and self.video_writers[camera_id] is not None:
                        self.video_writers[camera_id].write(annotated_frame)

                    # Save annotated frames periodically (if enabled)
                    if camera_config and camera_config.storage.save_annotated_frames:
                        engine.save_frame(annotated_frame, frame_num)

                    # Log significant events
                    for event in events:
                        self._log_event(camera_id, event)

                if frame_processed:
                    frame_num += 1

                # Log frame processing status periodically
                current_time = time.time()
                if current_time - last_frame_log_time >= frame_log_interval:
                    logger.info(f"Processing: Frame {frame_num} | FPS: {frame_num / (current_time - self.start_time):.1f}")
                    last_frame_log_time = current_time

                # # Log statistics periodically
                # if current_time - last_stats_time >= stats_interval:
                #     self._log_statistics()
                #     last_stats_time = current_time

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            logger.exception(e)
            raise

        finally:
            self.shutdown()

    def _get_camera_config(self, camera_id: int):
        """Get camera configuration by ID."""
        for cc in self.config.cameras:
            if cc.camera_id == camera_id:
                return cc
        return None

    def _log_event(self, camera_id: int, event: Dict) -> None:
        """Log a tracking event."""
        event_type = event.get('event_type', 'unknown')
        track_id = event.get('track_id', 0)
        identity = event.get('identity', 'Unknown')
        confidence = event.get('confidence', 0.0)

        if event_type in ['identity_locked']:
            logger.info(
                f"[Camera {camera_id}] EVENT: {event_type} | "
                f"Track: {track_id} | "
                f"Identity: {identity} | "
                f"Confidence: {confidence:.2f}"
            )

    def _log_statistics(self) -> None:
        """Log current statistics for all cameras."""
        for camera_id, engine in self.engines.items():
            stats = engine.get_statistics()
            logger.info(
                f"[Camera {camera_id}] Stats | "
                f"Frames: {stats['frame_count']} | "
                f"FPS: {stats['current_fps']:.1f} | "
                f"Latency: {stats['average_latency_ms']:.1f}ms | "
                f"Active Tracks: {stats['state_manager_stats']['total_persons']}"
            )

    def shutdown(self) -> None:
        """Shutdown the application gracefully."""
        logger.info("=" * 60)
        logger.info("Shutting down Person Tracking System")
        logger.info("=" * 60)

        # Release video writers
        for camera_id, writer in self.video_writers.items():
            if writer is not None:
                try:
                    writer.release()
                    logger.info(f"Camera {camera_id}: Video writer released")
                except Exception as e:
                    logger.error(f"Camera {camera_id}: Error releasing video writer - {e}")

        # Stop all streams
        for camera_id, stream in self.streams.items():
            try:
                stream.stop()
                logger.info(f"Camera {camera_id}: Stream stopped")
            except Exception as e:
                logger.error(f"Camera {camera_id}: Error stopping stream - {e}")

        # Cleanup all engines
        for camera_id, engine in self.engines.items():
            try:
                engine.cleanup()
                logger.info(f"Camera {camera_id}: Engine cleaned up")
            except Exception as e:
                logger.error(f"Camera {camera_id}: Error cleaning up engine - {e}")

        # Log final statistics
        logger.info("-" * 60)
        logger.info("Final Statistics:")
        logger.info("-" * 60)

        for camera_id, engine in self.engines.items():
            stats = engine.get_statistics()
            logger.info(
                f"Camera {camera_id} | "
                f"Total Frames: {stats['frame_count']} | "
                f"Avg FPS: {stats['current_fps']:.1f} | "
                f"Total Runtime: {stats['total_runtime_seconds']:.1f}s"
            )

        logger.info("=" * 60)
        logger.info("Shutdown complete")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    app = PersonTrackingApp()

    if not app.initialize():
        logger.error("Failed to initialize application")
        sys.exit(1)

    app.run()


if __name__ == "__main__":
    main()
