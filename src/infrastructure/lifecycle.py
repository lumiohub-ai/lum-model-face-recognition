"""Engine lifecycle management.

This module handles:
- Signal handling (SIGINT, SIGTERM)
- Startup tasks (embedding sync)
- Shutdown cleanup
- Statistics logging
"""

import signal
import time
from typing import Callable, Dict, List, Optional, Any

import cv2
from loguru import logger

from infrastructure.storage import EmbeddingSyncService
from domain.person_tracking import GlobalTrackManager


class EngineLifecycle:
    """Handles engine lifecycle: startup, shutdown, signals.

    Manages signal handlers, startup initialization, and cleanup procedures.
    """

    def __init__(self):
        """Initialize lifecycle manager."""
        self._shutdown_callback: Optional[Callable] = None
        self._start_time: float = 0
        self._running: bool = False

    def setup_signal_handlers(self, shutdown_callback: Callable) -> None:
        """Set up signal handlers for graceful shutdown.

        Args:
            shutdown_callback: Function to call when shutdown signal received
        """
        self._shutdown_callback = shutdown_callback

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.debug("Signal handlers configured")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.warning(f"Received signal {signum}, shutting down...")
        if self._shutdown_callback:
            self._shutdown_callback()

    def sync_embeddings_on_startup(
        self,
        client_slug: str,
        face_recognizer: Any,
        config: Dict[str, Any]
    ) -> bool:
        """Sync missing embeddings on startup.

        Args:
            client_slug: Organization slug
            face_recognizer: Face recognizer to reload
            config: Application config

        Returns:
            True if sync successful, False otherwise
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTUP EMBEDDING SYNC")
            logger.info("=" * 80)

            # Initialize sync service
            sync_service = EmbeddingSyncService(
                client_slug=client_slug,
                gpu_id=0,
                config=config
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
                    face_recognizer.reload_embeddings()
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

    def build_name_to_id_map(self, client_slug: str) -> Dict[str, int]:
        """Build mapping of user names to IDs from database.

        Args:
            client_slug: Organization slug

        Returns:
            Dictionary mapping name -> user_id
        """
        try:
            from infrastructure.storage import Repository
            repository = Repository(client_slug)
            users = repository.get_user_name_to_id()
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

    def mark_started(self) -> None:
        """Mark engine as started and record start time."""
        self._running = True
        self._start_time = time.time()

    def mark_stopped(self) -> None:
        """Mark engine as stopped."""
        self._running = False

    @property
    def running(self) -> bool:
        """Check if engine is running."""
        return self._running

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self._start_time > 0:
            return time.time() - self._start_time
        return 0

    def cleanup(
        self,
        stream_manager: Any,
        global_track_manager: Optional[GlobalTrackManager],
        entry_logger: Any,
        show_display: bool,
        total_frames: int
    ) -> None:
        """Clean up all resources.

        Args:
            stream_manager: StreamManager to cleanup
            global_track_manager: Optional GlobalTrackManager for final metrics
            entry_logger: EntryLogger to save status
            show_display: Whether display windows are open
            total_frames: Total frames processed
        """
        logger.info("Shutting down SmartOfficeEngine...")

        # Log final baseline metrics
        if global_track_manager and global_track_manager.enabled:
            self._log_final_metrics(global_track_manager)

        # Stop streams and writers
        stream_manager.cleanup()

        # Close display windows
        if show_display:
            cv2.destroyAllWindows()

        # Save entry logger status
        entry_logger.save_status_info()

        # Log final stats
        self._log_final_stats(total_frames)

        logger.info("SmartOfficeEngine shutdown complete")

    def _log_final_metrics(self, global_track_manager: GlobalTrackManager) -> None:
        """Log final global tracking metrics."""
        logger.info("=" * 80)
        logger.info("PHASE 0 - FINAL BASELINE METRICS")
        logger.info("=" * 80)

        global_track_manager.log_baseline_summary()

        metrics = global_track_manager.get_baseline_metrics()
        logger.info(f"Total tracks created: {metrics['total_tracks_created']}")
        logger.info(f"Total tracks removed: {metrics['total_tracks_removed']}")
        logger.info(f"Average track duration: {metrics['avg_track_duration_sec']:.1f}s")
        logger.info(f"Face visibility rate: {metrics['face_visibility_rate']:.1%}")
        logger.info(f"Faces detected: {metrics['total_faces_detected']}")
        logger.info(f"Faces not visible: {metrics['total_faces_not_visible']}")
        logger.info("=" * 80)

    def _log_final_stats(self, total_frames: int) -> None:
        """Log final processing statistics."""
        elapsed = self.elapsed_time
        fps = total_frames / elapsed if elapsed > 0 else 0

        logger.info(
            f"Final Stats | Frames: {total_frames} | "
            f"Avg FPS: {fps:.1f} | "
            f"Total Runtime: {elapsed:.0f}s"
        )
