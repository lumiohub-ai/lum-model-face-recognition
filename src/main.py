"""
Entry point for SmartOfficeEngine - Unified person tracking and face recognition.

Uses Pure Message-Driven Architecture (MDA) via Redis.

Architecture:
- MDAManager: Handles Redis pub/sub and stream consumption
- SmartOfficeEngine: Core video processing and face recognition
- Lifecycle management with proper signal handling
"""

import sys
import signal
import atexit
import threading
from pathlib import Path
from typing import Optional, Callable

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import init_smart_office_app, log_startup_info, build_vision_config
from pipeline.engine import SmartOfficeEngine
from config.settings import settings
from infrastructure.storage import PgVectorStore
from messaging import RedisClient, StreamConsumer
from messaging.channels import INTERNAL_CHANNELS
from messaging.subscriber import start_listener
from lum_vision import ModelFactory


# ============================================================
# Application Lifecycle Management
# ============================================================

class ApplicationLifecycle:
    """Manages application lifecycle with proper cleanup.

    Uses weak references to avoid circular dependencies and
    ensures proper shutdown even on signal interrupts.
    """

    _instance: Optional['ApplicationLifecycle'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._shutdown_callbacks: list[Callable[[], None]] = []
        self._is_shutting_down = False

    @classmethod
    def get_instance(cls) -> 'ApplicationLifecycle':
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called on shutdown.

        Args:
            callback: Function to call during shutdown
        """
        self._shutdown_callbacks.append(callback)

    def shutdown(self) -> None:
        """Execute all shutdown callbacks."""
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        logger.info("[Lifecycle] Shutting down...")

        for callback in reversed(self._shutdown_callbacks):
            try:
                callback()
            except Exception as e:
                logger.exception(f"[Lifecycle] Shutdown callback error: {e}")

        logger.info("[Lifecycle] Shutdown complete")

    @property
    def is_running(self) -> bool:
        """Check if application is still running."""
        return not self._is_shutting_down


class MDAManager:
    """Manages MDA components without global state."""

    def __init__(self, client_slug: str):
        self.client_slug = client_slug
        self.engine: Optional[SmartOfficeEngine] = None
        self.stream_consumer = None
        self._running = False
        self._camera_ready = threading.Event()

    def set_engine(self, engine: SmartOfficeEngine) -> None:
        """Set the engine reference for reload operations."""
        self.engine = engine

    def start(self) -> None:
        """Initialize and start all MDA components."""
        if self._running:
            logger.warning("Already running")
            return

        # Verify Redis connection
        if not RedisClient.get_instance().is_connected():
            logger.error("Redis not available")
            sys.exit(1)

        # Start stream consumer for Backend commands
        self.stream_consumer = StreamConsumer(client_slug=self.client_slug)
        self.stream_consumer.set_camera_handler(self._handle_camera_command)
        self.stream_consumer.start()
        logger.debug("StreamConsumer started")

        # Must be set before starting listeners: each listener thread loops on
        # `while self._running`, so if it's still False they exit before subscribing.
        self._running = True

        # Start internal reload listeners
        self._start_reload_listeners()

    def stop(self) -> None:
        """Gracefully shutdown MDA components."""
        if self.stream_consumer:
            logger.info("Stopping stream consumer...")
            self.stream_consumer.stop()
        self._running = False
        logger.info("Shutdown complete")

    def _start_reload_listeners(self) -> None:
        """Start background listeners for internal reload notifications."""
        start_listener(
            INTERNAL_CHANNELS['EMBEDDING_RELOAD'],
            self._handle_embedding_reload,
            is_running=lambda: self._running,
            name="reload-embeddings",
        )
        start_listener(
            INTERNAL_CHANNELS['STATUS_RELOAD'],
            self._handle_status_reload,
            is_running=lambda: self._running,
            name="reload-status",
        )

    def _handle_embedding_reload(self, data: dict) -> None:
        """Handle embedding reload notification."""
        logger.info(f"Embedding reload: {data}")
        if self.engine:
            self.engine.reload_embeddings()
            logger.info("Embeddings reloaded")
        else:
            logger.warning("Engine not available for embedding reload")

    def _handle_status_reload(self, data: dict) -> None:
        """Handle status reload notification."""
        logger.info(f"Status reload: {data}")
        if self.engine and hasattr(self.engine, 'entry_logger'):
            self.engine.entry_logger.reload_status()
        else:
            logger.warning("Engine not available")

    def _handle_camera_command(self, command_type: str, client_slug: str, payload: dict) -> None:
        """Handle camera config commands from Backend."""
        if client_slug and client_slug != self.client_slug:
            logger.debug(f"Ignoring {command_type} for {client_slug} (we are {self.client_slug})")
            return

        camera_id = payload.get('camera_id')
        if camera_id is not None:
            try:
                camera_id = int(camera_id)
            except (ValueError, TypeError):
                pass

        if command_type in ('ConfigureCamera', 'StartCamera'):
            if self.engine:
                success = self.engine.reload_camera_configs()
                if success:
                    logger.info("Camera configs reloaded")
                else:
                    logger.warning("Camera reload returned False")
            else:
                # Engine not initialized yet — signal main thread to init
                logger.info("Camera command received — signalling engine init")
                self._camera_ready.set()

        elif command_type == 'StopCamera':
            logger.info(f"StopCamera for camera {camera_id}")
            if self.engine:
                self.engine.reload_camera_configs(allow_empty=True)

        elif command_type == 'CaptureFrame':
            command_id = payload.get('command_id')
            frame_index = int(payload.get('frame_index', 1))
            undistort = bool(payload.get('undistort', False))
            camera_matrix = payload.get('camera_matrix')
            dist_coeffs = payload.get('dist_coeffs')
            calibration_model = payload.get('calibration_model', 'fisheye')
            logger.info(f"CaptureFrame for camera {camera_id}, command {command_id}, frame_index={frame_index}, undistort={undistort}")
            if self.engine:
                self.engine.capture_frame(
                    camera_id,
                    command_id,
                    frame_index,
                    undistort=undistort,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    calibration_model=calibration_model,
                )
            else:
                logger.warning("Engine not available for frame capture")

        elif command_type == 'CalibrateCamera':
            command_id = payload.get('command_id')
            logger.info(f"CalibrateCamera for camera {camera_id}, command {command_id}")
            if self.engine:
                self.engine.calibrate_camera(camera_id, command_id)
            else:
                logger.warning("Engine not available for calibration")

        elif command_type == 'TestCalibration':
            command_id = payload.get('command_id')
            camera_matrix = payload.get('camera_matrix')
            dist_coeffs = payload.get('dist_coeffs')
            model = payload.get('model', 'fisheye')
            logger.info(f"TestCalibration for camera {camera_id}, command {command_id}")
            if self.engine:
                self.engine.test_calibration(
                    camera_id, command_id, camera_matrix, dist_coeffs, model
                )
            else:
                logger.warning("Engine not available for test calibration")

        elif command_type == 'ComputeHomography':
            command_id = payload.get('command_id')
            src_pts = payload.get('src_pts') or []
            dst_pts = payload.get('dst_pts') or []
            logger.info(
                f"ComputeHomography for camera {camera_id}, command {command_id}, "
                f"{len(src_pts)} point pairs"
            )
            if self.engine:
                self.engine.compute_homography(camera_id, command_id, src_pts, dst_pts)
            else:
                logger.warning("Engine not available for compute_homography")


# ============================================================
# Model Warm-up
# ============================================================

def _warm_up_models(models) -> None:
    """Eagerly load the models this process actually runs.

    Replaces `ModelFactory.initialize_all()`, which touches every model —
    including `person_detector` and `face_detector`, both of which belong to
    their own Celery workers. Since ModelFactory's properties are lazy, the
    omissions below are the whole mechanism: never touching those two in
    this process means neither model loads in it.

    What is still loaded here genuinely runs here: `face_matcher` does numpy
    similarity against the embedding gallery (no GPU model),
    `global_track_manager` owns the cross-camera ReID state, and
    `action_recognizer` talks to Ollama over HTTP.

    Note this does NOT make the process face-model-free: EmbeddingSyncService
    builds its own FaceDetector for the startup user sync (engine.py). That
    copy is off the per-frame path but stays resident — see the plan's
    decision #5.
    """
    logger.info(
        "Warming up main-process models "
        "(YOLO and face inference excluded — they run in their own workers)..."
    )
    if models.embedding_provider is not None:
        _ = models.face_matcher
    _ = models.action_recognizer
    _ = models.global_track_manager
    _ = models.global_id_generator
    logger.info("Main-process models ready")


# ============================================================
# Signal Handling (No Global State)
# ============================================================

def create_signal_handler() -> Callable:
    """Create signal handler that uses ApplicationLifecycle.

    Returns:
        Signal handler function
    """
    def signal_handler(signum: int, frame) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.info(f"Received signal {signal_name} ({signum}), shutting down...")

        lifecycle = ApplicationLifecycle.get_instance()
        lifecycle.shutdown()
        sys.exit(0)

    return signal_handler


def main() -> None:
    """Main entry point for SmartOfficeEngine.

    Initializes and runs the SmartOffice system with proper
    lifecycle management and signal handling.
    """
    # Get lifecycle manager
    lifecycle = ApplicationLifecycle.get_instance()

    # Register signal handlers (no global state needed)
    handler = create_signal_handler()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # Initialize application config
    config = init_smart_office_app()

    client_slug = settings.client_slug
    if not client_slug:
        logger.error("SO_CLIENT_SLUG environment variable is required")
        sys.exit(1)

    # Log startup information
    log_startup_info(client_slug=client_slug)

    # Start MDA first so camera commands can be received while waiting for cameras
    mda_manager = MDAManager(client_slug)
    lifecycle.register_shutdown_callback(mda_manager.stop)
    atexit.register(lifecycle.shutdown)
    mda_manager.start()

    # Load ML models once — reused across all engine reinitializations
    models = ModelFactory(
        build_vision_config(config),
        embedding_provider=PgVectorStore(client_slug),
    )
    _warm_up_models(models)
    lifecycle.register_shutdown_callback(models.cleanup)

    # Initialize engine — if no cameras yet, wait for a camera command via MDA
    engine = None
    while lifecycle.is_running and engine is None:
        try:
            engine = SmartOfficeEngine(
                client_slug=client_slug,
                applications=['attendance'],
                model_factory=models,
                **config
            )
        except ValueError:
            logger.info("No cameras configured yet — waiting for camera")
            mda_manager._camera_ready.wait()
            mda_manager._camera_ready.clear()

    if engine is None:
        return

    mda_manager.set_engine(engine)

    # Outer loop: handles engine restarts when camera set changes
    while lifecycle.is_running:
        shutdown_cb = engine.stop if hasattr(engine, 'stop') else lambda: None
        lifecycle.register_shutdown_callback(shutdown_cb)

        try:
            engine.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            break
        except Exception as e:
            logger.exception(f"Fatal error: {e}")
            raise

        # engine.run() returned — check if it was due to a camera set change
        if not engine.needs_reinit:
            break

        logger.info("Reinitializing engine with updated camera set...")
        engine = None
        mda_manager.set_engine(None)  # clear stale ref so next camera command signals _camera_ready
        while lifecycle.is_running and engine is None:
            try:
                engine = SmartOfficeEngine(
                    client_slug=client_slug,
                    applications=['attendance'],
                    model_factory=models,
                    **config
                )
            except ValueError:
                logger.info("No cameras configured — waiting for camera")
                mda_manager._camera_ready.wait()
                mda_manager._camera_ready.clear()
            except Exception as e:
                logger.exception(f"Failed to reinitialize engine: {e}")
                break

        if engine is None:
            break

        mda_manager.set_engine(engine)

    lifecycle.shutdown()


if __name__ == "__main__":
    main()
