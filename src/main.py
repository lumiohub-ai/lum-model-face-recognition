"""
Entry point for SmartOfficeEngine - Unified person tracking and face recognition.

Uses Pure Message-Driven Architecture (MDA) via Redis.

Architecture:
- MDAManager: Handles Redis pub/sub and stream consumption
- SmartOfficeEngine: Core video processing and face recognition
- Lifecycle management with proper signal handling
"""

import os
import sys
import signal
import atexit
import threading
import json
import weakref
from pathlib import Path
from typing import Optional, Callable
from contextlib import contextmanager

import redis
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import init_smart_office_app, log_startup_info
from engine import SmartOfficeEngine
from messaging import RedisClient, StreamConsumer
from messaging.channels import INTERNAL_CHANNELS
from messaging.redis_config import REDIS_HOST, REDIS_PORT


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
                logger.error(f"[Lifecycle] Shutdown callback error: {e}")

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

    def set_engine(self, engine: SmartOfficeEngine) -> None:
        """Set the engine reference for reload operations."""
        self.engine = engine

    def start(self) -> None:
        """Initialize and start all MDA components."""
        if self._running:
            logger.warning("[MDA] Already running")
            return

        # Verify Redis connection
        if not RedisClient().is_connected():
            logger.error("[MDA] Redis not available")
            sys.exit(1)

        # Start stream consumer for Backend commands
        self.stream_consumer = StreamConsumer()
        self.stream_consumer.set_camera_handler(self._handle_camera_command)
        self.stream_consumer.start()
        logger.info("[MDA] StreamConsumer started")

        # Start internal reload listeners
        self._start_reload_listeners()
        self._running = True
        logger.info("[MDA] All components started")

    def stop(self) -> None:
        """Gracefully shutdown MDA components."""
        if self.stream_consumer:
            logger.info("[MDA] Stopping stream consumer...")
            self.stream_consumer.stop()
        self._running = False
        logger.info("[MDA] Shutdown complete")

    def _start_reload_listeners(self) -> None:
        """Start background listeners for internal reload notifications."""
        def create_listener(channel: str, handler):
            def listener():
                r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
                pubsub = r.pubsub()
                pubsub.subscribe(channel)
                logger.info(f"[MDA] Subscribed to {channel}")

                for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            # Handle commands for ALL tenants (multi-tenant support)
                            handler(data)
                        except Exception as e:
                            logger.error(f"[MDA] Error in {channel}: {e}")

            thread = threading.Thread(target=listener, daemon=True)
            thread.start()
            return thread

        # Embedding reload listener
        create_listener(
            INTERNAL_CHANNELS['EMBEDDING_RELOAD'],
            self._handle_embedding_reload
        )

        # Status reload listener
        create_listener(
            INTERNAL_CHANNELS['STATUS_RELOAD'],
            self._handle_status_reload
        )

    def _handle_embedding_reload(self, data: dict) -> None:
        """Handle embedding reload notification."""
        logger.info(f"[MDA] Embedding reload: {data}")
        if self.engine:
            self.engine.reload_embeddings()
            logger.info("[MDA] Embeddings reloaded")
        else:
            logger.warning("[MDA] Engine not available")

    def _handle_status_reload(self, data: dict) -> None:
        """Handle status reload notification."""
        logger.info(f"[MDA] Status reload: {data}")
        if self.engine and hasattr(self.engine, 'entry_logger'):
            self.engine.entry_logger.reload_status()
            logger.info("[MDA] Status reloaded")
        else:
            logger.warning("[MDA] Engine not available")

    def _handle_camera_command(self, command_type: str, client_slug: str, payload: dict) -> None:
        """Handle camera config commands from Backend (multi-tenant)."""
        camera_id = payload.get('camera_id')
        logger.info(f"[MDA] Camera command: {command_type} for {client_slug} camera_id={camera_id}")

        # Process commands for ALL tenants (multi-tenant support)
        if command_type in ('ConfigureCamera', 'StartCamera'):
            if self.engine:
                success = self.engine.reload_camera_configs()
                if success:
                    logger.info("[MDA] Camera configs reloaded")
                else:
                    logger.warning("[MDA] Camera reload returned False")
            else:
                logger.warning("[MDA] Engine not available")

        elif command_type == 'StopCamera':
            logger.info(f"[MDA] StopCamera for camera {camera_id}")


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
    config = init_smart_office_app(
        env_file=project_root.parent / '.env',
    )

    client_slug = os.getenv('HB_CLIENTSLUG')
    if not client_slug:
        logger.error("HB_CLIENTSLUG environment variable is required")
        sys.exit(1)

    # Log startup information
    log_startup_info(client_slug=client_slug)

    # Initialize SmartOfficeEngine
    engine = SmartOfficeEngine(
        client_slug=client_slug,
        applications=['attendance'],
        **config
    )

    # Initialize MDA manager
    mda_manager = MDAManager(client_slug)
    mda_manager.set_engine(engine)

    # Register shutdown callbacks (in order of dependency)
    lifecycle.register_shutdown_callback(engine.stop if hasattr(engine, 'stop') else lambda: None)
    lifecycle.register_shutdown_callback(mda_manager.stop)

    # Also register with atexit for non-signal exits
    atexit.register(lifecycle.shutdown)

    try:
        # Start MDA and run engine
        mda_manager.start()
        engine.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        lifecycle.shutdown()


if __name__ == "__main__":
    main()
