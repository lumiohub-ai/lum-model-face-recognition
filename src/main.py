"""
Entry point for SmartOfficeEngine - Unified person tracking and face recognition.

Uses Pure Message-Driven Architecture (MDA) via Redis.
"""

import os
import sys
import signal
import atexit
import logging
import threading
import json
from pathlib import Path
from typing import Optional

import redis

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import init_smart_office_app, log_startup_info
from engine import SmartOfficeEngine
from messaging import RedisClient, StreamConsumer
from messaging.channels import INTERNAL_CHANNELS
from messaging.redis_config import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)


class MDAManager:
    """Manages MDA components without global state."""

    def __init__(self, client_slug: str):
        self.client_slug = client_slug
        self.engine: Optional[SmartOfficeEngine] = None
        self.stream_consumer = None
        self._running = False

    def set_engine(self, engine: SmartOfficeEngine):
        """Set the engine reference for reload operations."""
        self.engine = engine

    def start(self):
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

    def stop(self):
        """Gracefully shutdown MDA components."""
        if self.stream_consumer:
            logger.info("[MDA] Stopping stream consumer...")
            self.stream_consumer.stop()
        self._running = False
        logger.info("[MDA] Shutdown complete")

    def _start_reload_listeners(self):
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

    def _handle_embedding_reload(self, data: dict):
        """Handle embedding reload notification."""
        logger.info(f"[MDA] Embedding reload: {data}")
        if self.engine:
            self.engine.reload_embeddings()
            logger.info("[MDA] Embeddings reloaded")
        else:
            logger.warning("[MDA] Engine not available")

    def _handle_status_reload(self, data: dict):
        """Handle status reload notification."""
        logger.info(f"[MDA] Status reload: {data}")
        if self.engine and hasattr(self.engine, 'entry_logger'):
            self.engine.entry_logger.reload_status()
            logger.info("[MDA] Status reloaded")
        else:
            logger.warning("[MDA] Engine not available")

    def _handle_camera_command(self, command_type: str, client_slug: str, payload: dict):
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


# Application instance (module-level for signal handlers)
_app: Optional[MDAManager] = None


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    if _app:
        _app.stop()
    sys.exit(0)


def main():
    """Main entry point for SmartOfficeEngine."""
    global _app

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize application config
    config = init_smart_office_app(
        env_file=project_root.parent / '.env',
    )

    client_slug = os.getenv('HB_CLIENTSLUG')

    # Log startup information
    log_startup_info(client_slug=client_slug)

    # Initialize SmartOfficeEngine
    engine = SmartOfficeEngine(
        client_slug=client_slug,
        applications=['attendance'],
        **config
    )

    # Initialize MDA manager
    _app = MDAManager(client_slug)
    _app.set_engine(engine)
    atexit.register(_app.stop)

    # Start MDA and run engine
    _app.start()
    engine.run()


if __name__ == "__main__":
    main()
