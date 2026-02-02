"""
Entry point for SmartOfficeEngine - Unified person tracking and face recognition.

Uses Pure Message-Driven Architecture (MDA) via Redis Pub/Sub.
"""

import os
import sys
import signal
import atexit
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import init_smart_office_app, log_startup_info
from engine import SmartOfficeEngine
from messaging import MDASubscriber, MDAPublisher, get_redis_client
from messaging.handlers import EmbeddingRequestHandler

# MDA components
_mda_subscriber = None
_engine = None


def handle_camera_config_change(message: dict):
    """Handle camera config change messages from backend."""
    global _engine

    try:
        action = message.get('action')
        client_slug = message.get('client_slug')
        camera_data = message.get('camera_data', {})

        print(f"[MDA] Camera config change: {action} for camera {camera_data.get('id')} (org: {client_slug})")

        # Only reload if this is for our client
        current_client = os.getenv('HB_CLIENTSLUG')
        if client_slug != current_client:
            print(f"[MDA] Ignoring camera config change for different client: {client_slug}")
            return

        # Reload camera configs
        if _engine:
            success = _engine.reload_camera_configs()
            if success:
                print(f"[MDA] Camera configurations reloaded successfully")
            else:
                print(f"[MDA] Camera configuration reload returned False")
        else:
            print("[MDA] Engine not available for camera config reload")

    except Exception as e:
        print(f"[MDA] Error handling camera config change: {e}")


def init_mda(client_slug: str, embedding_sync_service=None, engine=None):
    """
    Initialize MDA (Message-Driven Architecture) components.

    Args:
        client_slug: Organization identifier
        embedding_sync_service: Optional embedding sync service for handling requests
        engine: SmartOfficeEngine instance for camera config reload
    """
    global _mda_subscriber, _engine

    # Store engine reference for camera config reload
    _engine = engine

    try:
        # Test Redis connection
        redis_client = get_redis_client()
        if not redis_client.is_connected():
            print("[MDA] Redis not available - MDA requires Redis to function")
            sys.exit(1)

        # Initialize subscriber
        _mda_subscriber = MDASubscriber()

        # Set up embedding request handler if embedding sync is available
        if embedding_sync_service:
            publisher = MDAPublisher(client_slug)
            handler = EmbeddingRequestHandler(embedding_sync_service, publisher)
            _mda_subscriber.set_embedding_handler(handler.handle)
            print(f"[MDA] Embedding request handler configured for {client_slug}")

        # Set up camera config change handler
        _mda_subscriber.set_camera_config_handler(handle_camera_config_change)
        print(f"[MDA] Camera config handler configured for {client_slug}")

        # Start subscriber
        _mda_subscriber.start()
        print(f"[MDA] Subscriber started - listening for messages")

    except ImportError as e:
        print(f"[MDA] Import error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[MDA] Failed to initialize: {e}")
        sys.exit(1)


def shutdown_mda():
    """Gracefully shutdown MDA components."""
    global _mda_subscriber

    if _mda_subscriber:
        print("[MDA] Shutting down subscriber...")
        _mda_subscriber.stop()
        print("[MDA] Subscriber stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\nReceived signal {signum}, shutting down...")
    shutdown_mda()
    sys.exit(0)


def main():
    """Main entry point for SmartOfficeEngine."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(shutdown_mda)

    # Initialize application
    config = init_smart_office_app(
        env_file=project_root.parent / '.env',
    )

    client_slug = os.getenv('HB_CLIENTSLUG')

    # Log startup information
    log_startup_info(
        client_slug=client_slug,
        api_host=os.getenv('API_HOST'),
    )

    # Initialize SmartOfficeEngine
    engine = SmartOfficeEngine(
        email=os.getenv("SA_EMAIL"),
        password=os.getenv("SA_PASSWORD"),
        client_slug=client_slug,
        api_host=os.getenv("API_HOST"),
        applications=['attendance'],
        **config
    )

    # Initialize MDA
    embedding_sync = getattr(engine, 'embedding_sync', None)
    init_mda(client_slug, embedding_sync, engine)

    # Run the engine
    engine.run()


if __name__ == "__main__":
    main()
