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
from messaging import MDAPublisher, get_redis_client, start_stream_consumer, stop_stream_consumer, get_stream_consumer

# MDA components
_stream_consumer = None
_engine = None
_embedding_reload_thread = None


def start_embedding_reload_listener(client_slug: str):
    """Start a background thread to listen for embedding reload notifications."""
    import threading
    import json
    import redis

    global _embedding_reload_thread

    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))

    def listener():
        from messaging.channels import INTERNAL_CHANNELS
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        pubsub = r.pubsub()
        pubsub.subscribe(INTERNAL_CHANNELS['EMBEDDING_RELOAD'])

        print(f"[MDA] Embedding reload listener started - subscribed to {INTERNAL_CHANNELS['EMBEDDING_RELOAD']}")

        for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    msg_client_slug = data.get('client_slug')

                    # Only reload if this message is for our client
                    if msg_client_slug == client_slug:
                        print(f"[MDA] Embedding reload notification received: {data}")
                        if _engine:
                            _engine.reload_embeddings()
                            print(f"[MDA] Embeddings reloaded successfully")
                        else:
                            print("[MDA] Engine not available for embedding reload")
                except Exception as e:
                    print(f"[MDA] Error handling embedding reload: {e}")

    _embedding_reload_thread = threading.Thread(target=listener, daemon=True)
    _embedding_reload_thread.start()


def handle_camera_config_command(command_type: str, client_slug: str, payload: dict):
    """
    Handle camera config commands from Backend.

    Commands: ConfigureCamera, StartCamera, StopCamera
    """
    global _engine

    try:
        camera_id = payload.get('camera_id')
        current_client = os.getenv('HB_CLIENTSLUG')

        print(f"[MDA] Camera command received: {command_type}")
        print(f"[MDA]   - camera_id: {camera_id}")
        print(f"[MDA]   - client_slug from command: {client_slug}")
        print(f"[MDA]   - HB_CLIENTSLUG env: {current_client}")

        # Only process if this is for our client
        if client_slug != current_client:
            print(f"[MDA] Ignoring camera command for different client: {client_slug} (expected: {current_client})")
            return

        # Handle different command types
        if command_type in ('ConfigureCamera', 'StartCamera'):
            # Reload camera configs
            if _engine:
                success = _engine.reload_camera_configs()
                if success:
                    print(f"[MDA] Camera configurations reloaded successfully")
                else:
                    print(f"[MDA] Camera configuration reload returned False")
            else:
                print("[MDA] Engine not available for camera config reload")

        elif command_type == 'StopCamera':
            # Stop specific camera
            if _engine:
                # TODO: Implement stop_camera method in engine
                print(f"[MDA] StopCamera command received for camera {camera_id}")
            else:
                print("[MDA] Engine not available for camera stop")

    except Exception as e:
        print(f"[MDA] Error handling camera command: {e}")


def init_mda(client_slug: str, embedding_sync_service=None, engine=None):
    """
    Initialize MDA (Message-Driven Architecture) components.

    Uses:
    - StreamConsumer: Reads commands from Redis Streams (Backend → AI)
    - MDAPublisher: Publishes events to Redis Pub/Sub (AI → Backend)
    - Celery: Processes heavy tasks (embeddings, etc.)

    Args:
        client_slug: Organization identifier
        embedding_sync_service: Optional embedding sync service (not used - Celery handles it)
        engine: SmartOfficeEngine instance for camera config reload
    """
    global _stream_consumer, _engine

    # Store engine reference for camera config reload
    _engine = engine

    try:
        # Test Redis connection
        redis_client = get_redis_client()
        if not redis_client.is_connected():
            print("[MDA] Redis not available - MDA requires Redis to function")
            sys.exit(1)

        # Initialize stream consumer for commands from Backend
        _stream_consumer = get_stream_consumer()

        # Set up camera config change handler
        _stream_consumer.set_camera_handler(handle_camera_config_command)
        print(f"[MDA] Camera config handler configured for {client_slug}")

        # Start stream consumer
        # Note: Embedding commands are dispatched to Celery automatically by StreamConsumer
        _stream_consumer.start()
        print(f"[MDA] StreamConsumer started - listening for commands on Redis Streams")
        print(f"[MDA] Celery workers will process embedding tasks")

        # Start embedding reload listener
        start_embedding_reload_listener(client_slug)
        print(f"[MDA] Embedding reload listener started")

    except ImportError as e:
        print(f"[MDA] Import error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[MDA] Failed to initialize: {e}")
        sys.exit(1)


def shutdown_mda():
    """Gracefully shutdown MDA components."""
    global _stream_consumer

    if _stream_consumer:
        print("[MDA] Shutting down stream consumer...")
        _stream_consumer.stop()
        print("[MDA] Stream consumer stopped")


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
