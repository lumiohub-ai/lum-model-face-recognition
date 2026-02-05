"""
Redis Stream Consumer (MDA)

Consumes commands from Redis Streams and dispatches to Celery tasks.
Commands are reliable - persisted until acknowledged.

Commands: Backend → AI Service (Redis Streams)
Events: AI Service → Backend (Redis Pub/Sub)
"""

import os
import json
import time
import threading
import logging
from typing import Dict, Any, Optional, Callable
import redis

from workers.embedding_tasks import (
    process_add_user,
    process_update_user,
    process_delete_user,
)

logger = logging.getLogger(__name__)

# Redis connection settings
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_URL = os.getenv('REDIS_URL', f'redis://{REDIS_HOST}:{REDIS_PORT}')

# Stream names
STREAMS = {
    'EMBEDDING': 'commands:embedding',
    'CAMERA': 'commands:camera',
}

# Consumer group name
CONSUMER_GROUP = 'ai-service-group'

# Idempotency cache TTL (24 hours)
IDEMPOTENCY_TTL = 86400


class StreamConsumer:
    """
    Consume commands from Redis Streams and dispatch to Celery tasks.

    Features:
    - Consumer groups for reliability
    - Automatic acknowledgment on success
    - Idempotency checking
    - Graceful shutdown
    """

    def __init__(self, consumer_name: str = None):
        """
        Initialize the stream consumer.

        Args:
            consumer_name: Unique name for this consumer (default: hostname)
        """
        self.redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        self.consumer_name = consumer_name or os.getenv('HOSTNAME', f'ai-consumer-{os.getpid()}')
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._camera_handler: Optional[Callable] = None

        logger.info(f"[StreamConsumer] Initialized: {self.consumer_name}")

    def set_camera_handler(self, handler: Callable[[str, str, Dict], None]):
        """
        Set handler for camera commands.

        Args:
            handler: Function(command_type, client_slug, camera_data)
        """
        self._camera_handler = handler

    def _ensure_consumer_groups(self):
        """Create consumer groups if they don't exist."""
        for name, stream in STREAMS.items():
            try:
                self.redis.xgroup_create(stream, CONSUMER_GROUP, id='0', mkstream=True)
                logger.info(f"[StreamConsumer] Created consumer group for {stream}")
            except redis.ResponseError as e:
                if 'BUSYGROUP' not in str(e):
                    raise
                # Group already exists, that's fine
                logger.debug(f"[StreamConsumer] Consumer group already exists for {stream}")

    def start(self):
        """Start consuming from all streams in a background thread."""
        if self._running:
            logger.warning("[StreamConsumer] Already running")
            return

        self._ensure_consumer_groups()
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        logger.info("[StreamConsumer] Started consuming from Redis Streams")

    def _consume_loop(self):
        """Main consumption loop."""
        streams = {stream: '>' for stream in STREAMS.values()}

        while self._running:
            try:
                # XREADGROUP - read new messages from all streams
                # Block for 1 second, read up to 10 messages
                results = self.redis.xreadgroup(
                    CONSUMER_GROUP,
                    self.consumer_name,
                    streams,
                    count=10,
                    block=1000
                )

                if results:
                    for stream_name, messages in results:
                        for message_id, data in messages:
                            self._process_message(stream_name, message_id, data)

            except redis.ConnectionError as e:
                logger.error(f"[StreamConsumer] Redis connection error: {e}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"[StreamConsumer] Error in consume loop: {e}")
                time.sleep(0.5)

    def _process_message(self, stream: str, message_id: str, data: Dict[str, str]):
        """
        Process a single message and dispatch to appropriate handler.

        Args:
            stream: Stream name
            message_id: Redis message ID
            data: Message data dict
        """
        command_id = data.get('command_id', 'unknown')
        command_type = data.get('command_type', '')
        idempotency_key = data.get('idempotency_key', '')
        payload_str = data.get('payload', '{}')

        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            logger.error(f"[StreamConsumer] Invalid JSON payload: {payload_str}")
            self._ack(stream, message_id)
            return

        # Idempotency check
        if self._is_duplicate(idempotency_key):
            logger.warning(f"[StreamConsumer] Duplicate command ignored: {idempotency_key}")
            self._ack(stream, message_id)
            return

        logger.info(f"[StreamConsumer] Processing {command_type}", {
            'command_id': command_id,
            'stream': stream,
            'message_id': message_id,
        })

        try:
            # Dispatch based on stream
            if stream == STREAMS['EMBEDDING']:
                self._dispatch_embedding_command(command_id, command_type, payload)
            elif stream == STREAMS['CAMERA']:
                self._dispatch_camera_command(command_id, command_type, payload)
            else:
                logger.warning(f"[StreamConsumer] Unknown stream: {stream}")

            # Mark as processed
            self._mark_processed(idempotency_key)

            # Acknowledge the message
            self._ack(stream, message_id)

        except Exception as e:
            logger.error(f"[StreamConsumer] Error processing {command_type}: {e}")
            # Don't ACK - message will be redelivered

    def _dispatch_embedding_command(self, command_id: str, command_type: str, payload: Dict):
        """Dispatch embedding commands to Celery tasks."""
        client_slug = payload.get('client_slug')
        user_id = payload.get('user_id')
        full_name = payload.get('full_name', 'Unknown')
        image_urls = payload.get('image_urls', [])

        logger.info(f"[StreamConsumer] Dispatching {command_type} for user {user_id}")

        if command_type == 'CreateEmbedding':
            process_add_user.delay(
                command_id,
                client_slug,
                {
                    'id': user_id,
                    'full_name': full_name,
                    'image_urls': image_urls,
                }
            )
        elif command_type == 'UpdateEmbedding':
            process_update_user.delay(
                command_id,
                client_slug,
                {
                    'id': user_id,
                    'full_name': full_name,
                    'image_urls': image_urls,
                }
            )
        elif command_type == 'DeleteEmbedding':
            process_delete_user.delay(
                command_id,
                client_slug,
                {'id': user_id}
            )
        else:
            logger.warning(f"[StreamConsumer] Unknown embedding command: {command_type}")

    def _dispatch_camera_command(self, command_id: str, command_type: str, payload: Dict):
        """Dispatch camera commands to handler."""
        logger.info(f"[StreamConsumer] Dispatching camera command: {command_type}")

        if not self._camera_handler:
            logger.warning("[StreamConsumer] No camera handler set")
            return

        client_slug = payload.get('client_slug')

        try:
            self._camera_handler(command_type, client_slug, payload)
            logger.debug("[StreamConsumer] Camera handler completed successfully")
        except Exception as e:
            logger.error(f"[StreamConsumer] Camera handler error: {e}")

    def _is_duplicate(self, idempotency_key: str) -> bool:
        """Check if command was already processed."""
        if not idempotency_key:
            return False
        key = f"idempotency:{idempotency_key}"
        return bool(self.redis.exists(key))

    def _mark_processed(self, idempotency_key: str):
        """Mark command as processed."""
        if not idempotency_key:
            return
        key = f"idempotency:{idempotency_key}"
        self.redis.setex(key, IDEMPOTENCY_TTL, '1')

    def _ack(self, stream: str, message_id: str):
        """Acknowledge a message."""
        try:
            self.redis.xack(stream, CONSUMER_GROUP, message_id)
            logger.debug(f"[StreamConsumer] ACKed {message_id} on {stream}")
        except Exception as e:
            logger.error(f"[StreamConsumer] Failed to ACK {message_id}: {e}")

    def stop(self):
        """Stop the consumer gracefully."""
        logger.info("[StreamConsumer] Stopping...")
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        try:
            self.redis.close()
        except Exception as e:
            logger.error(f"[StreamConsumer] Error closing Redis: {e}")

        logger.info("[StreamConsumer] Stopped")

    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running and self._thread is not None and self._thread.is_alive()


# Singleton instance
_stream_consumer: Optional[StreamConsumer] = None


def get_stream_consumer() -> StreamConsumer:
    """Get or create the singleton stream consumer."""
    global _stream_consumer
    if _stream_consumer is None:
        _stream_consumer = StreamConsumer()
    return _stream_consumer


def start_stream_consumer():
    """Start the stream consumer."""
    consumer = get_stream_consumer()
    consumer.start()
    return consumer


def stop_stream_consumer():
    """Stop the stream consumer."""
    global _stream_consumer
    if _stream_consumer:
        _stream_consumer.stop()
        _stream_consumer = None
