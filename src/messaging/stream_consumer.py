"""
Redis Stream Consumer (MDA)

Consumes commands from Redis Streams and dispatches to Celery tasks.
Commands are reliable - persisted until acknowledged.

Commands: Backend → AI Service (Redis Streams)
Events: AI Service → Backend (Redis Pub/Sub)

Features:
- Consumer groups for reliable delivery
- Idempotency checking to prevent duplicate processing
- Dead-Letter Queue for invalid/unparseable messages
- Graceful shutdown with message acknowledgment
"""

import json
import time
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import redis
from loguru import logger

from config.settings import settings
from .channels import COMMAND_STREAMS

# NOTE: Worker task imports are done lazily in _dispatch_embedding_command
# to avoid circular imports with celery_app.py


# Consumer group name
CONSUMER_GROUP = 'ai-service-group'

# Idempotency cache TTL (24 hours)
IDEMPOTENCY_TTL = 86400

# DLQ TTL (7 days)
DLQ_TTL = 7 * 24 * 3600


class MessageDLQ:
    """Dead-Letter Queue for invalid or failed messages.

    Stores messages that cannot be processed for later inspection and retry.
    """

    DLQ_KEY_PREFIX = 'dlq:stream:'

    def __init__(self, redis_client: redis.Redis):
        """Initialize DLQ with Redis client.

        Args:
            redis_client: Redis connection
        """
        self.redis = redis_client

    def send(self, stream: str, message_id: str, raw_data: Any,
             error: str, error_type: str = 'PARSE_ERROR') -> None:
        """Send a failed message to the Dead-Letter Queue.

        Args:
            stream: Original stream name
            message_id: Redis stream message ID
            raw_data: Original message data
            error: Error description
            error_type: Category of error (PARSE_ERROR, VALIDATION_ERROR, etc.)
        """
        dlq_key = f'{self.DLQ_KEY_PREFIX}{stream.replace(":", "_")}'

        dlq_entry = {
            'stream': stream,
            'message_id': message_id,
            'raw_data': str(raw_data)[:5000],  # Truncate large payloads
            'error': error,
            'error_type': error_type,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'consumer': settings.hostname,
        }

        try:
            # Store in Redis list (LPUSH for FIFO when consuming with RPOP)
            self.redis.lpush(dlq_key, json.dumps(dlq_entry))
            self.redis.expire(dlq_key, DLQ_TTL)

            # Also add to index for quick lookup
            self.redis.hset(f'{dlq_key}:index', message_id, json.dumps(dlq_entry))
            self.redis.expire(f'{dlq_key}:index', DLQ_TTL)

            logger.warning(
                f"[StreamConsumer] Message sent to DLQ: stream={stream} "
                f"message_id={message_id} error_type={error_type} error={error}"
            )
        except Exception as e:
            logger.error(f"[StreamConsumer] Failed to send to DLQ: {e}")

    def get_count(self, stream: str) -> int:
        """Get number of messages in DLQ for a stream.

        Args:
            stream: Stream name

        Returns:
            Number of messages in DLQ
        """
        dlq_key = f'{self.DLQ_KEY_PREFIX}{stream.replace(":", "_")}'
        try:
            return self.redis.llen(dlq_key)
        except Exception:
            return 0


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
        self.redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.consumer_name = consumer_name or settings.hostname
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._camera_handler: Optional[Callable] = None

        # Initialize Dead-Letter Queue
        self.dlq = MessageDLQ(self.redis)

        logger.info(f"[StreamConsumer] Initialized: {self.consumer_name}")

    def set_camera_handler(self, handler: Callable[[str, str, Dict], None]) -> None:
        """
        Set handler for camera commands.

        Args:
            handler: Function(command_type, client_slug, camera_data)
        """
        self._camera_handler = handler

    def _ensure_consumer_groups(self) -> None:
        """Create consumer groups if they don't exist."""
        for name, stream in COMMAND_STREAMS.items():
            try:
                self.redis.xgroup_create(stream, CONSUMER_GROUP, id='0', mkstream=True)
                logger.info(f"[StreamConsumer] Created consumer group for {stream}")
            except redis.ResponseError as e:
                if 'BUSYGROUP' not in str(e):
                    raise
                # Group already exists, that's fine
                logger.debug(f"[StreamConsumer] Consumer group already exists for {stream}")

    def start(self) -> None:
        """Start consuming from all streams in a background thread."""
        if self._running:
            logger.warning("[StreamConsumer] Already running")
            return

        self._ensure_consumer_groups()
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        logger.info("[StreamConsumer] Started consuming from Redis Streams")

    def _consume_loop(self) -> None:
        """Main consumption loop."""
        streams = {stream: '>' for stream in COMMAND_STREAMS.values()}

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

    def _process_message(self, stream: str, message_id: str, data: Dict[str, str]) -> None:
        """
        Process a single message and dispatch to appropriate handler.

        Invalid or unparseable messages are sent to DLQ, not silently dropped.

        Args:
            stream: Stream name
            message_id: Redis message ID
            data: Message data dict
        """
        command_id = data.get('command_id', 'unknown')
        command_type = data.get('command_type', '')
        idempotency_key = data.get('idempotency_key', '')
        payload_str = data.get('payload', '{}')

        # Step 1: Parse JSON payload
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError as e:
            # IMPORTANT: Send to DLQ instead of silently dropping
            self.dlq.send(
                stream=stream,
                message_id=message_id,
                raw_data=data,
                error=f"Invalid JSON payload: {e}",
                error_type='PARSE_ERROR'
            )
            # ACK to remove from pending (it's now in DLQ)
            self._ack(stream, message_id)
            return

        # Step 2: Validate required fields
        if not command_type:
            self.dlq.send(
                stream=stream,
                message_id=message_id,
                raw_data=data,
                error="Missing required field: command_type",
                error_type='VALIDATION_ERROR'
            )
            self._ack(stream, message_id)
            return

        # Step 3: Idempotency check
        if self._is_duplicate(idempotency_key):
            logger.debug(f"[StreamConsumer] Duplicate command ignored: {idempotency_key}")
            self._ack(stream, message_id)
            return

        logger.info(
            f"[StreamConsumer] Processing {command_type} "
            f"command_id={command_id} stream={stream} message_id={message_id}"
        )

        try:
            # Step 4: Dispatch based on stream
            if stream == COMMAND_STREAMS['EMBEDDING']:
                self._dispatch_embedding_command(command_id, command_type, payload)
            elif stream == COMMAND_STREAMS['CAMERA']:
                self._dispatch_camera_command(command_id, command_type, payload)
            else:
                logger.warning(f"[StreamConsumer] Unknown stream: {stream}")
                self.dlq.send(
                    stream=stream,
                    message_id=message_id,
                    raw_data=data,
                    error=f"Unknown stream: {stream}",
                    error_type='ROUTING_ERROR'
                )
                self._ack(stream, message_id)
                return

            # Step 5: Mark as processed and acknowledge
            self._mark_processed(idempotency_key)
            self._ack(stream, message_id)

        except Exception as e:
            # Log error with traceback
            tb_str = traceback.format_exc()
            logger.error(
                f"[StreamConsumer] Error processing {command_type}: {e}\n{tb_str}"
            )
            # Don't ACK - message will be redelivered by Redis
            # After multiple redeliveries, consider moving to DLQ manually

    def _dispatch_embedding_command(self, command_id: str, command_type: str, payload: Dict) -> None:
        """Dispatch embedding commands to Celery tasks.

        Uses lazy imports to avoid circular import with celery_app.py.
        """
        # Lazy import to avoid circular imports
        from workers.embedding_tasks import (
            process_add_user,
            process_update_user,
            process_delete_user,
        )

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

    def _dispatch_camera_command(self, command_id: str, command_type: str, payload: Dict) -> None:
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

    def _mark_processed(self, idempotency_key: str) -> None:
        """Mark command as processed."""
        if not idempotency_key:
            return
        key = f"idempotency:{idempotency_key}"
        self.redis.setex(key, IDEMPOTENCY_TTL, '1')

    def _ack(self, stream: str, message_id: str) -> None:
        """Acknowledge a message."""
        try:
            self.redis.xack(stream, CONSUMER_GROUP, message_id)
            logger.debug(f"[StreamConsumer] ACKed {message_id} on {stream}")
        except Exception as e:
            logger.error(f"[StreamConsumer] Failed to ACK {message_id}: {e}")

    def stop(self) -> None:
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
