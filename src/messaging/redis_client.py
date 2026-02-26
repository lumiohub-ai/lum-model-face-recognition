"""
Redis Client for MDA

Provides Redis Pub/Sub functionality for message-driven communication.
"""

import json
import threading
import logging
from typing import Callable, Dict, Any, Optional
import redis

from .redis_config import REDIS_HOST, REDIS_PORT, REDIS_DB

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis Pub/Sub client for MDA communication."""

    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        self.pubsub = self.client.pubsub()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()

        logger.info(f"[MDA] Redis client initialized: {REDIS_HOST}:{REDIS_PORT}")

    def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """
        Publish a message to a Redis channel.

        Args:
            channel: Channel name
            message: Message dict (will be JSON serialized)

        Returns:
            True if successful
        """
        try:
            message_str = json.dumps(message)
            num_subscribers = self.client.publish(channel, message_str)
            logger.info(f"[MDA] Published to {channel}: event_id={message.get('event_id', 'N/A')}, subscribers={num_subscribers}")
            return True
        except Exception as e:
            logger.error(f"[MDA] Failed to publish to {channel}: {e}")
            return False

    def subscribe(self, channel: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to a Redis channel with a handler.

        Args:
            channel: Channel name
            handler: Callback function that receives parsed message dict
        """
        with self._lock:
            self._handlers[channel] = handler
            self.pubsub.subscribe(**{channel: self._create_handler(handler)})
            logger.info(f"[MDA] Subscribed to {channel}")

    def _create_handler(self, handler: Callable) -> Callable:
        """Create a message handler wrapper."""
        def wrapper(message):
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    handler(data)
                except json.JSONDecodeError as e:
                    logger.error(f"[MDA] JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"[MDA] Handler error: {e}")
        return wrapper

    def start(self) -> None:
        """Start listening for messages in a background thread."""
        if self._running:
            logger.warning("[MDA] Subscriber already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        logger.info("[MDA] Redis subscriber started")

    def _listen(self) -> None:
        """Background listening loop."""
        while self._running:
            try:
                self.pubsub.get_message(timeout=1.0)
            except Exception as e:
                if self._running:
                    logger.error(f"[MDA] Listen error: {e}")

    def stop(self) -> None:
        """Stop the subscriber gracefully."""
        logger.info("[MDA] Stopping Redis subscriber...")
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

        try:
            self.pubsub.close()
            self.client.close()
        except Exception as e:
            logger.error(f"[MDA] Error closing Redis: {e}")

        logger.info("[MDA] Redis subscriber stopped")

    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        try:
            self.client.ping()
            return True
        except Exception:
            return False


