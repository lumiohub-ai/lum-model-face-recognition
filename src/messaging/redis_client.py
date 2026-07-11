"""
Redis Client for MDA

Singleton publish-only client shared across all components.
Backed by redis-py's built-in connection pool, so it is thread-safe.
"""

import json
import logging
import threading
from typing import Any, Dict

import redis

from config.settings import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Singleton Redis client for all publish/stream operations."""

    _instance: 'RedisClient | None' = None
    _lock = threading.Lock()

    def __init__(self):
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            # Without these, a half-dead connection (Redis overloaded, network
            # black-holing packets) makes publish()/ping() block on OS-level
            # TCP retransmission — which can be minutes, not seconds — and
            # publish() is called synchronously from the per-camera frame
            # processing loop, so that would stall video/tracking for that
            # camera the whole time.
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        logger.info(f"Redis client initialized: {settings.redis_host}:{settings.redis_port}")

    @classmethod
    def get_instance(cls) -> 'RedisClient':
        """Return the shared singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish a message to a Redis Pub/Sub channel."""
        try:
            message_str = json.dumps(message)
            num_subscribers = self.client.publish(channel, message_str)
            logger.info(f"Published to {channel}: event_id={message.get('event_id', 'N/A')}, subscribers={num_subscribers}")
            return True
        except Exception as e:
            logger.exception(f"Failed to publish to {channel}: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if Redis is reachable."""
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def stop(self) -> None:
        """Close the Redis connection."""
        try:
            self.client.close()
        except Exception as e:
            logger.exception(f"Error closing Redis: {e}")
