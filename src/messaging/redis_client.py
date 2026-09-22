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
            decode_responses=True
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


class CentralRedisClient:
    """Singleton Redis client for the central pub/sub reload channels (LSO-189).

    Separate from RedisClient (which stays on the LOCAL Redis for the Celery
    broker + camera/slot leases): a branch AI needs a second, read-only
    connection to the CENTRAL Redis so it still hears the backend's
    enrollment-reload broadcasts. Points at settings.redis_host when
    SO_CENTRAL_REDIS_HOST is unset, so central/single-site deployments get one
    client object either way.
    """

    _instance: 'CentralRedisClient | None' = None
    _lock = threading.Lock()

    def __init__(self):
        self.client = redis.Redis(
            host=settings.central_redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )
        logger.info(f"Central Redis client initialized: {settings.central_redis_host}:{settings.redis_port}")

    @classmethod
    def get_instance(cls) -> 'CentralRedisClient':
        """Return the shared singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
