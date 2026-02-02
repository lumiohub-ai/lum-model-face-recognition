"""
Messaging Layer (MDA Core)

Pure Message-Driven Architecture implementation using Redis Pub/Sub.
Handles all communication between AI Service and Backend.
"""

from .redis_client import RedisClient, get_redis_client
from .publisher import MDAPublisher
from .subscriber import MDASubscriber

__all__ = [
    "RedisClient",
    "get_redis_client",
    "MDAPublisher",
    "MDASubscriber",
]
