"""
Messaging Layer (MDA Core)

Message-Driven Architecture implementation:
- Commands: Backend -> AI (Redis Streams) - handled by StreamConsumer
- Events: AI -> Backend (Redis Pub/Sub) - handled by MDAPublisher
"""

from .redis_client import CentralRedisClient, RedisClient
from .publisher import MDAPublisher
from .stream_consumer import StreamConsumer

__all__ = [
    "RedisClient",
    "CentralRedisClient",
    "MDAPublisher",
    "StreamConsumer",
]
