"""
MDA Subscriber

Subscribes to Backend messages via Redis Pub/Sub.
Handles embedding requests from Backend.
"""

import logging
from typing import Callable, Dict, Any, Optional

from face_recognition.mda.redis_client import get_redis_client

logger = logging.getLogger(__name__)

# Channels to subscribe to
CHANNELS = {
    'EMBEDDING_REQUESTS': 'embedding.requests',
    'CAMERA_CONFIG': 'camera.config',
}


class MDASubscriber:
    """
    Subscriber for receiving messages from Backend via Redis Pub/Sub.
    """

    def __init__(self):
        """Initialize the MDA subscriber."""
        self.redis = get_redis_client()
        self._handlers: Dict[str, Callable] = {}
        self._initialized = False

    def set_embedding_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Set the handler for embedding requests.

        Args:
            handler: Function to handle embedding request messages
        """
        self._handlers['embedding'] = handler

    def set_camera_config_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Set the handler for camera config changes.

        Args:
            handler: Function to handle camera config change messages
        """
        self._handlers['camera_config'] = handler

    def start(self):
        """Start listening for messages."""
        if self._initialized:
            logger.warning("[MDA] Subscriber already initialized")
            return

        # Subscribe to embedding requests
        if 'embedding' in self._handlers:
            self.redis.subscribe(
                CHANNELS['EMBEDDING_REQUESTS'],
                self._handlers['embedding']
            )
            logger.info(f"[MDA] Subscribed to {CHANNELS['EMBEDDING_REQUESTS']}")

        # Subscribe to camera config changes
        if 'camera_config' in self._handlers:
            self.redis.subscribe(
                CHANNELS['CAMERA_CONFIG'],
                self._handlers['camera_config']
            )
            logger.info(f"[MDA] Subscribed to {CHANNELS['CAMERA_CONFIG']}")

        # Start the Redis listener thread
        self.redis.start()
        self._initialized = True
        logger.info("[MDA] Subscriber started")

    def stop(self):
        """Stop listening for messages."""
        self.redis.stop()
        self._initialized = False
        logger.info("[MDA] Subscriber stopped")

    def is_running(self) -> bool:
        """Check if subscriber is running."""
        return self._initialized and self.redis.is_connected()
