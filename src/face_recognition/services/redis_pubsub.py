"""Redis Pub/Sub service for real-time embedding updates."""

import os
import redis
import json
import threading
from typing import Callable, Optional
from loguru import logger


class RedisPublisher:
    """Publisher for embedding update events."""

    def __init__(self):
        """Initialize Redis publisher."""
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))

        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Redis publisher connected to {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.redis_client = None

    def publish_embedding_update(self, client_slug: str, action: str, user_name: str) -> bool:
        """Publish embedding update event.

        Args:
            client_slug: Organization slug (e.g., 'humblebee')
            action: Action type ('add_user', 'update_user', 'delete_user')
            user_name: Name of the user affected

        Returns:
            True if published successfully, False otherwise
        """
        if not self.redis_client:
            logger.warning("Redis client not available, skipping publish")
            return False

        try:
            channel = f"embeddings:updates:{client_slug}"
            message = json.dumps({
                'action': action,
                'user_name': user_name,
                'client_slug': client_slug
            })

            subscribers = self.redis_client.publish(channel, message)
            logger.info(f"📢 Published {action} for {user_name} to {subscribers} subscriber(s)")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to publish to Redis: {e}")
            return False

    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis publisher closed")


class RedisSubscriber:
    """Subscriber for embedding update events."""

    def __init__(self, client_slug: str, on_update: Callable[[str, str], None]):
        """Initialize Redis subscriber.

        Args:
            client_slug: Organization slug to subscribe to
            on_update: Callback function(action, user_name) to call on updates
        """
        self.client_slug = client_slug
        self.on_update = on_update
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.pubsub = None
        self.redis_client = None

    def start(self):
        """Start subscribing to Redis channel in background thread."""
        if self.running:
            logger.warning("Redis subscriber already running")
            return

        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()

            self.pubsub = self.redis_client.pubsub()
            channel = f"embeddings:updates:{self.client_slug}"
            self.pubsub.subscribe(channel)

            self.running = True
            self.thread = threading.Thread(target=self._listen, daemon=True)
            self.thread.start()

            logger.info(f"✅ Redis subscriber started for channel: {channel}")

        except Exception as e:
            logger.error(f"❌ Failed to start Redis subscriber: {e}")
            self.running = False

    def _listen(self):
        """Listen for messages on subscribed channel."""
        logger.info(f"🔊 Listening for embedding updates on channel: embeddings:updates:{self.client_slug}")

        try:
            for message in self.pubsub.listen():
                if not self.running:
                    break

                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        action = data.get('action')
                        user_name = data.get('user_name')

                        logger.info(f"🔔 Received {action} event for {user_name}")

                        # Call the update callback
                        if self.on_update:
                            self.on_update(action, user_name)

                    except Exception as e:
                        logger.error(f"❌ Error processing message: {e}")

        except Exception as e:
            logger.error(f"❌ Redis listener error: {e}")
        finally:
            logger.info("Redis listener stopped")

    def stop(self):
        """Stop subscribing and close connection."""
        self.running = False

        if self.pubsub:
            try:
                self.pubsub.unsubscribe()
                self.pubsub.close()
            except Exception as e:
                logger.error(f"Error closing pubsub: {e}")

        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.error(f"Error closing Redis client: {e}")

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        logger.info("Redis subscriber stopped")
