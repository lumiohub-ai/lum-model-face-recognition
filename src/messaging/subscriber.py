"""Background Redis pub/sub listener.

Extracted from main.py so the camera Celery workers can reuse it: they own
their own FaceMatcher, EntryLogger and CameraEngine, so they need the same
reload notifications the main process has always consumed.

(`calibration_subscriber.py` is a third copy of this loop. It filters on
event_type inside the loop and exposes stop(), so folding it in here would
mean widening this signature or changing its public surface — left alone
deliberately.)
"""

import json
import threading
import time

from loguru import logger

from messaging.redis_client import RedisClient


def start_listener(channel, handler, *, is_running=None, name=None):
    """Subscribe to `channel` on a daemon thread, calling `handler(data)`
    with each decoded JSON message.

    Args:
        channel: Redis pub/sub channel name.
        handler: Called with the decoded dict. Exceptions are logged and
            swallowed so one bad message cannot kill the subscription.
        is_running: Zero-arg predicate polled to decide whether to keep
            listening. Defaults to running forever, which is what the
            Celery workers want; main.py passes its own shutdown flag.
        name: Thread name, for debugging.

    Returns:
        The started daemon thread.
    """
    still_running = is_running if is_running is not None else lambda: True

    def listener():
        retry_delay = 1
        pubsub = None
        while still_running():
            try:
                pubsub = RedisClient.get_instance().client.pubsub()
                pubsub.subscribe(channel)
                logger.debug(f"Subscribed to {channel}")
                retry_delay = 1  # reset on successful connect

                for message in pubsub.listen():
                    if not still_running():
                        return
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            handler(data)
                        except Exception as e:
                            logger.exception(f"Error in {channel}: {e}")

            except Exception as e:
                if not still_running():
                    return
                logger.warning(
                    f"[{channel}] Redis disconnected: {e} — retrying in {retry_delay}s"
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
            finally:
                # Every iteration creates a fresh pubsub (a new connection);
                # without closing the previous one on retry, each reconnect
                # leaked the old connection/socket instead of relying on GC.
                if pubsub is not None:
                    try:
                        pubsub.close()
                    except Exception:
                        pass

    thread = threading.Thread(target=listener, daemon=True, name=name)
    thread.start()
    return thread
