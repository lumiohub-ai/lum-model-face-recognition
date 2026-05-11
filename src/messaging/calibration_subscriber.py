"""Subscribes to events:calibration and invalidates the homography cache.

The AI service is both publisher and consumer of this channel: when its own
_do_compute_homography publishes a HomographyCalibrated event, this listener
clears the matching (client_slug, camera_id) registry entry so the next frame's
position-emit re-queries Postgres (DB is the single source of truth — also
benefits future admin-paste paths once backend adds CalibrationPersisted).
"""

import json
import threading
import time

from loguru import logger

from domain.calibration.homography_registry import HomographyRegistry
from messaging.channels import EVENT_CHANNELS, EVENT_TYPES
from messaging.redis_client import RedisClient


class CalibrationSubscriber(threading.Thread):
    def __init__(self, registry: HomographyRegistry):
        super().__init__(daemon=True, name="calibration-subscriber")
        self._registry = registry
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        retry_delay = 1
        channel = EVENT_CHANNELS["CALIBRATION"]
        while not self._stop.is_set():
            try:
                pubsub = RedisClient.get_instance().client.pubsub()
                pubsub.subscribe(channel)
                logger.debug(f"Subscribed to {channel} for homography invalidation")
                retry_delay = 1

                for message in pubsub.listen():
                    if self._stop.is_set():
                        return
                    if message.get("type") != "message":
                        continue
                    try:
                        data = json.loads(message["data"])
                    except (TypeError, ValueError) as e:
                        logger.warning(f"[{channel}] bad payload: {e}")
                        continue

                    if data.get("event_type") != EVENT_TYPES["HOMOGRAPHY_CALIBRATED"]:
                        continue

                    slug = data.get("client_slug")
                    camera_id = data.get("camera_id")
                    if slug is None or camera_id is None:
                        logger.warning(
                            f"[{channel}] HomographyCalibrated missing "
                            f"client_slug/camera_id: {data}"
                        )
                        continue
                    try:
                        camera_id = int(camera_id)
                    except (TypeError, ValueError):
                        logger.warning(
                            f"[{channel}] non-int camera_id: {camera_id!r}"
                        )
                        continue

                    self._registry.invalidate(slug, camera_id)

            except Exception as e:
                if self._stop.is_set():
                    return
                logger.warning(
                    f"[{channel}] Redis disconnected: {e} — retrying in {retry_delay}s"
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
