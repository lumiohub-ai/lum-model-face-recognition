"""Cross-process stream-health and raw-frame-handle reporting for decode
workers.

Two things `SmartOfficeEngine` used to reach synchronously, in-process,
before decoding moved into its own process:

  - the dashboard's `read_ms`/`decode_ms`/`stream_state` gauges
    (`engine.py`'s `_register_pipeline_gauges`, calling `StreamHandler`
    methods directly), and
  - a live frame for calibration commands (`stream_manager.get_frame` /
    `get_fresh_frame`).

Both are bridged the same way: the decode worker publishes a small JSON
blob per camera into Redis, and the engine reads it back. The frame itself
never goes through Redis — only the `FrameHandle` needed to find it in the
`camraw_<id>` shared-memory ring (`workers/frame_store.py`'s
`RawFrameSlot`/`attach_and_read_raw`) travels this way, the same "handle
through the broker, pixels through shared memory" split used everywhere
else in this pipeline.

`CeleryCameraProducer` is still given `metrics_collector=None` from the
decode worker — that collector lives in the engine's process and cannot be
reached from here. Per-camera fps rides this same blob instead: the worker
samples the producer's own frame counter each tick and publishes a rate,
rather than shipping `record_frame`'s individual timestamps across a
process boundary to be re-derived on the other side.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Dict, Optional

import numpy as np
from loguru import logger

from workers.frame_store import FrameHandle, attach_and_read_raw

_KEY_PREFIX = "decode:health:cam:"

# Comfortably longer than the publish interval (SO_DECODE_HEALTH_INTERVAL_S,
# default 2s in decode_main.py), so a dead worker's entries expire instead
# of the dashboard/calibration path reading frozen, stale data forever.
_TTL_S = 15


def _key(camera_id: int) -> str:
    return f"{_KEY_PREFIX}{camera_id}"


class StreamHealthReporter:
    """Decode-side: publish one camera's health snapshot and (optionally)
    its latest raw-frame handle, in one write.

    Mirrors `RedisIdentityThrottle`'s degrade-on-failure posture: a Redis
    outage just means health stops updating (stale, then expired) rather
    than raising and disrupting the decode loop over a metrics
    side-channel.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._owns_client = redis_client is None

    def _client(self):
        if self._redis is not None:
            return self._redis
        from messaging.redis_client import RedisClient

        self._redis = RedisClient.get_instance().client
        return self._redis

    def publish(
        self,
        camera_id: int,
        read_ms: float,
        decode_ms: float,
        state: Optional[str],
        frame_handle: Optional[FrameHandle] = None,
        fps: Optional[float] = None,
    ) -> None:
        payload = {
            "read_ms": read_ms,
            "decode_ms": decode_ms,
            "state": state,
            "published_at": time.time(),
        }
        if fps is not None:
            payload["fps"] = fps
        if frame_handle is not None:
            payload["frame_handle"] = dataclasses.asdict(frame_handle)
        try:
            self._client().set(_key(camera_id), json.dumps(payload), ex=_TTL_S)
        except Exception as e:
            logger.debug(f"StreamHealthReporter publish failed for camera {camera_id}: {e}")
            if self._owns_client:
                self._redis = None


def read_stream_health(camera_id: int, redis_client=None) -> Dict:
    """Engine-side: read one camera's last-published health.

    Returns `{}` if no decode worker currently owns this camera, or the
    entry has expired (worker crashed, or hasn't published yet) — the
    caller treats that exactly like "no data", not an error.
    """
    try:
        client = redis_client
        if client is None:
            from messaging.redis_client import RedisClient

            client = RedisClient.get_instance().client
        raw = client.get(_key(camera_id))
        if raw is None:
            return {}
        return json.loads(raw)
    except Exception as e:
        logger.debug(f"read_stream_health failed for camera {camera_id}: {e}")
        return {}


def read_raw_frame(
    camera_id: int, max_age_sec: Optional[float] = None, redis_client=None
) -> Optional[np.ndarray]:
    """Engine-side: fetch this camera's most recently published full frame.

    Reconstructs the `FrameHandle` from the same published blob
    `read_stream_health` reads, then attaches the `camraw_<id>` segment it
    names. Returns None for every "nothing usable right now" case — no
    decode worker owns this camera, it hasn't published a frame yet, the
    published copy is older than `max_age_sec`, or the segment was already
    recycled by the time this runs — exactly the same convention
    `attach_and_read`/`attach_and_read_raw` already use, so the caller does
    not need to distinguish why.
    """
    health = read_stream_health(camera_id, redis_client=redis_client)
    if max_age_sec is not None:
        published_at = health.get("published_at")
        if published_at is None or (time.time() - published_at) > max_age_sec:
            return None
    handle_fields = health.get("frame_handle")
    if not handle_fields:
        return None
    try:
        handle = FrameHandle(**handle_fields)
    except TypeError as e:
        logger.debug(f"read_raw_frame: malformed handle for camera {camera_id}: {e}")
        return None
    return attach_and_read_raw(handle)
