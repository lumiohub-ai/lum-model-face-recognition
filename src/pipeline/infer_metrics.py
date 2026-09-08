"""Cross-process GPU-latency reporting for the YOLO and face workers.

The same bridge `decode_metrics.py` builds for `read_ms`/`decode_ms`, for the
other half of the dashboard that the Celery split disconnected:
`MetricsCollector.record_yolo_ms` / `record_arcface_ms` and friends used to be
called in-process, on the same object the dashboard read from. Detection and
embedding now run in `yolo-worker` / `face-worker`, so those methods lost every
caller and their panels went blank (docs/FOLLOW_UPS.md item 10).

Workers publish, the engine reads back, exactly as with stream health. Two
deliberate differences from that module:

  - **One key per process, not per camera.** These stages are cross-camera and
    horizontally scaled: N replicas x M prefork children all serve one `yolo`
    queue, and none of them owns a camera the way a decode worker does. Each
    process therefore publishes under its own `<stage>:<worker_id>` key and the
    engine sums across whatever keys currently exist. FOLLOW_UPS proposed a
    single Redis hash; a hash has no per-field TTL, so a dead worker's fields
    would sit there being averaged in forever. Per-process keys expire
    themselves, which is the property that actually matters here.

  - **Cumulative counters, not gauges.** A worker publishes running totals
    (count/ms/batch) and the engine differences successive reads to get an
    average over the interval between them. Publishing a pre-averaged number
    instead would make every reader see whatever window the *worker* chose,
    and a worker that went idle would keep reporting its last busy average
    rather than falling to zero.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Dict

from loguru import logger

_KEY_PREFIX = "infer:latency:"

# Same reasoning as decode_metrics._TTL_S: comfortably longer than the publish
# interval below, so a killed worker's counters expire instead of being summed
# into the average forever.
_TTL_S = 15

# Publishing on every batch would put a Redis round-trip on the inference hot
# path — at a 10ms flush interval that is 100 writes/sec/worker to move a
# number the dashboard samples every few seconds. Accumulate in-process and
# write at most this often instead.
_PUBLISH_INTERVAL_S = float(os.environ.get("SO_INFER_METRICS_INTERVAL_S", "2"))


def _worker_id() -> str:
    """Stable within a process, unique across them.

    Hostname distinguishes replicas (each container gets its own), pid
    distinguishes prefork children within a replica. Both are needed: pid
    alone collides across containers.
    """
    return f"{socket.gethostname()}:{os.getpid()}"


def _key(stage: str, worker_id: str) -> str:
    return f"{_KEY_PREFIX}{stage}:{worker_id}"


class InferLatencyReporter:
    """Worker-side: accumulate per-batch timings, publish totals periodically.

    Mirrors `StreamHealthReporter`'s degrade-on-failure posture — a Redis
    outage means the dashboard's latency numbers go stale and then blank,
    never that inference raises over a metrics side-channel.

    Thread-safe: the face worker's task runs on a prefork child with a solo
    pool, but the yolo worker's `Batches` flush and any future threaded pool
    could both land here.
    """

    def __init__(self, stage: str, redis_client=None, interval_s: float = _PUBLISH_INTERVAL_S):
        self._stage = stage
        self._redis = redis_client
        self._owns_client = redis_client is None
        self._interval_s = interval_s
        self._worker_id = _worker_id()
        self._lock = threading.Lock()
        self._totals: Dict[str, float] = {}
        self._last_publish = 0.0

    def _client(self):
        if self._redis is not None:
            return self._redis
        from messaging.redis_client import RedisClient

        self._redis = RedisClient.get_instance().client
        return self._redis

    def record(self, **fields: float) -> None:
        """Add one batch's measurements to this process's running totals.

        Field names are free-form and pass through to the reader untouched;
        `read_infer_latency` divides `<name>_ms` by `<name>_count` to form an
        average, so callers pair them (`total_ms`/`total_count`, and so on).
        Publishes at most once per `interval_s`, on whichever call first
        crosses it — no background thread, so a worker that stops receiving
        work stops publishing and its key expires, which is exactly the
        signal the dashboard should show.
        """
        now = time.monotonic()
        with self._lock:
            for name, value in fields.items():
                self._totals[name] = self._totals.get(name, 0.0) + value
            if (now - self._last_publish) < self._interval_s:
                return
            self._last_publish = now
            payload = dict(self._totals)

        payload["published_at"] = time.time()
        try:
            self._client().set(
                _key(self._stage, self._worker_id), json.dumps(payload), ex=_TTL_S
            )
        except Exception as e:
            logger.debug(f"InferLatencyReporter[{self._stage}] publish failed: {e}")
            if self._owns_client:
                self._redis = None


class _RateReader:
    """Engine-side: turn published cumulative totals into per-interval averages.

    Holds the previous read so successive snapshots can be differenced. One
    instance per stage, kept alive across polls by the closures registered in
    `engine.py`'s `_register_pipeline_gauges`.
    """

    def __init__(self, stage: str, cache_ttl_s: float = 1.0):
        self._stage = stage
        self._cache_ttl_s = cache_ttl_s
        self._lock = threading.Lock()
        self._cached: Dict[str, float] = {}
        self._cached_at = 0.0
        self._prev: Dict[str, float] = {}
        self._prev_at = 0.0

    def _scan(self, client) -> Dict[str, float]:
        """Sum every live worker's totals for this stage.

        SCAN rather than KEYS: this runs against the same Redis the whole
        pipeline uses, and KEYS blocks the server for the duration of the
        scan.
        """
        summed: Dict[str, float] = {}
        for key in client.scan_iter(match=f"{_KEY_PREFIX}{self._stage}:*", count=100):
            raw = client.get(key)
            if raw is None:
                continue  # expired between SCAN and GET
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                continue
            for name, value in payload.items():
                if name == "published_at" or not isinstance(value, (int, float)):
                    continue
                summed[name] = summed.get(name, 0.0) + value
        return summed

    def read(self, redis_client=None) -> Dict[str, float]:
        """Averages since the previous read, as `{<name>_avg_ms: float}`.

        Returns `{}` when no worker for this stage is publishing — the caller
        treats that as "no data" the same way `read_stream_health` does. The
        first call after startup also returns `{}`: one cumulative sample has
        nothing to difference against, and dividing lifetime totals instead
        would report a process-lifetime average that drifts ever more slowly
        toward the truth (the same trap `_BatchStats` avoids by using a
        window).
        """
        now = time.monotonic()
        with self._lock:
            if self._cached and (now - self._cached_at) < self._cache_ttl_s:
                return dict(self._cached)

        try:
            client = redis_client
            if client is None:
                from messaging.redis_client import RedisClient

                client = RedisClient.get_instance().client
            totals = self._scan(client)
        except Exception as e:
            logger.debug(f"read_infer_latency[{self._stage}] failed: {e}")
            return {}

        with self._lock:
            prev, self._prev, self._prev_at = self._prev, totals, now
            out: Dict[str, float] = {}
            if prev:
                for name, value in totals.items():
                    if not name.endswith("_ms"):
                        continue
                    stem = name[: -len("_ms")]
                    d_ms = value - prev.get(name, 0.0)
                    d_count = totals.get(f"{stem}_count", 0.0) - prev.get(f"{stem}_count", 0.0)
                    # A worker restarting resets its counters, so a negative
                    # delta means "the fleet changed under us", not real work.
                    # Skip this interval rather than publishing a nonsense
                    # number; the next one is correct again.
                    if d_ms < 0 or d_count <= 0:
                        continue
                    out[f"{stem}_avg_ms"] = round(d_ms / d_count, 1)
            self._cached = out
            self._cached_at = now
            return dict(out)


_readers: Dict[str, _RateReader] = {}
_readers_lock = threading.Lock()


def read_infer_latency(stage: str, redis_client=None) -> Dict[str, float]:
    """Engine-side entry point: averaged latencies for one stage.

    Keyed by stage so repeated calls share the differencing state and the
    short cache — `snapshot()` reads several fields per poll and each one
    would otherwise pay its own SCAN.
    """
    with _readers_lock:
        reader = _readers.get(stage)
        if reader is None:
            reader = _RateReader(stage)
            _readers[stage] = reader
    return reader.read(redis_client=redis_client)
