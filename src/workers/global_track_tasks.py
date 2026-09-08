"""The cross-camera register (GlobalTrackManager), as Celery tasks.

Replaces `global_track_rpc.py`'s Unix socket. Same nine methods, same
blocking/one-way split, same degraded contracts — only the transport
underneath changes, so `RemoteGlobalTrackManager` (global_track_adapter.py)
and its callers need no changes at all. See docs/GLOBAL_TRACKING.md.

**Exactly one replica.** GlobalTrackManager is a single shared list
(`global_tracks`, `local_to_global`); a second consumer of the `globaltrack`
queue would mint duplicate global IDs — the exact problem global tracking
exists to solve. compose.yml is what enforces this; nothing here can.

This worker owns three things that used to live in `person-tracking`'s
engine.py, because the register cannot exist in two places:

  - `periodic_validation()` — the register's own 30s housekeeping, on a timer
    started here rather than by the engine's main loop.
  - The `global_tracks` dashboard gauge — published to Redis on a timer,
    following the same publish/read-back convention as
    `pipeline/infer_metrics.py`. Simpler than that module though: one
    publisher instead of N, so a single fixed key replaces per-process keys
    and `scan_iter`, and a live count replaces cumulative counters (the
    dashboard wants "how many right now", not an interval average).
  - The baseline summary — logged at this worker's own shutdown, where the
    true final numbers are, rather than at person-tracking's.

ReID runs in its own worker: the GlobalTrackManager built here gets a
`RemoteBodyReidExtractor` (workers/reid_client.py) instead of loading the
model locally, so `assign_global_id`'s existing
`self._reid_extractor.extract(...)` call reaches `reid-worker` over the
`reid` queue. That call already runs outside GlobalTrackManager's lock (see
lum_vision's global_track.py lock-scope comment), which is what makes a
network round trip there safe rather than a system-wide serialisation point.

No lum_vision import at module scope — the parent imports this module to
register tasks, and a CUDA touch there poisons `fork()`. Same constraint as
face_tasks.py and model_holder.py.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

from celery.signals import worker_process_init, worker_shutdown
from loguru import logger

from workers.celery_app import celery

# Single fixed key: this worker is single-replica by construction, so there is
# no per-process fan-out to scan for. TTL comfortably exceeds the publish
# interval, so a killed worker's gauge expires rather than reading as a live
# count forever — same reasoning as infer_metrics._TTL_S.
STATS_KEY = "globaltrack:stats"
_STATS_TTL_S = 15
_STATS_INTERVAL_S = float(os.environ.get("SO_GLOBALTRACK_STATS_INTERVAL_S", "2"))
_VALIDATION_INTERVAL_S = float(
    os.environ.get("SO_GLOBALTRACK_VALIDATION_INTERVAL_S", "30")
)

_manager = None
_manager_lock = threading.Lock()
_background_started = False


def ensure_manager_loaded():
    """Build the register once per process. Idempotent, thread-safe.

    Mirrors model_holder.ensure_*_loaded's double-checked-lock shape (the
    solo/threads pools don't emit worker_process_init, so every task also
    calls this rather than relying on the signal alone) — but lives here, not
    in model_holder.py, because this is not a GPU model singleton: it is
    mutable cross-camera state whose whole identity is that exactly one of it
    exists.
    """
    global _manager
    if _manager is not None:
        return _manager
    with _manager_lock:
        if _manager is not None:
            return _manager

        from lum_vision import GlobalTrackManager

        from config.startup import load_config
        from config.vision import build_vision_config
        from workers.reid_client import RemoteBodyReidExtractor

        cfg = build_vision_config(load_config() or {})
        extractor = RemoteBodyReidExtractor()
        extractor.on_fallback = lambda: logger.warning(
            "global_track_tasks: reid-worker fallback fired — "
            "no body embedding, falling through to a new global track"
        )

        config_path = cfg.global_tracking_config_path
        _manager = GlobalTrackManager(
            config_path=str(config_path) if config_path else None,
            enabled=cfg.enable_global_tracking,
            weights_dir=cfg.weights_dir,
            reid_extractor=extractor,
        )
        logger.info(
            f"global_track_tasks: GlobalTrackManager ready "
            f"(enabled={_manager.enabled}, reid=remote)"
        )
    return _manager


def _redis():
    from messaging.redis_client import RedisClient

    return RedisClient.get_instance().client


def _publish_stats(manager) -> None:
    """Publish the live gauge values person-tracking's dashboard reads.

    Best-effort: a Redis outage must never propagate into the register's own
    task path, matching how identity_throttle.py and infer_metrics.py treat
    their own publishes.
    """
    try:
        payload = {
            "global_tracks": len(manager.global_tracks),
            "ts": time.time(),
        }
        _redis().set(STATS_KEY, json.dumps(payload), ex=_STATS_TTL_S)
    except Exception as e:
        logger.debug(f"global_track_tasks: stats publish failed: {e}")


def _background_loop() -> None:
    """Periodic validation + stats publishing, on one thread.

    One thread for both rather than two: they run against the same object at
    similar cadences, and validation is the slower of the two, so a shared
    loop keeps the number of threads touching the register to exactly one.
    """
    manager = ensure_manager_loaded()
    last_validation = 0.0
    while True:
        try:
            time.sleep(_STATS_INTERVAL_S)
            _publish_stats(manager)

            now = time.monotonic()
            if now - last_validation >= _VALIDATION_INTERVAL_S:
                last_validation = now
                if manager.enabled:
                    result = manager.periodic_validation()
                    if any(result.values()):
                        logger.info(f"global_track_tasks: periodic_validation {result}")
        except Exception as e:
            # Never let this thread die: it owns the only periodic_validation
            # call in the system now, and a silent exit would stop conflict
            # detection and archival with no other symptom.
            logger.exception(f"global_track_tasks: background loop error: {e}")


def _start_background_once() -> None:
    global _background_started
    with _manager_lock:
        if _background_started:
            return
        _background_started = True
    threading.Thread(
        target=_background_loop, daemon=True, name="globaltrack-background"
    ).start()


@worker_process_init.connect
def _load_in_child(**_kwargs):
    """Preload the register post-fork, but ONLY in a worker that serves this
    queue — see yolo_tasks.py's matching guard for why the check is needed and
    why the worker must declare it via an env var."""
    if "globaltrack" in os.environ.get("SO_WORKER_PRELOAD", "").split(","):
        ensure_manager_loaded()
        _start_background_once()


@worker_shutdown.connect
def _log_baseline_on_shutdown(**_kwargs):
    """The baseline summary person-tracking's engine.py used to print at its
    own exit. It moved here with the register: this process owns the data, so
    these are the true final numbers rather than a last-published snapshot."""
    if _manager is None or not _manager.enabled:
        return
    try:
        logger.info("=" * 80)
        logger.info("PHASE 0 - FINAL BASELINE METRICS")
        logger.info("=" * 80)
        _manager.log_baseline_summary()
        m = _manager.get_baseline_metrics()
        logger.info(f"Total tracks created: {m['total_tracks_created']}")
        logger.info(f"Total tracks removed: {m['total_tracks_removed']}")
        logger.info(f"Average track duration: {m['avg_track_duration_sec']:.1f}s")
        logger.info(f"Face visibility rate: {m['face_visibility_rate']:.1%}")
        logger.info(f"Faces detected: {m['total_faces_detected']}")
        logger.info(f"Faces not visible: {m['total_faces_not_visible']}")
        logger.info("=" * 80)
    except Exception as e:
        logger.warning(f"global_track_tasks: baseline summary failed: {e}")


# ── Blocking: the caller uses the return value ──────────────────────────────


@celery.task(name="globaltrack.assign_global_id", queue="globaltrack", track_started=False)
def assign_global_id_task(handle: Optional[Dict[str, Any]] = None, **kwargs) -> Optional[int]:
    """Assign (or look up) a global ID for one local track.

    `handle` is a `RoiBatchHandle` dict naming the person crop in shared
    memory, or None when the caller had no usable crop — pixels never travel
    through the broker (frame_store.py measured ~15.6ms for a frame through
    Redis). GlobalTrackManager itself takes the crop as a plain array, so it
    is read back here and passed through unchanged.

    Never raises: on any failure the caller's client turns this into the same
    negative local-only ID a socket timeout used to produce, so a camera keeps
    tracking locally rather than stalling. Returning None here means "no
    answer"; the client, not this task, owns that fallback.
    """
    _start_background_once()
    manager = ensure_manager_loaded()

    person_crop = None
    if handle is not None:
        from workers.frame_store import (
            RoiBatchHandle,
            RoiHandle,
            attach_and_read_roi_batch,
        )

        batch = RoiBatchHandle(
            camera_id=handle["camera_id"],
            seq=handle["seq"],
            segment=handle["segment"],
            instance_id=handle["instance_id"],
            rois=tuple(RoiHandle(**r) for r in handle["rois"]),
            purpose=handle.get("purpose", "reid"),
        )
        packed = attach_and_read_roi_batch(batch)
        if packed:
            person_crop = packed[0][1]
        else:
            # Segment recycled before we read it. Not an error: the register
            # already treats a missing crop as "skip ReID, create a new global
            # track" (global_track.py STEP 1), which is the correct degraded
            # answer, not a reason to fail the task.
            logger.debug(
                f"globaltrack.assign_global_id: crop seq={batch.seq} gone — "
                f"assigning without ReID"
            )

    return manager.assign_global_id(person_crop=person_crop, **kwargs)


@celery.task(
    name="globaltrack.find_global_track_by_identity",
    queue="globaltrack",
    track_started=False,
)
def find_global_track_by_identity_task(identity: str) -> Optional[int]:
    """Look up a global track by locked identity.

    Returns the bare `global_id` int, not the GlobalTrack — the only caller
    (camera_engine.py's "Global ID reassignment" block) reads exactly that one
    field, and the full object carries live numpy that would be a stale
    snapshot by arrival. The client re-wraps it so the call site's
    `.global_id` access is unchanged; see GlobalTrackRef in
    global_track_adapter.py.
    """
    _start_background_once()
    manager = ensure_manager_loaded()
    track = manager.find_global_track_by_identity(identity)
    return track.global_id if track is not None else None


# ── One-way: the caller ignores the return value ────────────────────────────
#
# These were fire-and-forget over the socket too. A dropped one leaves the
# register's view of identity very slightly stale, which is a condition the
# system already tolerates (see global_track_rpc.py's module docstring) —
# acceptable, and called out here so it is not mistaken for solved.


@celery.task(name="globaltrack.on_face_detected", queue="globaltrack", track_started=False)
def on_face_detected_task(**kwargs) -> None:
    ensure_manager_loaded().on_face_detected(**kwargs)


@celery.task(name="globaltrack.on_face_not_visible", queue="globaltrack", track_started=False)
def on_face_not_visible_task(**kwargs) -> None:
    ensure_manager_loaded().on_face_not_visible(**kwargs)


@celery.task(
    name="globaltrack.update_global_track_identity", queue="globaltrack", track_started=False
)
def update_global_track_identity_task(*args, **kwargs) -> None:
    ensure_manager_loaded().update_global_track_identity(*args, **kwargs)


@celery.task(name="globaltrack.reassign_local_track", queue="globaltrack", track_started=False)
def reassign_local_track_task(**kwargs) -> None:
    ensure_manager_loaded().reassign_local_track(**kwargs)


@celery.task(name="globaltrack.on_track_created", queue="globaltrack", track_started=False)
def on_track_created_task(**kwargs) -> None:
    ensure_manager_loaded().on_track_created(**kwargs)


@celery.task(name="globaltrack.on_track_update", queue="globaltrack", track_started=False)
def on_track_update_task(**kwargs) -> None:
    ensure_manager_loaded().on_track_update(**kwargs)


@celery.task(name="globaltrack.on_track_removed", queue="globaltrack", track_started=False)
def on_track_removed_task(**kwargs) -> None:
    ensure_manager_loaded().on_track_removed(**kwargs)
