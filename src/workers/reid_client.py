"""Synchronous client for the `reid.extract_batch` Celery task.

Used from two places, both blocking, both wanting exactly one crop's
embedding back:

- `RemoteBodyReidExtractor` (this module) — the `BodyReidExtractor`-shaped
  object `global_track_tasks.py` hands to `GlobalTrackManager` in place of a
  locally-loaded model, so `assign_global_id`'s existing
  `self._reid_extractor.extract(person_crop)` call (see
  person_tracking/reid.py and global_track.py's lock-scope comment on
  assign_global_id — this call already runs OUTSIDE that method's lock, so a
  network round trip here is exactly the case that restructuring exists for)
  reaches reid-worker instead of an in-process model.

Follows `face_client.py`'s shape (`apply_async` + blocking `.get()`, never
raises, `on_fallback` hook, `async_result.forget()`), reduced to a batch of
one: `RemoteGlobalTrackManager._run_assign` and
`GlobalTrackManager.assign_global_id` both submit one crop per call, not a
per-frame batch across tracks — see global_track_adapter.py's docstring for
why. `reid.extract_batch_task` already accepts any batch length including 1;
building a second single-crop task would duplicate it for no reason.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Callable, Optional

import numpy as np
from loguru import logger

from workers.frame_store import RoiBatchSlot

DEFAULT_EXPIRES_S = float(os.environ.get("SO_REID_EXTRACT_EXPIRES_S", "1.0"))
DEFAULT_TIMEOUT_S = float(os.environ.get("SO_REID_EXTRACT_TIMEOUT_S", "1.5"))


class RemoteBodyReidExtractor:
    """`BodyReidExtractor`-shaped: implements the one method
    `GlobalTrackManager._extract_body_embedding` calls
    (`self._reid_extractor.extract(person_crop)`), so it is a drop-in for
    the real `lum_vision.BodyReidExtractor` without any change to
    GlobalTrackManager itself.

    Owns one `RoiBatchSlot(purpose="reid")` per camera_id, mirroring how
    `_CameraContext` in camera_tasks.py owns one per-camera slot for face
    crops — a single crop still needs a camera_id to name its shared-memory
    ring, since the ring is per-camera (see frame_store.py's RoiBatchSlot).
    Lazily created on first use per camera_id rather than requiring the
    caller to pre-register every camera, since global-track-worker doesn't
    otherwise track which cameras exist.
    """

    def __init__(
        self,
        expires_s: float = DEFAULT_EXPIRES_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._expires_s = expires_s
        self._timeout_s = timeout_s
        self._slots: dict = {}
        # Same convention as gpu_rpc.GpuRpcClient.on_fallback and
        # face_client.FaceEmbedClient.on_fallback: a struggling reid-worker
        # should be visible as a failure-rate metric, not only inferable
        # from global tracking silently degrading to "always create a new
        # track."
        self.on_fallback: Optional[Callable[[], None]] = None

    def _slot_for(self, camera_id: int) -> RoiBatchSlot:
        slot = self._slots.get(camera_id)
        if slot is None:
            slot = RoiBatchSlot(camera_id, purpose="reid")
            self._slots[camera_id] = slot
        return slot

    def extract(
        self, person_crop: np.ndarray, *, camera_id: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Extract one body ReID embedding via reid-worker.

        Signature matches `lum_vision.BodyReidExtractor.extract` exactly —
        `person_crop` positional, `camera_id` keyword-only — since
        `GlobalTrackManager._extract_body_embedding` calls
        `self._reid_extractor.extract(person_crop, camera_id=camera_id)`
        without knowing which implementation it holds (see reid.py's
        extract() docstring in lum-model-vision). Unlike the in-process
        class, this implementation actually uses camera_id: it names the
        shared-memory ring the crop travels through to reid-worker, so it is
        required here even though the shared base signature marks it
        optional. `None` only in practice if GlobalTrackManager is ever
        called with `camera_id=None` itself, which no caller in this
        codebase does — see global_track_tasks.py, the only place this
        class is constructed.

        Never raises: returns None on any failure (timeout, reid-worker
        down, a malformed crop) — GlobalTrackManager already treats a None
        embedding as "skip ReID matching, create a new global track"
        (global_track.py's STEP 2), the same degraded behavior a
        same-process model failure already produced.
        """
        if camera_id is None:
            logger.warning("reid_client: extract() called with camera_id=None — no embedding")
            return None
        from workers.reid_tasks import extract_batch_task

        # apply_async itself can raise (broker unreachable) - not just the
        # later .get(). Both must be caught for "never raises" to actually
        # hold; face_client.py's embed() has this same gap, not fixed here
        # since it is working, unrelated code - but this class's docstring
        # makes the same promise, so it must actually keep it.
        async_result = None
        try:
            slot = self._slot_for(camera_id)
            handle = slot.write([person_crop], [0])  # dummy track_id: single-crop batch
            async_result = extract_batch_task.apply_async(
                kwargs={"handle": dataclasses.asdict(handle)},
                queue="reid",
                expires=self._expires_s,
            )
            results = async_result.get(
                timeout=self._timeout_s, disable_sync_subtasks=False
            )
        except Exception as e:
            logger.warning(
                f"reid_client: extract(camera_id={camera_id}) failed: "
                f"{type(e).__name__}: {e} — no embedding this call"
            )
            if self.on_fallback is not None:
                self.on_fallback()
            return None
        finally:
            if async_result is not None:
                async_result.forget()

        if not results or results[0] is None:
            return None
        return np.asarray(results[0], dtype=np.float32)
