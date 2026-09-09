"""Celery-backed replacement for `global_track_rpc.GpuRpcClient`.

Same two methods, same signatures, same never-raises contracts, so
`RemoteGlobalTrackManager` (global_track_adapter.py) swaps one for the other
with no changes to itself or to any of its callers — only
`camera_tasks.py`'s construction line picks which. See docs/GLOBAL_TRACKING.md.

Three behaviours carried over deliberately, because callers depend on them:

  - `call()` never raises. On any failure it returns a negative, process-local
    ID from `_LocalIdFallback`, distinguishable at a glance from a real global
    ID (those start at 1000 and count up). `camera_engine.py`'s
    `current_global_id >= 0` guard already treats negative as "not yet
    identified", so a struggling global-track-worker degrades a camera to
    local-only tracking rather than stalling it.
  - `call_one_way()` never raises and never falls back. Its callers ignored
    the return value even when these ran in-process.
  - The method allow-lists stay closed. The task modules dispatch by name, so
    an open set would let a caller invoke anything the register exposes.

`person_crop` is the one argument that does not travel through the broker.
Pixels through Redis were measured at ~15.6ms for a frame (frame_store.py's
module docstring) — more than the inference they would be feeding. The crop
goes into this camera's shared-memory ring instead and only a handle crosses,
which the task reads back (see global_track_tasks.assign_global_id_task).
"""

from __future__ import annotations

import dataclasses
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from loguru import logger

from workers.frame_store import RoiBatchSlot


@dataclass(frozen=True)
class MethodCallResult:
    """What a blocking call returns. Moved here verbatim from the deleted
    global_track_rpc.py: the transport changed, this contract did not, and
    RemoteGlobalTrackManager still branches on `.ok` exactly as before."""

    value: Any
    ok: bool  # False when this came from the local-ID/None fallback


class _LocalIdFallback:
    """Per-process negative-ID counter, used only when the register is
    unreachable. Never shared across processes — two workers falling back
    concurrently must not hand out the same local ID, which a shared negative
    range would risk; a per-process counter starting at -1 and counting down
    guarantees no two *processes* collide as long as each process's IDs are
    tagged with its own identity downstream (not handled by this class —
    the caller is responsible for that if two local IDs must ever be told
    apart, which is not currently required).
    """

    def __init__(self):
        self._next = -1
        self._lock = threading.Lock()

    def next_id(self) -> int:
        with self._lock:
            value = self._next
            self._next -= 1
            return value

# Below the task's own soft limits, and sized like the socket timeout it
# replaces (SO_GPU_RPC_TIMEOUT_S defaulted to 0.25s): a timeout here degrades
# to a local-only ID rather than dropping a frame, so it does not need to be
# as tight as the frame budget. Expires strictly below the client timeout, so
# a legitimate server-side expiry can't race the client into logging a
# transport failure — same ordering rationale as face_client.py.
DEFAULT_EXPIRES_S = float(os.environ.get("SO_GLOBALTRACK_EXPIRES_S", "1.0"))
DEFAULT_TIMEOUT_S = float(os.environ.get("SO_GLOBALTRACK_TIMEOUT_S", "1.5"))

# Per-method expiry overrides. on_track_removed is rare (once per track, not
# once per frame) and idempotent (mark_camera_inactive and the
# local_to_global pop are both safe to apply twice), and nothing awaits its
# result — so unlike the per-frame one-way calls, it can afford a longer
# window to survive a busy or restarting global-track-worker. This narrows
# the "message silently expires before the worker is free" gap that leaves a
# camera's active flag stuck True forever (see docs/GLOBAL_TRACKS_LEAK.md);
# it does not close it — cleanup_inactive_global_tracks' hard-expiry fallback
# is what actually bounds the leak when a message is still lost.
_EXPIRES_OVERRIDES_S: Dict[str, float] = {
    "on_track_removed": float(os.environ.get("SO_GLOBALTRACK_REMOVED_EXPIRES_S", "30.0")),
}

_BLOCKING_METHODS = frozenset({"assign_global_id", "find_global_track_by_identity"})
_ONE_WAY_METHODS = frozenset(
    {
        "on_face_detected",
        "on_face_not_visible",
        "update_global_track_identity",
        "reassign_local_track",
        "on_track_created",
        "on_track_update",
        "on_track_removed",
    }
)

# get_global_id is deliberately absent from both sets: it is answered from the
# adapter's own local cache and never crosses a process boundary at all (see
# RemoteGlobalTrackManager.get_global_id). Naming it here would resurrect a
# round trip that was removed on purpose.


def _task_for(method: str):
    """Resolve a method name to its task. Imported lazily: the parent process
    constructs this client, and importing the task module at module scope
    would pull lum_vision into a process that must not touch CUDA before
    fork."""
    from workers import global_track_tasks as t

    return {
        "assign_global_id": t.assign_global_id_task,
        "find_global_track_by_identity": t.find_global_track_by_identity_task,
        "on_face_detected": t.on_face_detected_task,
        "on_face_not_visible": t.on_face_not_visible_task,
        "update_global_track_identity": t.update_global_track_identity_task,
        "reassign_local_track": t.reassign_local_track_task,
        "on_track_created": t.on_track_created_task,
        "on_track_update": t.on_track_update_task,
        "on_track_removed": t.on_track_removed_task,
    }[method]


class GlobalTrackClient:
    """Worker-side: call the register on `global-track-worker`.

    Drop-in for GpuRpcClient — see this module's docstring.
    """

    def __init__(
        self,
        expires_s: float = DEFAULT_EXPIRES_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._expires_s = expires_s
        self._timeout_s = timeout_s
        self._fallback = _LocalIdFallback()
        self._slots: Dict[int, RoiBatchSlot] = {}
        self._slots_lock = threading.Lock()
        # Same convention as GpuRpcClient.on_fallback: a struggling
        # global-track-worker should be visible as a failure-rate metric, not
        # only inferable from a spike in negative local-only IDs.
        self.on_fallback: Optional[Callable[[], None]] = None

    def _slot_for(self, camera_id: int) -> RoiBatchSlot:
        # purpose="reid": these are person crops headed for ReID matching,
        # namespaced away from the face-crop ring for the same camera (see
        # frame_store._roi_slot_name).
        with self._slots_lock:
            slot = self._slots.get(camera_id)
            if slot is None:
                slot = RoiBatchSlot(camera_id, purpose="reid")
                self._slots[camera_id] = slot
            return slot

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Recursively convert numpy into plain JSON-serializable types.

        Task payloads are JSON (task_serializer='json', celery_app.py) —
        unlike results, which are pickled and can carry numpy freely. The one
        known numpy-bearing argument across these nine methods is
        on_track_created's `bbox` (Optional[np.ndarray] — see
        global_track_adapter.py), but this is applied uniformly rather than
        special-cased to that one call site, so a future numpy-typed argument
        elsewhere fails loudly in a test instead of dropping silently in
        production the way this one did (EncodeError inside apply_async,
        caught by call_one_way's own try/except and merely logged).
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {k: GlobalTrackClient._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [GlobalTrackClient._json_safe(v) for v in value]
        return value

    def _to_handle(self, camera_id: int, person_crop: Optional[np.ndarray]):
        """Put the crop in shared memory and return its handle dict, or None
        if there is no crop. None is a legitimate value, not an error — the
        register treats a missing crop as 'skip ReID, create a new global
        track' (global_track.py STEP 1)."""
        if person_crop is None or getattr(person_crop, "size", 0) == 0:
            return None
        handle = self._slot_for(camera_id).write([person_crop], [0])
        return dataclasses.asdict(handle)

    def call(self, method: str, *args: Any, **kwargs: Any) -> MethodCallResult:
        """Blocking call. Never raises — see the module docstring."""
        if method not in _BLOCKING_METHODS:
            raise ValueError(
                f"global_track_client.call: {method!r} is not a recognised "
                f"blocking method (expected one of {sorted(_BLOCKING_METHODS)}); "
                f"did you mean call_one_way?"
            )

        async_result = None
        try:
            task = _task_for(method)
            if method == "assign_global_id":
                kwargs = dict(kwargs)
                crop = kwargs.pop("person_crop", None)
                # face_embedding is accepted by the register's signature but
                # unused by it today, and would pickle a 512-float array
                # through the broker for nothing. Dropped here rather than
                # sent; if the register ever uses it, this is the line to
                # change.
                kwargs.pop("face_embedding", None)
                kwargs["handle"] = self._to_handle(kwargs.get("camera_id", 0), crop)
                async_result = task.apply_async(
                    kwargs=self._json_safe(kwargs), queue="globaltrack",
                    expires=self._expires_s,
                )
            else:
                async_result = task.apply_async(
                    args=self._json_safe(args), kwargs=self._json_safe(kwargs),
                    queue="globaltrack", expires=self._expires_s,
                )
            value = async_result.get(
                timeout=self._timeout_s, disable_sync_subtasks=False
            )
            return MethodCallResult(value=value, ok=True)
        except Exception as e:
            logger.warning(
                f"global_track_client: {method} failed: {type(e).__name__}: {e} "
                f"— falling back to a local-only ID"
            )
            if self.on_fallback is not None:
                self.on_fallback()
            return MethodCallResult(value=self._fallback.next_id(), ok=False)
        finally:
            if async_result is not None:
                # Results are small ints here, but result_expires is an hour;
                # forgetting keeps a per-track-per-frame call rate from
                # accumulating dead keys in Redis. Same reasoning as
                # face_client.py's forget(), which documents the scale.
                try:
                    async_result.forget()
                except Exception:
                    pass

    def call_one_way(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Fire-and-forget. A failure is logged, never raised or returned."""
        if method not in _ONE_WAY_METHODS:
            raise ValueError(
                f"global_track_client.call_one_way: {method!r} is not a "
                f"recognised one-way method (expected one of "
                f"{sorted(_ONE_WAY_METHODS)}); did you mean call?"
            )
        try:
            _task_for(method).apply_async(
                args=self._json_safe(args),
                kwargs=self._json_safe(kwargs),
                queue="globaltrack",
                expires=_EXPIRES_OVERRIDES_S.get(method, self._expires_s),
            )
        except Exception as e:
            logger.warning(
                f"global_track_client: one-way {method} failed: "
                f"{type(e).__name__}: {e} — dropped, no fallback (caller does "
                f"not use this call's result)"
            )

    def close(self) -> None:
        """Release this client's shared-memory rings."""
        with self._slots_lock:
            for slot in self._slots.values():
                try:
                    slot.close()
                except Exception:
                    pass
            self._slots.clear()
