"""Drop-in replacement for lum_vision's GlobalTrackManager, backed by Celery.

`CameraEngine` and `PersonTracker` both hold a `global_track_manager`
reference and call its methods directly — this class implements the same 10
call sites (all read from actually tracing both classes' source, not
guessed from the constructor signature; see global_track_client.py for how
they split into blocking vs. one-way) so that passing an instance of this
class into their existing constructors requires *no* changes to either
class.

This class is transport-agnostic: it only ever calls `.call()` and
`.call_one_way()` on whatever client it is given. That is what let the
underlying transport move from a Unix socket to Celery
(workers/global_track_client.py, docs/GLOBAL_TRACKING.md) without touching
anything in here beyond the type annotation.

What this deliberately does NOT do: replace GlobalTrackIDGenerator.
`PersonTracker.global_id_generator.get_next_id()` is a separate object and a
separate problem (an in-process `threading.Lock` counter that needs Redis
`INCR` once IDs are handed out from multiple processes) — solving it here
would conflate two unrelated fixes under one adapter.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from workers.global_track_client import GlobalTrackClient


@dataclass(frozen=True)
class GlobalTrackRef:
    """The one field CameraEngine reads off find_global_track_by_identity's
    result (`existing_global.global_id`, camera_engine.py's "Global ID
    reassignment" block). The real method returns a full GlobalTrack; the
    task projects it to this int before it crosses (see
    global_track_tasks.py), and the adapter re-wraps it here so the call
    site's attribute access works
    unchanged — returning the bare int instead would crash CameraEngine with
    AttributeError on `.global_id`, breaking the drop-in contract this whole
    module exists to keep.
    """

    global_id: int


class RemoteGlobalTrackManager:
    """Everything CameraEngine/PersonTracker need from a GlobalTrackManager,
    routed through a GlobalTrackClient to the register in global-track-worker.

    `enabled` is read once at construction from the same YAML config value
    the main process reads (`enable_global_tracking` in configs/config.yaml)
    — not fetched over the RPC. It's a static feature flag, never mutated
    after GlobalTrackManager.__init__ (confirmed by reading global_track.py
    directly), so asking the main process for it on every check would be a
    network round-trip to answer a question both processes can already
    answer identically from the same file.
    """

    def __init__(
        self,
        client: GlobalTrackClient,
        enabled: bool,
        async_assign: bool = True,
    ):
        self._client = client
        self.enabled = enabled

        # assign_global_id runs ReID similarity matching over the whole
        # gallery, and camera_engine.py calls it once per active track per
        # frame. Waiting for that reply inline is what let a busy main
        # process stall tracking: every call carries a timeout, and a
        # timeout drops the track to a local-only negative ID, so the person
        # is never recognised (the failure observed live before decode moved
        # to its own process). Nothing about local tracking needs the
        # answer — person_tracker.update() has already finished by the time
        # this is called, and the result is only written onto the track dict
        # as a label. So it is dispatched to a background thread and the
        # caller is handed whatever answer arrived from an earlier frame.
        self._async_assign = async_assign
        self._assign_pool: Optional[ThreadPoolExecutor] = None
        if async_assign:
            # One worker, not several: the point of this thread is to move
            # the wait off the camera's frame path, not to run several
            # lookups at once. Consecutive frames of one track ask an
            # identical question, and _in_flight below already suppresses
            # those, so a second worker would have little to do.
            self._assign_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="global-assign"
            )
        # Last known global id per (camera_id, local_track_id), written by
        # the background thread, read by the frame path.
        self._assigned: Dict[Tuple[int, int], int] = {}
        # Tracks with a request already in flight. One at a time per track:
        # consecutive frames of the same person ask an identical question,
        # so sending each one would multiply ReID work on the main process
        # for no new information, and out-of-order replies could overwrite a
        # newer answer with an older one.
        self._in_flight: set = set()
        self._assign_lock = threading.Lock()
        self._submitted_count = 0

    # ── Called from CameraEngine ────────────────────────────────────────────

    def assign_global_id(
        self,
        camera_id: int,
        local_track_id: int,
        person_crop: Optional[np.ndarray],
        face_embedding: Optional[np.ndarray] = None,
        detection_confidence: float = 0.0,
        frame_num: int = 0,
        identity: Optional[str] = None,
        identity_locked: bool = False,
    ) -> Optional[int]:
        """Returns the most recent global id known for this track, or None
        if no reply has arrived yet (the first one or two frames of a new
        track, typically). CameraEngine already treats a falsy/negative
        global_track_id as "not yet identified" — see camera_engine.py's
        `current_global_id >= 0` guard before publishing identity — so a
        transient None here degrades exactly the same way a slow RPC call
        already did, without ever blocking this frame on it.

        Synchronous fallback (async_assign=False) kept for tests and for
        anyone who needs the old blocking contract; not used by
        camera_tasks.py.
        """
        if not self._async_assign:
            result = self._client.call(
                "assign_global_id",
                camera_id=camera_id,
                local_track_id=local_track_id,
                person_crop=person_crop,
                face_embedding=face_embedding,
                detection_confidence=detection_confidence,
                frame_num=frame_num,
                identity=identity,
                identity_locked=identity_locked,
            )
            return result.value

        key = (camera_id, local_track_id)
        with self._assign_lock:
            already_running = key in self._in_flight
            if not already_running:
                self._in_flight.add(key)

        if not already_running:
            assert self._assign_pool is not None
            self._submitted_count += 1  # test-visible: proves suppression, not just serialization
            self._assign_pool.submit(
                self._run_assign,
                key,
                person_crop,
                face_embedding,
                detection_confidence,
                frame_num,
                identity,
                identity_locked,
            )

        with self._assign_lock:
            return self._assigned.get(key)

    def _run_assign(
        self,
        key: Tuple[int, int],
        person_crop: Optional[np.ndarray],
        face_embedding: Optional[np.ndarray],
        detection_confidence: float,
        frame_num: int,
        identity: Optional[str],
        identity_locked: bool,
    ) -> None:
        """Runs on the background pool. Owns clearing key from _in_flight —
        every exit path (success, RPC failure, unexpected exception) must
        clear it, or that track's global id freezes forever on whatever
        _assigned already holds.
        """
        camera_id, local_track_id = key
        try:
            result = self._client.call(
                "assign_global_id",
                camera_id=camera_id,
                local_track_id=local_track_id,
                person_crop=person_crop,
                face_embedding=face_embedding,
                detection_confidence=detection_confidence,
                frame_num=frame_num,
                identity=identity,
                identity_locked=identity_locked,
            )
            # result.ok=False means the client already fell back to a
            # negative local-only id (GlobalTrackClient.call's own contract).
            # Recording it anyway keeps this path's degraded behaviour
            # identical to the previous synchronous one: camera_engine.py's
            # `current_global_id >= 0` guard already treats negative the
            # same as "not yet identified" wherever it matters.
            with self._assign_lock:
                self._assigned[key] = result.value
        except Exception:
            logger.exception(
                f"global_track_adapter: background assign_global_id failed "
                f"for camera={camera_id} track={local_track_id}"
            )
        finally:
            with self._assign_lock:
                self._in_flight.discard(key)

    def forget_track(self, camera_id: int, local_track_id: int) -> None:
        """Drop cached state for a track that no longer exists, so a reused
        local_track_id on this camera doesn't inherit a stale global id.
        Safe to call even if a request for this key is still in flight —
        _run_assign re-checks nothing about liveness, so a late reply just
        repopulates _assigned under the same key; call this again after the
        id is no longer wanted if that matters for a given caller."""
        key = (camera_id, local_track_id)
        with self._assign_lock:
            self._assigned.pop(key, None)

    def get_global_id(self, camera_id: int, local_track_id: int) -> Optional[int]:
        """Pure local cache read — no RPC call, no failure mode to degrade.

        assign_global_id() already runs every frame for every active track,
        identity-locked or not (camera_engine.py's active-tracks loop is
        unconditional on that), so by the time a track shows up in
        removed_tracks its id is already sitting in _assigned. Returns None
        if it never was — e.g. global tracking only just got enabled, or the
        one call this frame raced the very first assign for this track.

        The caller is expected to call forget_track() right after reading,
        once it no longer needs the value — see on_track_removed's docstring
        for why that can't happen automatically in here.
        """
        with self._assign_lock:
            return self._assigned.get((camera_id, local_track_id))

    def find_global_track_by_identity(self, identity: str) -> Optional[GlobalTrackRef]:
        """Returns a GlobalTrackRef (or None), matching how CameraEngine uses
        the real method's GlobalTrack result: it reads `.global_id` and
        nothing else. See GlobalTrackRef for why the bare int the server
        sends must be re-wrapped here.

        On RPC failure this returns None ("not found"), never a fallback:
        presenting the client's negative local ID as a *found track* would
        make CameraEngine reassign the person to that fabricated global ID
        via reassign_local_track — silent identity corruption, the exact
        failure mode this adapter's degraded path must never produce.
        """
        result = self._client.call("find_global_track_by_identity", identity)
        if not result.ok or result.value is None:
            return None
        return GlobalTrackRef(global_id=result.value)

    def update_global_track_identity(
        self, global_id: int, identity: str, locked: bool = True
    ) -> None:
        # Real method returns bool; CameraEngine never checks it. One-way.
        self._client.call_one_way(
            "update_global_track_identity",
            global_id,
            identity,
            locked=locked,
        )

    def reassign_local_track(
        self, camera_id: int, local_track_id: int, new_global_id: int
    ) -> None:
        # Real method returns bool; CameraEngine never checks it. One-way.
        self._client.call_one_way(
            "reassign_local_track",
            camera_id=camera_id,
            local_track_id=local_track_id,
            new_global_id=new_global_id,
        )

    def on_face_detected(
        self,
        camera_id: int,
        local_track_id: int,
        quality: float = 0.0,
        recognized: bool = False,
        identity: Optional[str] = None,
    ) -> None:
        self._client.call_one_way(
            "on_face_detected",
            camera_id=camera_id,
            local_track_id=local_track_id,
            quality=quality,
            recognized=recognized,
            identity=identity,
        )

    def on_face_not_visible(self, camera_id: int, local_track_id: int) -> None:
        self._client.call_one_way(
            "on_face_not_visible", camera_id=camera_id, local_track_id=local_track_id
        )

    # ── Called from PersonTracker ───────────────────────────────────────────

    def on_track_created(
        self,
        camera_id: int,
        local_track_id: int,
        bbox: Optional[np.ndarray] = None,
        frame_num: int = 0,
    ) -> None:
        self._client.call_one_way(
            "on_track_created",
            camera_id=camera_id,
            local_track_id=local_track_id,
            bbox=bbox,
            frame_num=frame_num,
        )

    def on_track_update(self, camera_id: int, local_track_id: int) -> None:
        self._client.call_one_way(
            "on_track_update", camera_id=camera_id, local_track_id=local_track_id
        )

    def on_track_removed(
        self,
        camera_id: int,
        local_track_id: int,
        track_history: Optional[List[dict]] = None,
        total_frames: int = 0,
    ) -> None:
        # Does NOT call forget_track(). PersonTracker.update() (the vendored
        # lum_vision package) calls this internally, before returning
        # removed_tracks to CameraEngine — so forgetting here would clear
        # the cache before CameraEngine's removed-tracks loop gets a chance
        # to read it via get_global_id(). That loop calls forget_track()
        # itself, right after reading, once the tracker's own integer is
        # actually free to be reused for someone new.
        # Real method returns int | None; PersonTracker never checks it. One-way.
        self._client.call_one_way(
            "on_track_removed",
            camera_id=camera_id,
            local_track_id=local_track_id,
            track_history=track_history,
            total_frames=total_frames,
        )
