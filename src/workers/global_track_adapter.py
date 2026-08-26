"""Drop-in replacement for lum_vision's GlobalTrackManager, backed by gpu_rpc.

`CameraEngine` and `PersonTracker` both hold a `global_track_manager`
reference and call its methods directly — this class implements the same 10
call sites (all read from actually tracing both classes' source, not
guessed from the constructor signature; see gpu_rpc.py's module docstring
for how the 7 CameraEngine methods split into blocking vs. one-way) so that
passing an instance of this class into their existing constructors requires
*no* changes to either class.

What this deliberately does NOT do: replace GlobalTrackIDGenerator.
`PersonTracker.global_id_generator.get_next_id()` is a separate object and a
separate problem (an in-process `threading.Lock` counter that needs Redis
`INCR` once IDs are handed out from multiple processes) — solving it here
would conflate two unrelated fixes under one adapter.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from workers.gpu_rpc import GpuRpcClient


class RemoteGlobalTrackManager:
    """Everything CameraEngine/PersonTracker need from a GlobalTrackManager,
    routed through a GpuRpcClient to the real one in the main process.

    `enabled` is read once at construction from the same YAML config value
    the main process reads (`enable_global_tracking` in configs/config.yaml)
    — not fetched over the RPC. It's a static feature flag, never mutated
    after GlobalTrackManager.__init__ (confirmed by reading global_track.py
    directly), so asking the main process for it on every check would be a
    network round-trip to answer a question both processes can already
    answer identically from the same file.
    """

    def __init__(self, client: GpuRpcClient, enabled: bool):
        self._client = client
        self.enabled = enabled

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
    ) -> int:
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

    def get_global_id(self, camera_id: int, local_track_id: int) -> Optional[int]:
        # result.value is the right thing to return either way: on success
        # it's the real global ID (or None, if this local track has none
        # yet — a legitimate result, not a failure); on RPC failure it's the
        # local-ID fallback's negative int. Both are valid "some ID or None"
        # answers from this method's caller's point of view.
        result = self._client.call(
            "get_global_id", camera_id=camera_id, local_track_id=local_track_id
        )
        return result.value

    def find_global_track_by_identity(self, identity: str) -> Optional[int]:
        """Returns the global_id directly, NOT a GlobalTrack object — see
        gpu_rpc.py's module docstring for why the server already projects
        this down before it crosses the wire. CameraEngine's only call site
        (camera_engine.py, the "Global ID reassignment" block) reads exactly
        `.global_id` off the real method's return value, so callers of this
        adapter must be written the same way today's CameraEngine is: treat
        the return value as the global ID itself, not an object to unwrap.
        """
        result = self._client.call("find_global_track_by_identity", identity)
        return result.value

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
        # Real method returns int | None; PersonTracker never checks it. One-way.
        self._client.call_one_way(
            "on_track_removed",
            camera_id=camera_id,
            local_track_id=local_track_id,
            track_history=track_history,
            total_frames=total_frames,
        )
