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

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from workers.gpu_rpc import GpuRpcClient


@dataclass(frozen=True)
class GlobalTrackRef:
    """The one field CameraEngine reads off find_global_track_by_identity's
    result (`existing_global.global_id`, camera_engine.py's "Global ID
    reassignment" block). The real method returns a full GlobalTrack; the
    server projects it to this int on the wire (see gpu_rpc.py), and the
    adapter re-wraps it here so the call site's attribute access works
    unchanged — returning the bare int instead would crash CameraEngine with
    AttributeError on `.global_id`, breaking the drop-in contract this whole
    module exists to keep.
    """

    global_id: int


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
        """On RPC failure this returns None ("unknown"), NOT the client's
        negative-ID fallback. The negative fallback is only meaningful for
        assign_global_id, whose contract is "give me an ID to use"; this
        method's contract is "tell me what exists," and its caller
        (camera_engine.py, building the recognized_persons payload) already
        treats None as a legitimate answer — a fabricated negative ID would
        instead flow to the backend as if it were a real assignment.
        """
        result = self._client.call(
            "get_global_id", camera_id=camera_id, local_track_id=local_track_id
        )
        return result.value if result.ok else None

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
        # Real method returns int | None; PersonTracker never checks it. One-way.
        self._client.call_one_way(
            "on_track_removed",
            camera_id=camera_id,
            local_track_id=local_track_id,
            track_history=track_history,
            total_frames=total_frames,
        )
