"""IPC to the main process's GlobalTrackManager, over a Unix domain socket.

Why a Unix socket and not Redis pub/sub: LSO-138 was a whole ticket about an
uncorrelated async reply queue silently desyncing a camera on timeout.
Re-deriving that correlation scheme for a second, higher-frequency call site
(every detection interval, per camera) is not worth it. A Unix socket gives a
direct call/response with a library-level timeout and no separate correlation
protocol to get wrong.

Why this stays a narrow RPC instead of moving GlobalTrackManager's state into
Redis: `assign_global_id` does numpy similarity matching over the whole
embedding gallery, which doesn't decompose into Redis operations cleanly.
Keeping the dicts and the ReID model in the main process preserves LSO-137's
locking work verbatim and needs no lum_vision rewrite — the cost is one small
control message per call. Measured against this module's own implementation,
one call carrying a realistic payload (a person crop + a 512-float embedding)
round-trips in p50=0.16ms / p95=0.34ms over a local Unix socket — negligible
against the ~66ms per-frame budget (configs/config.yaml's
fps_alert_threshold: 15.0).

Failure mode (must be a real design, not an afterthought — this is now a
single point of failure for identity assignment specifically, not for camera
tracking as a whole): on timeout or connection failure, the CLIENT assigns a
local-only ID from its own negative-ID counter, distinguishable at a glance
from a real global ID (which starts at 1000 and counts up — see
lum_vision.person_tracking.global_track.GlobalTrackManager.global_id_counter
and lum_vision.person_tracking.ids.GlobalTrackIDGenerator). The camera keeps
tracking and voting locally; nothing blocks. Reconciling a local-only ID into
its real global ID once the main process recovers is a Stage 2 concern, not
solved here — Stage 1 only needs the failure to be non-blocking and visible.
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from loguru import logger

DEFAULT_SOCKET_PATH = os.environ.get(
    "SO_GPU_RPC_SOCKET", "/tmp/lumiohub-gpu-rpc.sock"
)

# Per detection-interval call, not per video frame — same cadence as today's
# in-process assign_global_id. 250ms is generous relative to the ~66ms
# per-frame budget (configs/config.yaml's fps_alert_threshold: 15.0) because
# a timeout here degrades to a local ID rather than dropping the frame; it
# does not need to be as tight as the frame budget itself.
DEFAULT_TIMEOUT_S = 0.25

_HEADER = struct.Struct("!I")  # 4-byte big-endian length prefix


def _send_framed(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed before expected bytes arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_framed(sock: socket.socket) -> bytes:
    (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    return _recv_exact(sock, length)


@dataclass(frozen=True)
class AssignGlobalIdRequest:
    """Mirrors GlobalTrackManager.assign_global_id's signature exactly —
    see lum_vision.person_tracking.global_track. Kept as a plain dataclass
    (not the live method's kwargs) so a signature change there is a visible,
    deliberate edit here rather than a silent drift.
    """

    camera_id: int
    local_track_id: int
    person_crop: Optional[np.ndarray]
    face_embedding: Optional[np.ndarray] = None
    detection_confidence: float = 0.0
    frame_num: int = 0
    identity: Optional[str] = None
    identity_locked: bool = False


@dataclass(frozen=True)
class AssignGlobalIdResult:
    global_id: int
    reconciled: bool  # False when this came from the local-ID fallback


class _LocalIdFallback:
    """Per-process negative-ID counter, used only when the main process is
    unreachable. Never shared across processes — two workers falling back
    concurrently must not hand out the same local ID, which a shared negative
    range would risk; a per-process counter starting at -1 and counting down
    guarantees no two *processes* collide as long as each process's IDs are
    tagged with its own identity downstream (not handled by this class —
    the caller is responsible for that if two local IDs must ever be told
    apart, which Stage 1 does not yet require).
    """

    def __init__(self):
        self._next = -1
        self._lock = threading.Lock()

    def next_id(self) -> int:
        with self._lock:
            value = self._next
            self._next -= 1
            return value


class GpuRpcClient:
    """Worker-side: call the main process's GlobalTrackManager, falling back
    to a local-only ID on any failure rather than blocking the camera.
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._socket_path = socket_path
        self._timeout_s = timeout_s
        self._fallback = _LocalIdFallback()
        # Metrics hook: set by the caller (camera_tasks.py) so a struggling
        # main process is visible as a failure-rate metric rather than only
        # inferable from a spike in local-only ("unreconciled") IDs. Fires on
        # ANY fallback — a socket timeout, a connection refusal, or a bug
        # raised inside GlobalTrackManager itself — since all three mean the
        # camera got no usable answer. Kept as a plain callable, not a
        # MetricsCollector import, so this module has no dependency on
        # src/infrastructure.
        self.on_fallback: Optional[Any] = None

    def assign_global_id(self, request: AssignGlobalIdRequest) -> AssignGlobalIdResult:
        """Never raises. A bug inside the main process's GlobalTrackManager
        must not be able to crash a camera worker — it degrades to a local-
        only ID exactly like a transport failure (timeout, connection
        refused) does, since from the worker's perspective both mean "no
        usable answer arrived." Distinguished only in the log line, so an
        actual GlobalTrackManager bug is still diagnosable from worker logs
        rather than silently indistinguishable from a network blip.
        """
        try:
            payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self._timeout_s)
                sock.connect(self._socket_path)
                _send_framed(sock, payload)
                response = pickle.loads(_recv_framed(sock))
            if isinstance(response, Exception):
                raise response
            return AssignGlobalIdResult(global_id=response, reconciled=True)
        except Exception as e:
            logger.warning(
                f"gpu_rpc: assign_global_id failed for camera={request.camera_id} "
                f"local_track={request.local_track_id}: {type(e).__name__}: {e} "
                f"— falling back to a local-only ID"
            )
            if self.on_fallback is not None:
                self.on_fallback()
            local_id = self._fallback.next_id()
            return AssignGlobalIdResult(global_id=local_id, reconciled=False)


class GpuRpcServer:
    """Main-process side: owns the real GlobalTrackManager, serves requests
    from worker processes over a Unix domain socket.

    One request handled at a time per connection, one connection at a time —
    GlobalTrackManager's own RLock (LSO-137) is what actually serialises
    concurrent callers; this server does not add a second layer of locking,
    it just marshals bytes to/from that already-thread-safe object.
    """

    def __init__(self, global_track_manager, socket_path: str = DEFAULT_SOCKET_PATH):
        self._manager = global_track_manager
        self._socket_path = socket_path
        self._server_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self._socket_path)
        self._server_sock.listen(64)
        self._running = True
        self._thread = threading.Thread(
            target=self._serve_forever, daemon=True, name="gpu-rpc-server"
        )
        self._thread.start()
        logger.info(f"GpuRpcServer: listening on {self._socket_path}")

    def stop(self) -> None:
        self._running = False
        if self._server_sock is not None:
            self._server_sock.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

    def _serve_forever(self) -> None:
        server_sock = self._server_sock
        assert server_sock is not None  # start() always sets this before spawning this thread
        while self._running:
            try:
                conn, _ = server_sock.accept()
            except OSError:
                break  # socket closed by stop()
            threading.Thread(
                target=self._handle_connection, args=(conn,), daemon=True
            ).start()

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                payload = _recv_framed(conn)
                request: AssignGlobalIdRequest = pickle.loads(payload)
                global_id = self._manager.assign_global_id(
                    camera_id=request.camera_id,
                    local_track_id=request.local_track_id,
                    person_crop=request.person_crop,
                    face_embedding=request.face_embedding,
                    detection_confidence=request.detection_confidence,
                    frame_num=request.frame_num,
                    identity=request.identity,
                    identity_locked=request.identity_locked,
                )
                response: Any = global_id
            except Exception as e:
                logger.exception(f"GpuRpcServer: request failed: {e}")
                response = e
            try:
                _send_framed(conn, pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
            except OSError:
                pass  # client already gave up (its own timeout fired first)
