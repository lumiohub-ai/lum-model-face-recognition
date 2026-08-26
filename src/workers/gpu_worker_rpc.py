"""IPC to the main process's GPUInferenceWorker, over a Unix domain socket.

Same shape as gpu_rpc.py's GlobalTrackManager bridge — a separate module
because this is a different real object with its own method surface, not a
generalisation of that one. Shares only the wire-level framing (rpc_framing.py).

## Why the frame/ROI payloads carry a handle, not pixels, but the responses
## don't need one

The *request* side reuses frame_store.py's shared-memory transport
(FrameHandle / RoiBatchHandle) — a 720p frame costs ~15.6ms through a
plain socket/Redis payload, more than the YOLO inference itself, so pixels
must never touch this RPC's wire format on the way in.

The *response* side (detections, embeddings) is different: these are small
structured dicts of scalars/lists, plus one numpy array field each
(`embedding`, and `face_image` — a face-sized crop, not a full frame or
person ROI). Measured directly rather than assumed: a realistic embed
response for 3 tracks with 150x150 face crops pickles to ~68 KiB and
round-trips over this module's actual socket implementation in
p50=0.43ms / p95=0.78ms; even a generous 250x200 crop (~149 KiB) comes in at
p50=0.82ms / p95=1.08ms. Both are negligible against the ~66ms per-frame
budget, so the response goes straight through the socket as a plain pickled
value — no second shared-memory direction needed.

## Surface

Mirrors GPUInferenceWorker's own public API exactly (`submit_frame` +
`get_detections` collapse into one round-trip here, since there is no
reason to split them into two RPC calls the way they're two in-process
queue operations — nothing else can interleave between them for one
caller, unlike the in-process version where other cameras' submits could
land between them on shared queues):

  - detect(camera_id, frame_handle, frame_num) -> List[Dict]
  - embed(camera_id, roi_batch_handle)         -> Dict[int, Dict]

Failure mode: on any failure, `detect` degrades to `[]` (no detections this
cycle — the tracker ages the existing tracks by one frame, same as a
same-process `get_detections` timeout does today) and `embed` degrades to
`{}` (no embeddings this cycle — recognition just doesn't advance for any
track this interval, same as `get_embeddings` timing out today). Neither
fabricates a plausible-looking result; both must be visibly "nothing" so a
struggling main process doesn't quietly present matches or boxes it never
computed.
"""

from __future__ import annotations

import os
import pickle
import socket
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from workers import frame_store
from workers.rpc_framing import recv_framed, send_framed

DEFAULT_SOCKET_PATH = os.environ.get(
    "SO_GPU_WORKER_RPC_SOCKET", "/tmp/lumiohub-gpu-worker-rpc.sock"
)

# The server's own worst case is GPUInferenceWorker.get_detections/
# get_embeddings' internal 2.0s timeout (gpu_worker.py), after which it still
# sends a valid empty reply. The client's socket timeout sits ABOVE that, not
# equal to it: at exactly 2.0s the client's recv deadline races the server's
# legitimate timed-out-but-empty answer, and losing that race logs a spurious
# transport failure (and fires on_fallback) for what was really the GPU
# worker's own, correctly-reported timeout.
DEFAULT_TIMEOUT_S = 2.5


@dataclass(frozen=True)
class DetectRequest:
    camera_id: int
    frame_handle: frame_store.FrameHandle
    frame_num: int


@dataclass(frozen=True)
class EmbedRequest:
    camera_id: int
    roi_batch_handle: frame_store.RoiBatchHandle


class GpuWorkerRpcClient:
    """Worker-side: submit a frame/ROI-batch handle to the main process's
    GPUInferenceWorker and get back detections/embeddings. Never raises —
    degrades to an empty result, matching what a same-process timeout
    already does today (see module docstring).
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._socket_path = socket_path
        self._timeout_s = timeout_s
        # Metrics hook, same convention as gpu_rpc.GpuRpcClient.on_fallback —
        # a struggling main process should be visible as a failure-rate
        # metric, not only inferable from every camera going quiet.
        self.on_fallback: Optional[Any] = None

    def _call(self, kind: str, request: Any) -> Any:
        payload = pickle.dumps((kind, request), protocol=pickle.HIGHEST_PROTOCOL)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self._timeout_s)
            sock.connect(self._socket_path)
            send_framed(sock, payload)
            response = pickle.loads(recv_framed(sock))
        if isinstance(response, Exception):
            raise response
        return response

    def detect(
        self, camera_id: int, frame_handle: frame_store.FrameHandle, frame_num: int
    ) -> List[Dict]:
        try:
            return self._call(
                "detect", DetectRequest(camera_id, frame_handle, frame_num)
            )
        except Exception as e:
            logger.warning(
                f"gpu_worker_rpc: detect(camera_id={camera_id}, "
                f"frame_num={frame_num}) failed: {type(e).__name__}: {e} "
                f"— no detections this cycle"
            )
            if self.on_fallback is not None:
                self.on_fallback()
            return []

    def embed(
        self, camera_id: int, roi_batch_handle: frame_store.RoiBatchHandle
    ) -> Dict[int, Dict]:
        try:
            return self._call("embed", EmbedRequest(camera_id, roi_batch_handle))
        except Exception as e:
            logger.warning(
                f"gpu_worker_rpc: embed(camera_id={camera_id}, "
                f"seq={roi_batch_handle.seq}) failed: {type(e).__name__}: {e} "
                f"— no embeddings this cycle"
            )
            if self.on_fallback is not None:
                self.on_fallback()
            return {}


class GpuWorkerRpcServer:
    """Main-process side: owns the real GPUInferenceWorker, serves
    detect/embed requests from camera worker processes.

    Frame/ROI pixels never pass through this class — it reads them out of
    shared memory via frame_store, then calls straight into the existing
    submit_frame/get_detections and submit_faces/get_embeddings pair, which
    already implement the correlation and drop-handling logic (LSO-138).
    This class adds no new synchronisation of its own.
    """

    def __init__(self, gpu_worker, socket_path: str = DEFAULT_SOCKET_PATH):
        self._gpu_worker = gpu_worker
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
            target=self._serve_forever, daemon=True, name="gpu-worker-rpc-server"
        )
        self._thread.start()
        logger.info(f"GpuWorkerRpcServer: listening on {self._socket_path}")

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

    def _dispatch(self, kind: str, request: Any) -> Any:
        if kind == "detect":
            frame = frame_store.attach_and_read(request.frame_handle)
            if frame is None:
                # The camera's frame slot vanished between write and this
                # read (camera removed mid-flight) — same "not registered"
                # case GPUInferenceWorker.submit_frame already tolerates.
                return []
            if not self._gpu_worker.submit_frame(
                request.camera_id, frame, request.frame_num
            ):
                # Dropped (queue full even after drop-oldest, or camera
                # unregistered): no reply will EVER be produced for this
                # frame, so waiting on get_detections would burn its full
                # timeout for nothing. This is LSO-138's exact contract —
                # camera_worker.py skips its get on False for the same
                # reason — and drops happen precisely under load, when a
                # 2s stall per dropped frame hurts most.
                return []
            return self._gpu_worker.get_detections(request.camera_id, request.frame_num)

        if kind == "embed":
            crops = frame_store.attach_and_read_roi_batch(request.roi_batch_handle)
            if crops is None:
                return {}
            person_rois = [c for _tid, c in crops]
            track_ids = [tid for tid, _c in crops]
            # get_embeddings must be given the seq submit_faces RETURNS —
            # the GPU worker issues its own per-camera counter (LSO-138:
            # "there is no caller-side id to reuse, so the worker issues
            # one"). RoiBatchHandle.seq is a DIFFERENT counter
            # (RoiBatchSlot's own); the two only coincide while both
            # processes live in lockstep forever. After a camera-worker
            # restart the slot's counter resets while the GPU worker's
            # keeps counting, and _await_response treats the mismatch as
            # a newer-reply-supersedes case — every embed call would
            # return {} permanently from then on.
            seq = self._gpu_worker.submit_faces(
                request.camera_id, person_rois, track_ids
            )
            if seq is None:
                # Camera no longer registered — same tolerated case as the
                # detect path's missing slot.
                return {}
            return self._gpu_worker.get_embeddings(request.camera_id, seq)

        raise ValueError(f"gpu_worker_rpc: refusing unrecognised request kind {kind!r}")

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                kind, request = pickle.loads(recv_framed(conn))
                response: Any = self._dispatch(kind, request)
            except Exception as e:
                logger.exception(f"GpuWorkerRpcServer: request failed: {e}")
                response = e
            try:
                send_framed(conn, pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
            except OSError:
                pass  # client already gave up (its own timeout fired first)
