"""IPC to the main process's GlobalTrackManager, over a Unix domain socket.

Why a Unix socket and not Redis pub/sub: an uncorrelated async reply queue
can silently desync a camera on timeout. Re-deriving a correlation scheme for
a second, higher-frequency call site (every detection interval, per camera)
is not worth it. A Unix socket gives a direct call/response with a
library-level timeout and no separate correlation protocol to get wrong.

Why this stays a narrow RPC instead of moving GlobalTrackManager's state into
Redis: `assign_global_id` does numpy similarity matching over the whole
embedding gallery, which doesn't decompose into Redis operations cleanly.
Keeping the dicts and the ReID model in the main process preserves the
existing locking work verbatim and needs no lum_vision rewrite — the cost is
one small control message per call. Measured against this module's own implementation,
one call carrying a realistic payload (a person crop + a 512-float embedding)
round-trips in p50=0.16ms / p95=0.34ms over a local Unix socket — negligible
against the ~66ms per-frame budget (configs/config.yaml's
fps_alert_threshold: 15.0).

Failure mode (must be a real design, not an afterthought — this is now a
single point of failure for identity assignment specifically, not for camera
tracking as a whole): on any failure — timeout, connection refused, or a bug
raised inside GlobalTrackManager itself — a blocking call's caller gets a
local-only ID from its own negative-ID counter, distinguishable at a glance
from a real global ID (which starts at 1000 and counts up — see
lum_vision.person_tracking.global_track.GlobalTrackManager.global_id_counter
and lum_vision.person_tracking.ids.GlobalTrackIDGenerator). The camera keeps
tracking and voting locally; nothing blocks. Reconciling a local-only ID into its real global ID once the main process
recovers is a later concern, not solved here — this only needs the failure
to be non-blocking and visible.

## RPC surface

`camera_engine.py` calls seven GlobalTrackManager methods. Three return a
value the caller actually branches on and therefore need request/reply:

  - assign_global_id(...)                     -> int
  - get_global_id(camera_id, local_track_id)  -> int | None
  - find_global_track_by_identity(identity)   -> int | None

The last one is a deliberate narrowing: the real method returns a full
`GlobalTrack` object (live numpy embedding arrays, nested dataclasses), but
the only caller (camera_engine.py) reads exactly one field off it —
`.global_id`. Shipping the whole object would be wasteful and would invite a
future caller to read a field from what is, by the time it arrives, already a
stale snapshot. The server-side handler projects the object down to that one
field before it ever reaches the wire.

The rest are called today with their return value ignored — neither
camera_engine.py nor PersonTracker branches on them — so they are one-way
(fire-and-forget): sending never blocks a camera frame, and a dropped one-way
call degrades silently to "this camera's view of identity state is very
slightly behind," which is what already happens today whenever the periodic
validator or another camera's thread wins a race on the same unlocked dict —
a lost one-way RPC message reopens the same class of staleness across the
process boundary. Acceptable, called out here so it isn't mistaken for solved.

  - on_face_detected(camera_id, local_track_id, quality=0.0, recognized=False, identity=None) -> None
  - on_face_not_visible(camera_id, local_track_id)                                             -> None
  - update_global_track_identity(global_id, identity, locked=True)                              -> bool (ignored)
  - reassign_local_track(camera_id, local_track_id, new_global_id)                              -> bool (ignored)
  - on_track_created(camera_id, local_track_id, bbox=None, frame_num=0)                         -> None
  - on_track_update(camera_id, local_track_id)                                                  -> None
  - on_track_removed(camera_id, local_track_id, track_history=None, total_frames=0)             -> int | None (ignored)

The last three are called from `PersonTracker`, not `CameraEngine` — found by
tracing PersonTracker's own `global_track_manager` constructor argument, a
second direct holder of the reference besides CameraEngine's. `PersonTracker`
also calls `global_id_generator.get_next_id()` directly; that is a *different*
object (`GlobalTrackIDGenerator`, an in-process `threading.Lock` counter) and
a different fix — Redis `INCR`, not this RPC — tracked separately, not here.
"""

from __future__ import annotations

import os
import pickle
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from workers.rpc_framing import recv_framed, send_framed

# /run/lumiohub, not /tmp: main and the camera worker are separate containers
# with separate /tmp, so a socket there is unreachable. compose.yml mounts a
# shared volume here. The env override exists for tests and for pointing at a
# deliberately-bogus path when verifying the fallback actually fires.
DEFAULT_SOCKET_PATH = os.environ.get(
    "SO_GPU_RPC_SOCKET", "/run/lumiohub/gpu-rpc.sock"
)

# Per detection-interval call, not per video frame — same cadence as today's
# in-process assign_global_id. 250ms is generous relative to the ~66ms
# per-frame budget (configs/config.yaml's fps_alert_threshold: 15.0) because
# a timeout here degrades to a local ID rather than dropping the frame; it
# does not need to be as tight as the frame budget itself.
DEFAULT_TIMEOUT_S = 0.25

# The methods a one-way call is allowed to name. A closed set rather than
# "any string" — the server executes these by attribute lookup on a live
# GlobalTrackManager, so an open set would let a caller invoke arbitrary
# methods (including private/mutating ones never meant to cross this RPC)
# by simply naming them.
_BLOCKING_METHODS = frozenset(
    {"assign_global_id", "get_global_id", "find_global_track_by_identity"}
)
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


def _format_call(method: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
    """Render a call for a log line — includes kwargs, not just positional
    args. camera_engine.py calls these methods almost entirely by keyword, so
    logging only `args` would render every failure as an uninformative
    `method_name()`.
    """
    parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    return f"{method}({', '.join(parts)})"


@dataclass(frozen=True)
class MethodCallRequest:
    """One call to a GlobalTrackManager method, by name.

    `one_way=True` tells the server not to bother sending a reply (the
    client that sent it isn't listening for one either — see `call_one_way`)
    and tells the client not to wait for one.
    """

    method: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    one_way: bool = False


@dataclass(frozen=True)
class MethodCallResult:
    value: Any
    ok: bool  # False when this came from the local-ID/None fallback


class _LocalIdFallback:
    """Per-process negative-ID counter, used only when the main process is
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


class GpuRpcClient:
    """Worker-side: call the main process's GlobalTrackManager.

    Blocking calls (`call`) fall back to a local-only ID on any failure
    rather than blocking the camera or raising. One-way calls (`call_one_way`)
    fire and forget — a failure is logged but never surfaces to the caller,
    since camera_engine.py never checked their return values even when they
    ran in-process.
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
        # ANY blocking-call fallback — a socket timeout, a connection
        # refusal, or a bug raised inside GlobalTrackManager itself — since
        # all three mean the camera got no usable answer. Not fired for
        # one-way call failures; those are logged only, per the class
        # docstring. Kept as a plain callable, not a MetricsCollector import,
        # so this module has no dependency on src/infrastructure.
        self.on_fallback: Optional[Any] = None

    def _send_and_maybe_recv(
        self, request: MethodCallRequest
    ) -> Optional[MethodCallResult]:
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self._timeout_s)
            sock.connect(self._socket_path)
            send_framed(sock, payload)
            if request.one_way:
                return None
            response = pickle.loads(recv_framed(sock))
        if isinstance(response, Exception):
            raise response
        return MethodCallResult(value=response, ok=True)

    def call(self, method: str, *args: Any, **kwargs: Any) -> MethodCallResult:
        """Blocking call. Never raises — see the failure-mode note above."""
        if method not in _BLOCKING_METHODS:
            raise ValueError(
                f"gpu_rpc.call: {method!r} is not a recognised blocking method "
                f"(expected one of {sorted(_BLOCKING_METHODS)}); did you mean "
                f"call_one_way?"
            )
        request = MethodCallRequest(method=method, args=args, kwargs=kwargs)
        try:
            result = self._send_and_maybe_recv(request)
            assert result is not None  # one_way=False always returns
            return result
        except Exception as e:
            logger.warning(
                f"gpu_rpc: {_format_call(method, args, kwargs)} failed: "
                f"{type(e).__name__}: {e} — falling back to a local-only ID"
            )
            if self.on_fallback is not None:
                self.on_fallback()
            return MethodCallResult(value=self._fallback.next_id(), ok=False)

    def call_one_way(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Fire-and-forget. A failure is logged, never raised or returned —
        matches camera_engine.py's existing behavior of not checking these
        methods' return values even when they ran in-process.
        """
        if method not in _ONE_WAY_METHODS:
            raise ValueError(
                f"gpu_rpc.call_one_way: {method!r} is not a recognised one-way "
                f"method (expected one of {sorted(_ONE_WAY_METHODS)}); did you "
                f"mean call?"
            )
        request = MethodCallRequest(method=method, args=args, kwargs=kwargs, one_way=True)
        try:
            self._send_and_maybe_recv(request)
        except Exception as e:
            logger.warning(
                f"gpu_rpc: one-way {_format_call(method, args, kwargs)} failed: "
                f"{type(e).__name__}: {e} — dropped, no fallback (caller does "
                f"not use this call's result)"
            )


class GpuRpcServer:
    """Main-process side: owns the real GlobalTrackManager, serves requests
    from worker processes over a Unix domain socket.

    One request handled per connection — GlobalTrackManager's own RLock
    is what actually serialises concurrent callers; this server
    does not add a second layer of locking, it just marshals bytes to/from
    that already-thread-safe object and dispatches by method name from the
    fixed allow-list in _BLOCKING_METHODS / _ONE_WAY_METHODS.
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

    def _dispatch(self, request: MethodCallRequest) -> Any:
        """Execute one request against the real GlobalTrackManager.

        Raises on an unrecognised method name rather than falling through to
        a generic getattr — the allow-lists above are the actual security/
        correctness boundary (see GpuRpcServer's docstring), not just
        documentation.
        """
        if request.method not in _BLOCKING_METHODS and request.method not in _ONE_WAY_METHODS:
            raise ValueError(f"gpu_rpc: refusing unrecognised method {request.method!r}")

        method = getattr(self._manager, request.method)
        result = method(*request.args, **request.kwargs)

        if request.method == "find_global_track_by_identity":
            # Project the live GlobalTrack down to the one field every
            # current caller reads — see the module docstring's "RPC
            # surface" section for why the full object never crosses.
            return result.global_id if result is not None else None
        return result

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                payload = recv_framed(conn)
                request: MethodCallRequest = pickle.loads(payload)
                response: Any = self._dispatch(request)
            except ConnectionError:
                # Connected then closed without sending: the compose
                # healthcheck probing that we are listening, or a client whose
                # timeout fired mid-handshake. Not exception-worthy — logging
                # a stack trace every probe would train readers to skim past
                # the one that matters.
                logger.debug("GpuRpcServer: connection closed before a request arrived")
                return
            except Exception as e:
                logger.exception(f"GpuRpcServer: request failed: {e}")
                response = e
                request = None  # type: ignore[assignment]

            if request is not None and request.one_way:
                return  # caller isn't reading a reply — see docstring
            try:
                send_framed(conn, pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
            except OSError:
                pass  # client already gave up (its own timeout fired first)
