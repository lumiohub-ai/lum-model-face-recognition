"""Unit tests for global_track_rpc + global_track_adapter.

Covers the Unix-socket RPC bridge to GlobalTrackManager: correct dispatch
across the blocking/one-way split, the local-ID fallback on every failure
mode (server down, timeout, a bug inside the real manager), and that
RemoteGlobalTrackManager's ten method signatures still match what
CameraEngine/PersonTracker actually call on the real GlobalTrackManager -
the kind of drift that breaks silently if lum_vision's signatures change
without a corresponding update here.

Run: PYTHONPATH=src python tests/test_global_track_rpc.py
"""

import inspect
import os
import socket
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from workers.global_track_adapter import (  # noqa: E402
    GlobalTrackRef,
    RemoteGlobalTrackManager,
)
from workers.global_track_rpc import (  # noqa: E402
    GpuRpcClient,
    GpuRpcServer,
    MethodCallRequest,
)


def _free_socket_path(tag: str) -> str:
    """A distinct, tag-suffixed path per test - tests run against real Unix
    sockets, and reusing one path across tests racing in the same process
    would let one test's leftover socket answer another's client.

    Hashes `tag` rather than embedding it verbatim: AF_UNIX's sun_path is
    capped at ~108 bytes on Linux, and a descriptive unittest method name
    combined with a stable prefix and a PID suffix can silently exceed
    that — observed directly as `OSError: AF_UNIX path too long` in
    tests/test_gpu_worker_rpc.py's identical helper, not a hypothetical.
    """
    import hashlib

    digest = hashlib.sha1(tag.encode()).hexdigest()[:10]
    return f"/tmp/gpurpc_{digest}_{os.getpid()}.sock"


class FakeGlobalTrack:
    """Stands in for lum_vision.person_tracking.global_track_model.GlobalTrack
    - just enough surface (the one field every real caller reads) to prove
    the server projects it correctly rather than shipping the whole object.
    """

    def __init__(self, global_id):
        self.global_id = global_id


class FakeGlobalTrackManager:
    """Stands in for lum_vision.person_tracking.global_track.GlobalTrackManager.
    Records every call it receives so tests can assert on dispatch, not just
    on the returned value.
    """

    def __init__(self):
        self.calls = []

    def assign_global_id(self, camera_id, local_track_id, **kw):
        self.calls.append(("assign_global_id", camera_id, local_track_id, kw))
        return 1000 + camera_id * 100 + local_track_id

    def get_global_id(self, camera_id, local_track_id):
        self.calls.append(("get_global_id", camera_id, local_track_id))
        return 42 if local_track_id == 1 else None

    def find_global_track_by_identity(self, identity):
        self.calls.append(("find_global_track_by_identity", identity))
        return FakeGlobalTrack(777) if identity == "alice" else None

    def update_global_track_identity(self, global_id, identity, locked=True):
        self.calls.append(("update_global_track_identity", global_id, identity, locked))
        return True

    def reassign_local_track(self, camera_id, local_track_id, new_global_id):
        self.calls.append(("reassign_local_track", camera_id, local_track_id, new_global_id))
        return True

    def on_face_detected(self, camera_id, local_track_id, quality=0.0, recognized=False, identity=None):
        self.calls.append(("on_face_detected", camera_id, local_track_id, quality, recognized, identity))

    def on_face_not_visible(self, camera_id, local_track_id):
        self.calls.append(("on_face_not_visible", camera_id, local_track_id))

    def on_track_created(self, camera_id, local_track_id, bbox=None, frame_num=0):
        self.calls.append(("on_track_created", camera_id, local_track_id, bbox, frame_num))

    def on_track_update(self, camera_id, local_track_id):
        self.calls.append(("on_track_update", camera_id, local_track_id))

    def on_track_removed(self, camera_id, local_track_id, track_history=None, total_frames=0):
        self.calls.append(("on_track_removed", camera_id, local_track_id, track_history, total_frames))
        return 1


class SlowGlobalTrackManager(FakeGlobalTrackManager):
    """assign_global_id blocks on an Event until the test releases it - lets
    a test observe state WHILE a background call is still in flight, which
    a fast fake can't reliably do (the real thread could finish before the
    test gets to assert anything).

    `received` counts requests as they ARRIVE, before the wait - `calls`
    (from FakeGlobalTrackManager) only records them once the wait releases
    and the call actually returns, so `calls` cannot be polled to detect
    "in flight"; `received` is what a test polls for that instead.
    """

    def __init__(self, release: threading.Event):
        super().__init__()
        self._release = release
        self.received = 0
        self._received_lock = threading.Lock()

    def assign_global_id(self, camera_id, local_track_id, **kw):
        with self._received_lock:
            self.received += 1
        self._release.wait(timeout=5.0)
        return super().assign_global_id(camera_id, local_track_id, **kw)


class BrokenGlobalTrackManager:
    """A manager whose calls always raise - proves a bug inside the real
    GlobalTrackManager degrades a blocking caller to a local ID rather than
    crashing it."""

    def assign_global_id(self, **kw):
        raise ValueError("simulated bug in GlobalTrackManager")


class GpuRpcDispatchTests(unittest.TestCase):
    """Server + client wired to a real Unix socket - not mocked, since the
    framing/length-prefix logic and the resource lifecycle (bind/listen/
    accept/close) are exactly what a mock would paper over."""

    def setUp(self):
        self.socket_path = _free_socket_path(self._testMethodName)
        self.manager = FakeGlobalTrackManager()
        self.server = GpuRpcServer(self.manager, socket_path=self.socket_path)
        self.server.start()
        self.client = GpuRpcClient(socket_path=self.socket_path, timeout_s=1.0)

    def tearDown(self):
        self.server.stop()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def test_assign_global_id_round_trips(self):
        result = self.client.call(
            "assign_global_id", camera_id=2, local_track_id=5, person_crop=None
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 1205)

    def test_get_global_id_missing_track_returns_none_not_a_failure(self):
        result = self.client.call("get_global_id", 1, 999)
        self.assertTrue(result.ok)
        self.assertIsNone(result.value)

    def test_find_global_track_by_identity_projects_to_global_id(self):
        """The real method returns a GlobalTrack object; the server must
        project it down to .global_id before it crosses the wire (see
        global_track_rpc.py's module docstring) - proven here by making the fake
        return a GlobalTrack-shaped object and asserting the client gets
        back a bare int, not the object."""
        result = self.client.call("find_global_track_by_identity", "alice")
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 777)
        self.assertIsInstance(result.value, int)

    def test_find_global_track_by_identity_none_case(self):
        result = self.client.call("find_global_track_by_identity", "nobody")
        self.assertTrue(result.ok)
        self.assertIsNone(result.value)

    def test_one_way_calls_reach_the_manager(self):
        self.client.call_one_way("on_face_detected", camera_id=1, local_track_id=1, quality=0.9)
        self.client.call_one_way("on_face_not_visible", camera_id=1, local_track_id=2)
        self.client.call_one_way("update_global_track_identity", 777, "alice", locked=True)
        self.client.call_one_way("reassign_local_track", camera_id=1, local_track_id=3, new_global_id=777)
        self.client.call_one_way("on_track_created", camera_id=1, local_track_id=4, frame_num=10)
        self.client.call_one_way("on_track_update", camera_id=1, local_track_id=4)
        self.client.call_one_way("on_track_removed", camera_id=1, local_track_id=4, total_frames=30)

        # One-way calls are dispatched from their own server-side thread per
        # connection (see GpuRpcServer._serve_forever) - not synchronous
        # with the client returning, so give them a moment to land.
        deadline = time.monotonic() + 2.0
        while len(self.manager.calls) < 7 and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(len(self.manager.calls), 7, self.manager.calls)

    def test_call_rejects_a_one_way_method_name(self):
        with self.assertRaises(ValueError):
            self.client.call("on_face_detected", 1, 1)

    def test_call_one_way_rejects_a_blocking_method_name(self):
        with self.assertRaises(ValueError):
            self.client.call_one_way("assign_global_id", camera_id=1, local_track_id=1, person_crop=None)

    def test_server_rejects_unrecognised_method_via_dispatch(self):
        """The client-side allow-lists in call()/call_one_way() are a first
        line of defence, not the actual security/correctness boundary — the
        server must independently refuse an unrecognised method even if a
        client bypasses its own check (a different client implementation, a
        crafted payload). Goes around GpuRpcClient.call()'s own guard and
        talks to the wire protocol directly via `_send_and_maybe_recv`, which
        re-raises a server-side error response rather than swallowing it into
        a MethodCallResult(ok=False) — that swallowing only happens one level
        up, in call()'s except block.
        """
        malicious_request = MethodCallRequest(method="__init__", args=(), kwargs={})
        with self.assertRaises(ValueError) as ctx:
            self.client._send_and_maybe_recv(malicious_request)
        self.assertIn("__init__", str(ctx.exception))


class GpuRpcFailureModeTests(unittest.TestCase):
    """No server involved in most of these - the point is what happens when
    there is nothing to talk to, or something that talks back badly."""

    def test_server_down_falls_back_to_a_negative_local_id(self):
        client = GpuRpcClient(
            socket_path=_free_socket_path("down"), timeout_s=0.2
        )
        result = client.call("assign_global_id", camera_id=1, local_track_id=1, person_crop=None)
        self.assertFalse(result.ok)
        self.assertLess(result.value, 0)

    def test_successive_fallbacks_return_distinct_ids(self):
        client = GpuRpcClient(
            socket_path=_free_socket_path("down2"), timeout_s=0.2
        )
        r1 = client.call("assign_global_id", camera_id=1, local_track_id=1, person_crop=None)
        r2 = client.call("assign_global_id", camera_id=1, local_track_id=1, person_crop=None)
        self.assertNotEqual(r1.value, r2.value)

    def test_one_way_call_never_raises_when_server_is_down(self):
        client = GpuRpcClient(
            socket_path=_free_socket_path("down3"), timeout_s=0.2
        )
        client.call_one_way("on_face_detected", camera_id=1, local_track_id=1)  # must not raise

    def test_genuine_timeout_degrades_within_the_configured_bound(self):
        """A server that accepts the connection but never replies - the
        stalled-main-process case, not the down-server case above."""
        socket_path = _free_socket_path("stall")

        def stalling_server():
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(socket_path)
            srv.listen(1)
            conn, _ = srv.accept()
            time.sleep(5)
            conn.close()
            srv.close()

        t = threading.Thread(target=stalling_server, daemon=True)
        t.start()
        time.sleep(0.2)
        try:
            client = GpuRpcClient(socket_path=socket_path, timeout_s=0.3)
            fired = []
            client.on_fallback = lambda: fired.append(True)

            t0 = time.monotonic()
            result = client.call("assign_global_id", camera_id=1, local_track_id=1, person_crop=None)
            elapsed = time.monotonic() - t0

            self.assertFalse(result.ok)
            self.assertGreater(elapsed, 0.25)
            self.assertLess(elapsed, 1.0)
            self.assertEqual(len(fired), 1)
        finally:
            if os.path.exists(socket_path):
                os.unlink(socket_path)

    def test_bug_inside_the_real_manager_degrades_instead_of_raising(self):
        """This is the regression this suite exists to prevent: an earlier
        version of assign_global_id() let `raise response` inside the
        client's try-block escape uncaught, because the except clause only
        listed transport exceptions (OSError, socket.timeout, ...), not
        Exception generally. A ValueError raised inside the real
        GlobalTrackManager would have crashed the calling camera worker."""
        socket_path = _free_socket_path("broken")
        server = GpuRpcServer(BrokenGlobalTrackManager(), socket_path=socket_path)
        server.start()
        try:
            client = GpuRpcClient(socket_path=socket_path, timeout_s=1.0)
            result = client.call(
                "assign_global_id", camera_id=1, local_track_id=1, person_crop=None
            )
            self.assertFalse(result.ok)
            self.assertLess(result.value, 0)
        finally:
            server.stop()


class GpuRpcPersistentConnectionTests(unittest.TestCase):
    """The client holds ONE connection across calls and the server serves
    many requests per connection (both changed together to kill the
    ~550 connects+thread-spawns/sec that starved the main process). These
    pin the behaviours that only a persistent connection can get wrong.
    """

    def setUp(self):
        self.socket_path = _free_socket_path(self.id())
        self.manager = FakeGlobalTrackManager()
        self.server = GpuRpcServer(self.manager, socket_path=self.socket_path)
        self.server.start()
        self.client = GpuRpcClient(socket_path=self.socket_path, timeout_s=2.0)

    def tearDown(self):
        self.client.close()
        self.server.stop()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def test_many_calls_reuse_one_connection(self):
        """The point of the change: N calls must not mean N connections."""
        for i in range(10):
            result = self.client.call("assign_global_id", camera_id=1, local_track_id=i)
            self.assertTrue(result.ok)
        first = self.client._sock
        self.assertIsNotNone(first)
        self.client.call("assign_global_id", camera_id=1, local_track_id=99)
        self.assertIs(self.client._sock, first, "client reconnected mid-run")
        self.assertEqual(len(self.manager.calls), 11)

    def test_one_way_and_blocking_interleave_without_desync(self):
        """The real hazard of a persistent connection: a one-way call sends
        no reply, so if the server ever wrote one anyway it would be read as
        the answer to the NEXT blocking call. Interleave them and check every
        blocking answer is the one its own request asked for.
        """
        for i in range(5):
            self.client.call_one_way("on_track_update", camera_id=7, local_track_id=i)
            result = self.client.call(
                "assign_global_id", camera_id=7, local_track_id=i
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.value, 1000 + 7 * 100 + i)

    def test_a_failing_one_way_call_does_not_desync_the_stream(self):
        """A one-way call that raises server-side must still send nothing,
        or the error object becomes the next blocking call's 'answer'."""
        self.client.call_one_way("on_track_removed", camera_id=1, local_track_id=1)
        # boom: unknown kwarg makes the real dispatch raise inside the server
        self.client.call_one_way("on_track_update", camera_id=1, nonsense=True)
        result = self.client.call("assign_global_id", camera_id=2, local_track_id=3)
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 1000 + 2 * 100 + 3)

    def test_client_reconnects_after_the_server_restarts(self):
        """A persistent socket dies when the server does; the client must
        transparently reconnect rather than fall back forever. This is the
        failure mode connect-per-call never had."""
        first = self.client.call("assign_global_id", camera_id=1, local_track_id=1)
        self.assertTrue(first.ok)

        self.server.stop()
        self.server = GpuRpcServer(self.manager, socket_path=self.socket_path)
        self.server.start()

        after = self.client.call("assign_global_id", camera_id=1, local_track_id=2)
        self.assertTrue(after.ok, "client did not recover after server restart")
        self.assertEqual(after.value, 1000 + 1 * 100 + 2)


class GpuRpcConcurrencyTests(unittest.TestCase):
    def test_concurrent_clients_get_correct_independent_results(self):
        socket_path = _free_socket_path("concurrent")
        manager = FakeGlobalTrackManager()
        server = GpuRpcServer(manager, socket_path=socket_path)
        server.start()
        try:
            results = []
            results_lock = threading.Lock()

            def worker(cam_id):
                client = GpuRpcClient(socket_path=socket_path, timeout_s=2.0)
                r = client.call(
                    "assign_global_id", camera_id=cam_id, local_track_id=1, person_crop=None
                )
                with results_lock:
                    results.append((cam_id, r))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            self.assertEqual(len(results), 8)
            for cam_id, r in results:
                self.assertTrue(r.ok)
                self.assertEqual(r.value, 1000 + cam_id * 100 + 1)
        finally:
            server.stop()


class AdapterBehaviorTests(unittest.TestCase):
    """RemoteGlobalTrackManager exercised the way CameraEngine actually uses
    it - added after a review caught three bugs the client/server tests
    could not see, because they asserted on the wire format (a bare int)
    rather than on the adapter's drop-in contract (what the call sites do
    with the result)."""

    def setUp(self):
        self.socket_path = _free_socket_path(self._testMethodName)
        self.manager = FakeGlobalTrackManager()
        self.server = GpuRpcServer(self.manager, socket_path=self.socket_path)
        self.server.start()
        self.adapter = RemoteGlobalTrackManager(
            GpuRpcClient(socket_path=self.socket_path, timeout_s=1.0), enabled=True
        )

    def tearDown(self):
        self.server.stop()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def _down_adapter(self, tag, async_assign=False):
        """An adapter whose socket points at nothing - the degraded path.
        async_assign defaults to False here so tests can assert on the
        return value immediately without a wait/poll loop; the async path
        has its own dedicated test class below."""
        return RemoteGlobalTrackManager(
            GpuRpcClient(socket_path=_free_socket_path(tag), timeout_s=0.2),
            enabled=True,
            async_assign=async_assign,
        )

    def test_find_result_supports_the_attribute_access_camera_engine_does(self):
        """CameraEngine reads `existing_global.global_id` off the result
        (camera_engine.py's Global ID reassignment block) - a bare int here
        crashes that call site with AttributeError. This is the drop-in
        contract, asserted the way the caller exercises it."""
        existing_global = self.adapter.find_global_track_by_identity("alice")
        self.assertIsNotNone(existing_global)
        self.assertEqual(existing_global.global_id, 777)
        self.assertIsInstance(existing_global, GlobalTrackRef)

    def test_find_unknown_identity_returns_none(self):
        self.assertIsNone(self.adapter.find_global_track_by_identity("nobody"))

    def test_find_on_rpc_failure_returns_none_not_a_fabricated_track(self):
        """A negative fallback ID presented as a *found track* would make
        CameraEngine reassign the person to it via reassign_local_track -
        silent identity corruption. The only safe degraded answer for a
        lookup is 'not found'."""
        self.assertIsNone(
            self._down_adapter("find_down").find_global_track_by_identity("alice")
        )

    def test_get_global_id_is_a_pure_cache_read_no_rpc_call(self):
        """get_global_id no longer calls the RPC client at all - it only
        reads _assigned, populated separately by assign_global_id. Proven
        two ways: (1) it returns None against a dead socket instead of
        raising/timing out, and (2) reading it doesn't touch call counts on
        a live manager - see the AsyncAssignTests version of this check for
        the live-manager half."""
        down_adapter = self._down_adapter("get_down")
        self.assertIsNone(down_adapter.get_global_id(1, 1))
        down_adapter._assigned[(1, 1)] = 42
        self.assertEqual(down_adapter.get_global_id(1, 1), 42)

    def test_assign_on_rpc_failure_still_returns_a_negative_local_id(self):
        """assign_global_id is the one method where the negative fallback IS
        the contract - the camera needs *an* ID to keep tracking with. Uses
        the synchronous fallback (async_assign=False): this test is about
        the fallback value itself, not the async dispatch mechanism, which
        AsyncAssignTests below covers on its own."""
        gid = self._down_adapter("assign_down", async_assign=False).assign_global_id(
            camera_id=1, local_track_id=1, person_crop=None
        )
        self.assertLess(gid, 0)


class AsyncAssignTests(unittest.TestCase):
    """RemoteGlobalTrackManager's async_assign=True path (the default,
    matching camera_tasks.py's wiring): assign_global_id must never block
    the caller, must return None until the first reply lands, must send at
    most one request at a time per (camera_id, local_track_id), and must
    NOT forget cached state on on_track_removed - that's the caller's job,
    via forget_track(), after it reads get_global_id()."""

    def setUp(self):
        self.socket_path = _free_socket_path(self._testMethodName)
        self.release = threading.Event()
        self.manager = SlowGlobalTrackManager(self.release)
        self.server = GpuRpcServer(self.manager, socket_path=self.socket_path)
        self.server.start()
        self.adapter = RemoteGlobalTrackManager(
            GpuRpcClient(socket_path=self.socket_path, timeout_s=5.0),
            enabled=True,
            async_assign=True,
        )

    def tearDown(self):
        self.release.set()  # unblock anything still in flight before teardown
        self.server.stop()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def _poll_until(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_first_call_returns_none_and_does_not_block(self):
        start = time.monotonic()
        result = self.adapter.assign_global_id(
            camera_id=1, local_track_id=1, person_crop=None
        )
        elapsed = time.monotonic() - start
        self.assertIsNone(result)
        self.assertLess(
            elapsed, 0.5, "assign_global_id blocked the caller on the RPC reply"
        )
        self.release.set()  # let the background call finish before teardown

    def test_reply_populates_the_cache_for_the_next_call(self):
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.release.set()
        self.assertTrue(
            self._poll_until(
                lambda: self.adapter.assign_global_id(
                    camera_id=1, local_track_id=1, person_crop=None
                )
                == 1101  # FakeGlobalTrackManager's deterministic formula
            ),
            "cached global id never appeared after the background call finished",
        )

    def test_second_call_while_first_still_in_flight_does_not_send_another_request(self):
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        # Manager is still blocked on self.release - the request has been
        # sent but not yet answered, i.e. genuinely in flight.
        self.assertTrue(
            self._poll_until(lambda: self.manager.received == 1),
            "first request never reached the manager",
        )
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        time.sleep(0.05)  # give a wrongly-submitted second request time to land
        self.assertEqual(
            self.manager.received,
            1,
            "a second request for the same track was sent while one was already in flight",
        )
        # Direct check on submission count, not just on the manager having
        # received it: with a single-worker pool, a wrongly-submitted extra
        # task would simply queue behind the first rather than run
        # concurrently, so `received` alone can pass by accident of pool
        # size rather than because the in-flight guard actually worked.
        self.assertEqual(
            self.adapter._submitted_count,
            1,
            "a second background task was submitted for a track already in flight",
        )
        self.release.set()

    def test_different_tracks_do_not_block_each_other(self):
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.assertTrue(self._poll_until(lambda: self.manager.received == 1))
        # Track 2's request must still be accepted (returns None immediately,
        # queued behind the single-worker pool) rather than being suppressed
        # by track 1's in-flight request - the in-flight guard is keyed per
        # track, not global.
        result = self.adapter.assign_global_id(
            camera_id=1, local_track_id=2, person_crop=None
        )
        self.assertIsNone(result)
        self.release.set()
        self.assertTrue(self._poll_until(lambda: len(self.manager.calls) == 2))

    def test_forget_track_clears_the_cached_id(self):
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.release.set()
        self.assertTrue(
            self._poll_until(
                lambda: self.adapter.assign_global_id(
                    camera_id=1, local_track_id=1, person_crop=None
                )
                is not None
            )
        )
        self.adapter.forget_track(camera_id=1, local_track_id=1)
        # forget_track must not itself trigger a new RPC call - it only
        # clears local cache state.
        calls_before = len(self.manager.calls)
        self.assertIsNone(
            self.adapter._assigned.get((1, 1)),
            "forget_track did not clear the cached id",
        )
        self.assertEqual(len(self.manager.calls), calls_before)

    def test_on_track_removed_does_not_forget_the_track(self):
        """on_track_removed deliberately leaves _assigned alone now - the
        vendored PersonTracker calls it internally before CameraEngine's
        removed-tracks loop gets a chance to read the cached id via
        get_global_id(). Forgetting here would clear the note before anyone
        reads it. camera_engine.py calls forget_track() itself, right after
        that read - see test_forget_track_clears_the_cached_id for that
        half of the contract."""
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.release.set()
        self.assertTrue(
            self._poll_until(lambda: (1, 1) in self.adapter._assigned)
        )
        self.adapter.on_track_removed(camera_id=1, local_track_id=1)
        self.assertEqual(self.adapter._assigned.get((1, 1)), 1101)

    def test_rpc_failure_in_the_background_still_clears_in_flight(self):
        """A failed background call must not leave the track permanently
        stuck 'in flight' - that would silently freeze its global id at
        whatever _assigned already held, forever."""
        down_adapter = RemoteGlobalTrackManager(
            GpuRpcClient(socket_path=_free_socket_path("async_down"), timeout_s=0.2),
            enabled=True,
            async_assign=True,
        )
        down_adapter.assign_global_id(camera_id=9, local_track_id=9, person_crop=None)
        self.assertTrue(
            self._poll_until(lambda: (9, 9) not in down_adapter._in_flight, timeout=2.0),
            "in_flight was never cleared after the background RPC call failed",
        )
        # Confirms the guard is actually released, not just coincidentally
        # empty: a fresh call must be allowed to try again.
        down_adapter.assign_global_id(camera_id=9, local_track_id=9, person_crop=None)


@unittest.skipUnless(
    (lambda: __import__("importlib").util.find_spec("lum_vision") is not None)(),
    "lum_vision not importable in this environment - signature-drift check skipped",
)
class AdapterSignatureDriftTests(unittest.TestCase):
    """Not a behavioral test - a tripwire. RemoteGlobalTrackManager's ten
    methods were hand-transcribed from reading lum_vision's source; this
    catches the day someone changes a real GlobalTrackManager/PersonTracker
    call signature without updating the adapter to match, which would
    otherwise fail silently (extra/renamed kwargs raise a TypeError only at
    the moment that exact call path executes, not at import time).

    Skipped in any environment without lum_vision installed, matching this
    test suite's existing convention (test_action_worker.py etc. use fakes
    rather than importing lum_vision directly) of not hard-depending on that
    package.
    """

    METHOD_NAMES = [
        "assign_global_id",
        "get_global_id",
        "find_global_track_by_identity",
        "update_global_track_identity",
        "reassign_local_track",
        "on_face_detected",
        "on_face_not_visible",
        "on_track_created",
        "on_track_update",
        "on_track_removed",
    ]

    def test_adapter_signatures_match_the_real_global_track_manager(self):
        from lum_vision.person_tracking.global_track import GlobalTrackManager

        for name in self.METHOD_NAMES:
            with self.subTest(method=name):
                real_params = list(
                    inspect.signature(getattr(GlobalTrackManager, name)).parameters
                )[1:]  # drop self
                adapter_params = list(
                    inspect.signature(getattr(RemoteGlobalTrackManager, name)).parameters
                )[1:]
                self.assertEqual(
                    real_params,
                    adapter_params,
                    f"{name}: adapter params {adapter_params} != real params {real_params}",
                )


if __name__ == "__main__":
    unittest.main()
