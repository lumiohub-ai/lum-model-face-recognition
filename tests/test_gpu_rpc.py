"""Unit tests for gpu_rpc + global_track_adapter (LSO-67, Stage 1).

Covers the Unix-socket RPC bridge to GlobalTrackManager: correct dispatch
across the blocking/one-way split, the local-ID fallback on every failure
mode (server down, timeout, a bug inside the real manager), and that
RemoteGlobalTrackManager's ten method signatures still match what
CameraEngine/PersonTracker actually call on the real GlobalTrackManager -
the kind of drift that breaks silently if lum_vision's signatures change
without a corresponding update here.

Run: PYTHONPATH=src python tests/test_gpu_rpc.py
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
from workers.gpu_rpc import (  # noqa: E402
    GpuRpcClient,
    GpuRpcServer,
    MethodCallRequest,
)


def _free_socket_path(tag: str) -> str:
    """A distinct, tag-suffixed path per test - tests run against real Unix
    sockets, and reusing one path across tests racing in the same process
    would let one test's leftover socket answer another's client."""
    return f"/tmp/test_gpu_rpc_{tag}_{os.getpid()}.sock"


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
        gpu_rpc.py's module docstring) - proven here by making the fake
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

    def _down_adapter(self, tag):
        """An adapter whose socket points at nothing - the degraded path."""
        return RemoteGlobalTrackManager(
            GpuRpcClient(socket_path=_free_socket_path(tag), timeout_s=0.2),
            enabled=True,
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

    def test_get_global_id_on_rpc_failure_returns_none_not_a_fabricated_id(self):
        """The result flows into the recognized_persons payload toward the
        backend; None ('unknown') is what the caller already handles, a
        fabricated negative ID is not."""
        self.assertIsNone(self._down_adapter("get_down").get_global_id(1, 1))

    def test_assign_on_rpc_failure_still_returns_a_negative_local_id(self):
        """assign_global_id is the one method where the negative fallback IS
        the contract - the camera needs *an* ID to keep tracking with."""
        gid = self._down_adapter("assign_down").assign_global_id(
            camera_id=1, local_track_id=1, person_crop=None
        )
        self.assertLess(gid, 0)


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
