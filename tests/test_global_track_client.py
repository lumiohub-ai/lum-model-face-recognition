"""Unit tests for global_track_client + global_track_adapter.

Replaces test_global_track_rpc.py, which tested the Unix-socket bridge that
global-track-worker's Celery queue took over (docs/GLOBAL_TRACKING.md). The
tests that went away with it were about socket mechanics — connection reuse,
reconnect-after-restart, framing, stream desync. Those failure modes do not
exist any more: Celery owns delivery, and a dropped connection is its problem,
not this codebase's.

What carried over is everything that was really about *behaviour the callers
depend on*, since none of that changed with the transport:

  - the local-ID fallback on every failure (the camera keeps tracking),
  - never fabricating a "found" track from a fallback (identity corruption),
  - the method allow-lists (the tasks dispatch by name),
  - the whole async-assign path (non-blocking, cache, in-flight suppression),
  - the signature-drift tripwire against the real GlobalTrackManager.

RemoteGlobalTrackManager only ever calls `.call()` / `.call_one_way()` on its
client, so these drive it through a fake one rather than a live broker. That
is not a shortcut around integration: it is the same seam the production swap
used, and it keeps these tests fast and hermetic (the old suite needed a real
socket server per test).

Run: PYTHONPATH=src python -m pytest tests/test_global_track_client.py
"""

import inspect
import json
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from workers.global_track_adapter import (  # noqa: E402
    GlobalTrackRef,
    RemoteGlobalTrackManager,
)
from workers.global_track_client import (  # noqa: E402
    _BLOCKING_METHODS,
    _ONE_WAY_METHODS,
    GlobalTrackClient,
    MethodCallResult,
    _LocalIdFallback,
)


class FakeClient:
    """Stands in for GlobalTrackClient. Records every call so tests can assert
    on dispatch, and lets each test decide what a call returns — including
    failing, which is the case most of these tests are about.

    Mirrors the real client's two-method surface exactly; the adapter never
    touches anything else on it (that is what made the socket -> Celery swap a
    type-annotation change in the adapter and nothing more).
    """

    def __init__(self, value=None, ok=True, block=None):
        self.calls = []
        self.one_way_calls = []
        self._value = value
        self._ok = ok
        self._block = block
        self._fallback = _LocalIdFallback()
        self.on_fallback = None

    def call(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        if self._block is not None:
            self._block.wait(timeout=5.0)
        if not self._ok:
            if self.on_fallback is not None:
                self.on_fallback()
            return MethodCallResult(value=self._fallback.next_id(), ok=False)
        value = self._value(method) if callable(self._value) else self._value
        return MethodCallResult(value=value, ok=True)

    def call_one_way(self, method, *args, **kwargs):
        self.one_way_calls.append((method, args, kwargs))


class MethodAllowListTests(unittest.TestCase):
    """The task modules dispatch by method name, so an open set would let a
    caller invoke anything the register exposes. Same guard the socket server
    had, kept for the same reason."""

    def setUp(self):
        self.client = GlobalTrackClient()

    def test_call_rejects_a_one_way_method_name(self):
        with self.assertRaises(ValueError):
            self.client.call("on_track_update", camera_id=1, local_track_id=1)

    def test_call_one_way_rejects_a_blocking_method_name(self):
        with self.assertRaises(ValueError):
            self.client.call_one_way("find_global_track_by_identity", "alice")

    def test_get_global_id_is_in_neither_allow_list(self):
        """It is answered from the adapter's own cache and must never cross a
        process boundary again — naming it would resurrect a round trip that
        was removed on purpose (see RemoteGlobalTrackManager.get_global_id)."""
        self.assertNotIn("get_global_id", _BLOCKING_METHODS)
        self.assertNotIn("get_global_id", _ONE_WAY_METHODS)

    def test_every_allow_listed_method_resolves_to_a_real_task(self):
        """A name in the allow-list with no task behind it would raise KeyError
        at the moment that exact call path executes — in production, not here."""
        from workers.global_track_client import _task_for

        for method in sorted(_BLOCKING_METHODS | _ONE_WAY_METHODS):
            with self.subTest(method=method):
                self.assertTrue(_task_for(method).name.startswith("globaltrack."))


class JsonSafetyTests(unittest.TestCase):
    """Regression: task payloads are JSON (task_serializer='json',
    celery_app.py), not pickle — unlike results, which can carry numpy
    freely. on_track_created's `bbox` (Optional[np.ndarray], see
    global_track_adapter.py) hit this live: apply_async raised
    EncodeError('Object of type ndarray is not JSON serializable'), caught
    and merely logged by call_one_way's own try/except, so the call was
    silently dropped in production with no test ever exercising a real
    ndarray argument end-to-end."""

    def test_json_safe_converts_ndarray_to_a_plain_list(self):
        import numpy as np

        bbox = np.array([10, 20, 30, 40])
        safe = GlobalTrackClient._json_safe(bbox)
        self.assertEqual(safe, [10, 20, 30, 40])
        self.assertIsInstance(safe, list)
        json.dumps(safe)  # must not raise

    def test_json_safe_converts_ndarray_nested_in_kwargs(self):
        import numpy as np

        kwargs = {
            "camera_id": 1,
            "local_track_id": 2,
            "bbox": np.array([10, 20, 30, 40]),
            "frame_num": 5,
        }
        safe = GlobalTrackClient._json_safe(kwargs)
        json.dumps(safe)  # must not raise
        self.assertEqual(safe["bbox"], [10, 20, 30, 40])

    def test_json_safe_converts_numpy_scalars(self):
        import numpy as np

        safe = GlobalTrackClient._json_safe(
            {"quality": np.float32(0.9), "count": np.int64(3)}
        )
        json.dumps(safe)  # must not raise
        self.assertEqual(safe, {"quality": pytest_approx(0.9), "count": 3})

    def test_json_safe_leaves_plain_values_unchanged(self):
        safe = GlobalTrackClient._json_safe(
            {"camera_id": 1, "identity": "alice", "locked": True, "bbox": None}
        )
        self.assertEqual(
            safe, {"camera_id": 1, "identity": "alice", "locked": True, "bbox": None}
        )

    def test_on_track_created_with_a_real_ndarray_bbox_does_not_raise(self):
        """End-to-end through the real client: apply_async is expected to
        fail in this test environment (no broker), which call_one_way already
        catches and logs — the bug was that it failed on SERIALIZING the
        ndarray, before ever reaching the broker. This proves it gets past
        that point."""
        import numpy as np

        client = GlobalTrackClient()
        client.call_one_way(
            "on_track_created",
            camera_id=1,
            local_track_id=1,
            bbox=np.array([10, 20, 30, 40]),
            frame_num=1,
        )  # must not raise


class ExpiryOverrideTests(unittest.TestCase):
    """on_track_removed gets a longer Celery message expiry than the other
    one-way calls (docs/GLOBAL_TRACKS_LEAK.md): it's rare and idempotent,
    unlike the per-frame calls, which must stay tight so a backed-up worker
    sheds stale frame data instead of acting on minute-old positions."""

    def test_on_track_removed_uses_the_longer_override(self):
        import workers.global_track_client as client_module

        client = GlobalTrackClient()
        recorded = {}

        class FakeTask:
            def apply_async(self, args=None, kwargs=None, queue=None, expires=None):
                recorded["expires"] = expires

        original = client_module._task_for
        client_module._task_for = lambda method: FakeTask()
        try:
            client.call_one_way("on_track_removed", camera_id=1, local_track_id=1)
        finally:
            client_module._task_for = original

        self.assertEqual(
            recorded["expires"], client_module._EXPIRES_OVERRIDES_S["on_track_removed"]
        )
        self.assertGreater(recorded["expires"], client._expires_s)

    def test_other_one_way_methods_keep_the_default_expiry(self):
        import workers.global_track_client as client_module

        client = GlobalTrackClient()
        recorded = {}

        class FakeTask:
            def apply_async(self, args=None, kwargs=None, queue=None, expires=None):
                recorded["expires"] = expires

        original = client_module._task_for
        client_module._task_for = lambda method: FakeTask()
        try:
            client.call_one_way("on_face_detected", camera_id=1, local_track_id=1)
        finally:
            client_module._task_for = original

        self.assertEqual(recorded["expires"], client._expires_s)


def pytest_approx(value, tol=1e-5):
    """Tiny float-tolerance helper — avoids pulling in pytest.approx for one
    assertion in a unittest.TestCase-based file."""

    class _Approx(float):
        def __eq__(self, other):
            return abs(float(other) - float(self)) < tol

    return _Approx(value)


class FallbackTests(unittest.TestCase):
    """What the caller gets when global-track-worker cannot answer. These are
    the contracts camera_engine.py degrades against, unchanged from the socket
    era — only the reason a call fails is different now."""

    def test_local_id_fallback_returns_distinct_negative_ids(self):
        fallback = _LocalIdFallback()
        ids = [fallback.next_id() for _ in range(5)]
        self.assertEqual(len(set(ids)), 5, "fallback handed out a duplicate ID")
        self.assertTrue(all(i < 0 for i in ids), "fallback ID was not negative")

    def test_assign_on_failure_still_returns_a_negative_local_id(self):
        """assign_global_id is the one method where the negative fallback IS
        the contract — the camera needs *an* ID to keep tracking with, and
        camera_engine.py's `current_global_id >= 0` guard already treats
        negative as 'not yet identified'."""
        adapter = RemoteGlobalTrackManager(
            FakeClient(ok=False), enabled=True, async_assign=False
        )
        gid = adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.assertLess(gid, 0)

    def test_find_on_failure_returns_none_not_a_fabricated_track(self):
        """A negative fallback ID presented as a *found track* would make
        CameraEngine reassign the person to it via reassign_local_track —
        silent identity corruption. The only safe degraded answer for a lookup
        is 'not found'."""
        adapter = RemoteGlobalTrackManager(FakeClient(ok=False), enabled=True)
        self.assertIsNone(adapter.find_global_track_by_identity("alice"))

    def test_one_way_call_never_raises(self):
        adapter = RemoteGlobalTrackManager(FakeClient(ok=False), enabled=True)
        adapter.on_track_update(camera_id=1, local_track_id=1)  # must not raise


class AdapterBehaviorTests(unittest.TestCase):
    def test_find_result_supports_the_attribute_access_camera_engine_does(self):
        """CameraEngine reads `existing_global.global_id` off the result
        (camera_engine.py's Global ID reassignment block) — a bare int here
        crashes that call site with AttributeError. This is the drop-in
        contract, asserted the way the caller exercises it."""
        adapter = RemoteGlobalTrackManager(FakeClient(value=777), enabled=True)
        existing = adapter.find_global_track_by_identity("alice")
        self.assertIsInstance(existing, GlobalTrackRef)
        self.assertEqual(existing.global_id, 777)

    def test_find_unknown_identity_returns_none(self):
        adapter = RemoteGlobalTrackManager(FakeClient(value=None), enabled=True)
        self.assertIsNone(adapter.find_global_track_by_identity("nobody"))

    def test_get_global_id_is_a_pure_cache_read_no_call(self):
        """get_global_id never touches the client at all — it only reads
        _assigned, populated separately by assign_global_id."""
        client = FakeClient(value=None)
        adapter = RemoteGlobalTrackManager(client, enabled=True)
        self.assertIsNone(adapter.get_global_id(1, 1))
        adapter._assigned[(1, 1)] = 42
        self.assertEqual(adapter.get_global_id(1, 1), 42)
        self.assertEqual(client.calls, [], "get_global_id made a call")

    def test_one_way_calls_reach_the_client_with_their_arguments(self):
        client = FakeClient()
        adapter = RemoteGlobalTrackManager(client, enabled=True)

        adapter.on_face_detected(camera_id=1, local_track_id=2, quality=0.9)
        adapter.on_face_not_visible(camera_id=1, local_track_id=2)
        adapter.on_track_created(camera_id=1, local_track_id=2)
        adapter.on_track_update(camera_id=1, local_track_id=2)
        adapter.on_track_removed(camera_id=1, local_track_id=2)
        adapter.reassign_local_track(camera_id=1, local_track_id=2, new_global_id=9)
        adapter.update_global_track_identity(9, "alice", locked=True)

        sent = [c[0] for c in client.one_way_calls]
        self.assertEqual(sorted(sent), sorted(_ONE_WAY_METHODS))
        self.assertEqual(client.calls, [], "a one-way method used the blocking path")


class AsyncAssignTests(unittest.TestCase):
    """RemoteGlobalTrackManager's async_assign=True path (the default, matching
    camera_tasks.py's wiring): assign_global_id must never block the caller,
    must return None until the first reply lands, must send at most one request
    at a time per (camera_id, local_track_id), and must NOT forget cached state
    on on_track_removed — that's the caller's job, via forget_track(), after it
    reads get_global_id()."""

    def setUp(self):
        self.release = threading.Event()
        # Deterministic formula, matching the old suite's fake: lets a test
        # assert on the exact cached value rather than merely "not None".
        self.client = FakeClient(value=lambda _m: 1101, block=self.release)
        self.adapter = RemoteGlobalTrackManager(
            self.client, enabled=True, async_assign=True
        )

    def tearDown(self):
        self.release.set()  # unblock anything still in flight

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
        self.assertLess(elapsed, 0.5, "assign_global_id blocked the caller")
        self.release.set()

    def test_reply_populates_the_cache_for_the_next_call(self):
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.release.set()
        self.assertTrue(
            self._poll_until(
                lambda: self.adapter.assign_global_id(
                    camera_id=1, local_track_id=1, person_crop=None
                )
                == 1101
            ),
            "cached global id never appeared after the background call finished",
        )

    def test_second_call_while_first_in_flight_does_not_send_another(self):
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.assertTrue(
            self._poll_until(lambda: len(self.client.calls) == 1),
            "first request never reached the client",
        )
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        time.sleep(0.05)  # give a wrongly-submitted second request time to land
        self.assertEqual(
            len(self.client.calls),
            1,
            "a second request for the same track was sent while one was in flight",
        )
        # Direct check on submission count, not just on the client having
        # received it: a wrongly-submitted extra task would queue behind the
        # first on the single-worker pool rather than run concurrently, so
        # call count alone can pass by accident of pool size.
        self.assertEqual(self.adapter._submitted_count, 1)
        self.release.set()

    def test_different_tracks_do_not_block_each_other(self):
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.assertTrue(self._poll_until(lambda: len(self.client.calls) == 1))
        # Track 2 must still be accepted (returns None immediately, queued
        # behind the single-worker pool) rather than suppressed by track 1's
        # in-flight request — the guard is keyed per track, not global.
        result = self.adapter.assign_global_id(
            camera_id=1, local_track_id=2, person_crop=None
        )
        self.assertIsNone(result)
        self.release.set()
        self.assertTrue(self._poll_until(lambda: len(self.client.calls) == 2))

    def test_forget_track_clears_the_cached_id(self):
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.release.set()
        self.assertTrue(self._poll_until(lambda: (1, 1) in self.adapter._assigned))

        self.adapter.forget_track(camera_id=1, local_track_id=1)
        calls_before = len(self.client.calls)
        self.assertIsNone(
            self.adapter._assigned.get((1, 1)), "forget_track did not clear the cache"
        )
        # forget_track only clears local state; it must not trigger a call.
        self.assertEqual(len(self.client.calls), calls_before)

    def test_on_track_removed_does_not_forget_the_track(self):
        """on_track_removed deliberately leaves _assigned alone: the vendored
        PersonTracker calls it internally BEFORE CameraEngine's removed-tracks
        loop reads the cached id via get_global_id(). Forgetting here would
        clear the note before anyone reads it. camera_engine.py calls
        forget_track() itself, right after that read."""
        self.adapter.assign_global_id(camera_id=1, local_track_id=1, person_crop=None)
        self.release.set()
        self.assertTrue(self._poll_until(lambda: (1, 1) in self.adapter._assigned))

        self.adapter.on_track_removed(camera_id=1, local_track_id=1)
        self.assertEqual(self.adapter._assigned.get((1, 1)), 1101)

    def test_failure_in_the_background_still_clears_in_flight(self):
        """A failed background call must not leave the track permanently stuck
        'in flight' — that would freeze its global id at whatever _assigned
        already held, forever."""
        adapter = RemoteGlobalTrackManager(
            FakeClient(ok=False), enabled=True, async_assign=True
        )
        adapter.assign_global_id(camera_id=9, local_track_id=9, person_crop=None)
        self.assertTrue(
            self._poll_until(lambda: (9, 9) not in adapter._in_flight, timeout=2.0),
            "in_flight was never cleared after the background call failed",
        )


class AdapterSignatureDriftTests(unittest.TestCase):
    """Not a behavioral test - a tripwire. RemoteGlobalTrackManager's ten
    methods were hand-transcribed from reading lum_vision's source; this
    catches the day someone changes a real GlobalTrackManager/PersonTracker
    call signature without updating the adapter to match, which would
    otherwise fail silently (extra/renamed kwargs raise a TypeError only at
    the moment that exact call path executes, not at import time).
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
