"""Unit tests for decode_main.DecodeWorker's reconcile loop.

Everything that touches real RTSP or shared memory is faked
(FakeStreamHandler, FakeProducer, FakeRawFrameSlot) — this covers the
claim/renew/release orchestration and the config-diff logic
(LSO-155's stream_url-changes-need-a-restart / everything-else-doesn't),
not StreamHandler or CeleryCameraProducer themselves, which have their own
tests.

Run: PYTHONPATH=src python -m pytest tests/test_decode_worker.py
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))


class FakeLease:
    """In-memory stand-in for CameraLeaseManager — no Redis at all, since
    these tests are about DecodeWorker's orchestration, not the lease
    primitive itself (that's test_camera_lease.py's job)."""

    def __init__(self, worker_id="worker-a"):
        self._worker_id = worker_id
        self.owned = set()
        # Cameras this worker has lost to a (simulated) other owner and can
        # never re-claim in this test — a real stolen lease means the Redis
        # key still exists, held by someone else, not that it's free again.
        # Without this, "renew fails" and "claim succeeds" would
        # contradict each other, which real CameraLeaseManager cannot do.
        self._blocked = set()
        self.claim_calls = []
        self.renew_calls = []
        self.release_calls = []
        self.renew_should_fail_for = set()

    @property
    def worker_id(self):
        return self._worker_id

    def claim(self, camera_id):
        self.claim_calls.append(camera_id)
        if camera_id in self.owned or camera_id in self._blocked:
            return False
        self.owned.add(camera_id)
        return True

    def renew(self, camera_id):
        self.renew_calls.append(camera_id)
        if camera_id in self.renew_should_fail_for:
            self.owned.discard(camera_id)
            self._blocked.add(camera_id)
            return False
        return camera_id in self.owned

    def release(self, camera_id):
        self.release_calls.append(camera_id)
        self.owned.discard(camera_id)

    def is_claimed(self, camera_id):
        return camera_id in self.owned or camera_id in self._blocked


class FakeStreamHandler:
    instances = []

    def __init__(self, src, logger):
        self.src = src
        self.started = False
        self.stopped = False
        FakeStreamHandler.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read(self):
        return False, None

    def get_health(self):
        return {"state": "streaming"}

    def get_read_avg_ms(self):
        return 1.0

    def get_decode_avg_ms(self):
        return 2.0


class FakeProducer:
    instances = []

    def __init__(self, camera_id, camera_config, stream_handler, **kwargs):
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.stream_handler = stream_handler
        self.started = False
        self.stopped = False
        FakeProducer.instances.append(self)

    def start(self):
        self.started = True

    def stop(self, timeout=5.0):
        self.stopped = True


class FakeRawFrameSlot:
    instances = []

    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.closed = False
        FakeRawFrameSlot.instances.append(self)

    def close(self):
        self.closed = True


def _worker(**kwargs):
    from decode_main import DecodeWorker

    worker = DecodeWorker(
        client_slug="test-client", applications=["attendance"], detection_interval=2
    )
    worker.lease = kwargs.get("lease", FakeLease())
    worker.health = mock.Mock()
    # The real worker is running whenever these loops execute; _renew_held
    # deliberately bails when it isn't, so leave this True or renewal tests
    # silently pass by doing nothing.
    worker._running = True
    return worker


def _patches(cameras):
    return [
        mock.patch("decode_main.load_cameras_from_db", return_value=cameras),
        mock.patch("decode_main.StreamHandler", FakeStreamHandler),
        mock.patch("decode_main.CeleryCameraProducer", FakeProducer),
        mock.patch("decode_main.RawFrameSlot", FakeRawFrameSlot),
    ]


def _apply(patches):
    for p in patches:
        p.start()
    return patches


def _stop(patches):
    for p in patches:
        p.stop()


class ReconcileClaimTests(unittest.TestCase):
    def setUp(self):
        FakeStreamHandler.instances = []
        FakeProducer.instances = []
        FakeRawFrameSlot.instances = []

    def test_claims_and_starts_every_eligible_camera_up_to_capacity(self):
        cameras = [
            {"camera_id": 1, "stream_url": "rtsp://a/1"},
            {"camera_id": 2, "stream_url": "rtsp://a/2"},
        ]
        worker = _worker()
        patches = _apply(_patches(cameras))
        try:
            with mock.patch("decode_main._CAPACITY", 6):
                worker._reconcile()
        finally:
            _stop(patches)

        self.assertEqual(set(worker._owned.keys()), {1, 2})
        self.assertTrue(all(p.started for p in FakeProducer.instances))
        self.assertTrue(all(s.started for s in FakeStreamHandler.instances))

    def test_does_not_claim_past_capacity(self):
        cameras = [{"camera_id": i, "stream_url": f"rtsp://a/{i}"} for i in range(5)]
        worker = _worker()
        patches = _apply(_patches(cameras))
        try:
            with mock.patch("decode_main._CAPACITY", 2):
                worker._reconcile()
        finally:
            _stop(patches)

        self.assertEqual(len(worker._owned), 2)

    def test_does_not_reclaim_a_camera_it_already_holds(self):
        cameras = [{"camera_id": 1, "stream_url": "rtsp://a/1"}]
        lease = FakeLease()
        worker = _worker(lease=lease)
        patches = _apply(_patches(cameras))
        try:
            with mock.patch("decode_main._CAPACITY", 6):
                worker._reconcile()
                worker._reconcile()
        finally:
            _stop(patches)

        # Claimed exactly once — the second pass must not re-claim it.
        # (Renewal is no longer _reconcile's job; see RenewalTests.)
        self.assertEqual(lease.claim_calls.count(1), 1)


class ReconcileLossTests(unittest.TestCase):
    def setUp(self):
        FakeStreamHandler.instances = []
        FakeProducer.instances = []
        FakeRawFrameSlot.instances = []

    def test_a_camera_no_longer_eligible_is_released_and_stopped(self):
        cameras = [{"camera_id": 1, "stream_url": "rtsp://a/1"}]
        lease = FakeLease()
        worker = _worker(lease=lease)
        patches = _apply(_patches(cameras))
        try:
            with mock.patch("decode_main._CAPACITY", 6):
                worker._reconcile()
        finally:
            _stop(patches)

        producer = FakeProducer.instances[0]
        stream = FakeStreamHandler.instances[0]
        raw_slot = FakeRawFrameSlot.instances[0]

        # Camera 1 drops out of the DB-eligible list on the next reconcile.
        patches = _apply(_patches([]))
        try:
            worker._reconcile()
        finally:
            _stop(patches)

        self.assertNotIn(1, worker._owned)
        self.assertIn(1, lease.release_calls)
        self.assertTrue(producer.stopped)
        self.assertTrue(stream.stopped)
        self.assertTrue(raw_slot.closed)

    def test_a_failed_renew_stops_the_camera_without_releasing_the_lease(self):
        """A failed renew means someone else already owns this camera's
        lease — releasing here would delete THEIR lease, so this must stop
        locally only, matching CameraLeaseManager.renew's contract."""
        cameras = [{"camera_id": 1, "stream_url": "rtsp://a/1"}]
        lease = FakeLease()
        worker = _worker(lease=lease)
        patches = _apply(_patches(cameras))
        try:
            with mock.patch("decode_main._CAPACITY", 6):
                worker._reconcile()
        finally:
            _stop(patches)

        producer = FakeProducer.instances[0]
        lease.renew_should_fail_for.add(1)

        worker._renew_held([1])

        self.assertNotIn(1, worker._owned)
        self.assertTrue(producer.stopped)
        self.assertNotIn(1, lease.release_calls)


class RenewalTests(unittest.TestCase):
    """Renewal runs on its own timer (_periodic_loop), deliberately NOT
    inside _reconcile: claiming a camera opens an RTSP connection whose
    duration is unbounded (~1.7s warm, ~20s during a cold start), and
    renewing behind that starved the leases of cameras already held —
    observed live churning 4 of 6 cameras even at a 60s TTL."""

    def setUp(self):
        FakeStreamHandler.instances = []
        FakeProducer.instances = []
        FakeRawFrameSlot.instances = []

    def test_renew_covers_cameras_still_starting_up(self):
        """A camera is claimed before _start_camera runs, and isn't in
        _owned until it finishes — but it holds a lease the whole time, so
        renewal must cover that window too."""
        lease = FakeLease()
        worker = _worker(lease=lease)
        lease.claim(1)
        with worker._lock:
            worker._starting.add(1)

        worker._periodic_tick_renew()

        self.assertIn(1, lease.renew_calls)

    def test_reconcile_no_longer_renews(self):
        """Renewal must not be coupled to the claim loop's duration."""
        cameras = [{"camera_id": 1, "stream_url": "rtsp://a/1"}]
        lease = FakeLease()
        worker = _worker(lease=lease)
        patches = _apply(_patches(cameras))
        try:
            with mock.patch("decode_main._CAPACITY", 6):
                worker._reconcile()
                lease.renew_calls.clear()
                worker._reconcile()
        finally:
            _stop(patches)

        self.assertEqual(lease.renew_calls, [])


class SyncConfigTests(unittest.TestCase):
    """LSO-155's actual regression, now covered here instead of
    test_engine_reload.py: a stream_url change must restart exactly that
    camera; any other field change must reach the live config object
    without restarting anything."""

    def setUp(self):
        FakeStreamHandler.instances = []
        FakeProducer.instances = []
        FakeRawFrameSlot.instances = []

    def test_stream_url_change_restarts_only_that_cameras_stream(self):
        cam1 = {"camera_id": 1, "stream_url": "rtsp://old-ip/1"}
        cam2 = {"camera_id": 2, "stream_url": "rtsp://stable/2"}
        worker = _worker()
        patches = _apply(_patches([cam1, cam2]))
        try:
            with mock.patch("decode_main._CAPACITY", 6):
                worker._reconcile()
        finally:
            _stop(patches)

        old_producer1 = worker._owned[1].producer
        old_stream1 = worker._owned[1].stream
        producer2 = worker._owned[2].producer
        stream2 = worker._owned[2].stream

        new_cam1 = {"camera_id": 1, "stream_url": "rtsp://new-ip/1"}
        new_cam2 = {"camera_id": 2, "stream_url": "rtsp://stable/2"}
        patches = _apply(_patches([new_cam1, new_cam2]))
        try:
            worker._reconcile()
        finally:
            _stop(patches)

        # Camera 1: old producer/stream stopped, a new pair started against
        # the new URL.
        self.assertTrue(old_producer1.stopped)
        self.assertTrue(old_stream1.stopped)
        new_owned1 = worker._owned[1]
        self.assertIsNot(new_owned1.producer, old_producer1)
        self.assertTrue(new_owned1.producer.started)
        self.assertEqual(new_owned1.config["stream_url"], "rtsp://new-ip/1")

        # Camera 2: untouched.
        self.assertIs(worker._owned[2].producer, producer2)
        self.assertIs(worker._owned[2].stream, stream2)
        self.assertFalse(producer2.stopped)
        self.assertFalse(stream2.stopped)

    def test_non_stream_field_change_updates_live_config_without_restart(self):
        cam_config = {
            "camera_id": 1,
            "stream_url": "rtsp://stable/1",
            "application": ["attendance"],
        }
        worker = _worker()
        patches = _apply(_patches([cam_config]))
        try:
            with mock.patch("decode_main._CAPACITY", 6):
                worker._reconcile()
        finally:
            _stop(patches)

        producer = worker._owned[1].producer
        live_config = producer.camera_config

        new_config = {
            "camera_id": 1,
            "stream_url": "rtsp://stable/1",
            "application": ["attendance", "action_recognition"],
        }
        patches = _apply(_patches([new_config]))
        try:
            worker._reconcile()
        finally:
            _stop(patches)

        self.assertFalse(producer.stopped)
        self.assertIs(producer.camera_config, live_config)
        self.assertEqual(
            live_config["application"], ["attendance", "action_recognition"]
        )


if __name__ == "__main__":
    unittest.main()
