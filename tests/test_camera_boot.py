"""Unit tests for the per-camera claim/reconcile loop (LSO-218).

The behaviour that matters: the fleet divides cameras by fair share (not by
hash luck), a worker only grows past its fair share to cover a camera nobody
has owned for the failover grace, capacity is a hard ceiling, and leasing
answers are acted on (a lost lease stops consumption; a camera that leaves
the eligible set is released). A FakeRedis with an explicit clock and the
real CameraLeaseManager are used, so the lease semantics are exercised, not
mocked.

Run: PYTHONPATH=src python -m pytest tests/test_camera_boot.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from loguru import logger  # noqa: E402

from camera_boot import CameraWorker  # noqa: E402
from pipeline.camera_lease import (  # noqa: E402
    CameraLeaseManager,
    _RELEASE_SCRIPT,
    _RENEW_SCRIPT,
)


class FakeRedis:
    """Enough of the redis client for CameraLeaseManager: SET NX EX, GET,
    EXISTS, DELETE and evalsha for the two compare-and-act scripts."""

    def __init__(self, fail=False):
        self.store = {}  # key -> (value, expires_at or None)
        self.now = 1000.0
        self.fail = fail
        self._scripts = {}
        self._next_sha = 0

    def _check_fail(self):
        if self.fail:
            raise ConnectionError("redis down")

    def _get_live(self, key):
        entry = self.store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and expires_at <= self.now:
            del self.store[key]
            return None
        return value

    def set(self, key, value, nx=False, ex=None):
        self._check_fail()
        if nx and self._get_live(key) is not None:
            return None
        self.store[key] = (value, self.now + ex if ex else None)
        return True

    def get(self, key):
        self._check_fail()
        return self._get_live(key)

    def delete(self, key):
        self._check_fail()
        self.store.pop(key, None)

    def exists(self, key):
        self._check_fail()
        return 1 if self._get_live(key) is not None else 0

    def script_load(self, script):
        self._check_fail()
        self._next_sha += 1
        sha = f"sha{self._next_sha}"
        self._scripts[sha] = script
        return sha

    def evalsha(self, sha, _numkeys, *args):
        self._check_fail()
        script = self._scripts.get(sha)
        if script is None:
            raise Exception("NOSCRIPT No matching script")
        key = args[0]
        if script == _RENEW_SCRIPT:
            worker_id, ttl = args[1], args[2]
            if self._get_live(key) == worker_id:
                self.store[key] = (worker_id, self.now + float(ttl))
                return 1
            return 0
        if script == _RELEASE_SCRIPT:
            worker_id = args[1]
            if self._get_live(key) == worker_id:
                self.store.pop(key, None)
                return 1
            return 0
        raise Exception(f"unrecognised script: {script!r}")


class FakeRegistry:
    """Records add/remove of consumers, and can fail adds on demand — either
    synchronously (`fail_on`) or via the deferred `on_error` callback
    (`fail_async_on`) that models `call_soon`'s async add."""

    def __init__(self):
        self.added = []
        self.removed = []
        self.fail_on = set()
        self.fail_async_on = set()
        self._pending_errors = []

    def add(self, queue, on_error=None):
        if queue in self.fail_on:
            raise RuntimeError("add failed")
        self.added.append(queue)
        if queue in self.fail_async_on and on_error is not None:
            self._pending_errors.append(on_error)

    def remove(self, queue, on_error=None):
        self.removed.append(queue)

    def fire_async_errors(self):
        for on_error in self._pending_errors:
            on_error(RuntimeError("async boom"))
        self._pending_errors.clear()


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _worker(
    worker_id,
    redis,
    cameras,
    *,
    replicas=3,
    capacity=7,
    grace=30.0,
    registry=None,
    clock=None,
    warn_interval=30.0,
):
    lease = CameraLeaseManager(
        ttl_seconds=60,
        worker_id=worker_id,
        redis_client=redis,
        key_prefix="track:cam:lease:",
    )
    return CameraWorker(
        registry=registry or FakeRegistry(),
        lease=lease,
        load_eligible=lambda: [{"camera_id": c} for c in cameras],
        replicas=replicas,
        capacity=capacity,
        failover_grace_s=grace,
        claim_interval_s=5,
        warn_interval_s=warn_interval,
        ttl_seconds=60,
        clock=clock or Clock(),
    )


class FairShareTests(unittest.TestCase):
    def test_first_worker_claims_only_its_fair_share(self):
        # 14 cameras / 3 replicas -> ceil = 5, NOT all 14.
        w = _worker("a", FakeRedis(), list(range(1, 15)))
        w.reconcile()
        self.assertEqual(len(w.held), 5)

    def test_three_workers_cover_every_camera_evenly(self):
        redis = FakeRedis()
        cams = list(range(1, 15))
        a, b, c = (
            _worker("a", redis, cams),
            _worker("b", redis, cams),
            _worker("c", redis, cams),
        )
        a.reconcile()
        b.reconcile()
        c.reconcile()
        self.assertEqual(len(a.held), 5)
        self.assertEqual(len(b.held), 5)
        self.assertEqual(len(c.held), 4)
        self.assertEqual(a.held | b.held | c.held, set(cams))

    def test_capacity_caps_claims_even_below_fair_share(self):
        # fair_share = ceil(10/1) = 10, but capacity 3 is a hard ceiling.
        w = _worker("a", FakeRedis(), list(range(1, 11)), replicas=1, capacity=3)
        w.reconcile()
        self.assertEqual(len(w.held), 3)


class FailoverTests(unittest.TestCase):
    def _boot(self, redis, cams, clock):
        workers = {
            name: _worker(
                name, redis, cams, clock=clock, replicas=3, capacity=7, grace=30
            )
            for name in ("a", "b", "c")
        }
        for w in workers.values():
            w.reconcile()
        return workers

    def test_no_growth_before_the_failover_grace(self):
        redis, clock = FakeRedis(), Clock()
        workers = self._boot(redis, list(range(1, 15)), clock)
        # b dies: its leases simply stop being renewed (drop them).
        for key, (val, _exp) in list(redis.store.items()):
            if val == "b":
                del redis.store[key]
        clock.advance(1)
        workers["a"].reconcile()
        self.assertEqual(len(workers["a"].held), 5)  # still its fair share

    def test_grows_to_capacity_for_orphans_after_the_grace(self):
        redis, clock = FakeRedis(), Clock()
        workers = self._boot(redis, list(range(1, 15)), clock)
        a = workers["a"]
        for key, (val, _exp) in list(redis.store.items()):
            if val == "b":
                del redis.store[key]
        clock.advance(1)
        a.reconcile()  # records when the orphans were first seen unowned
        clock.advance(31)
        a.reconcile()
        self.assertEqual(len(a.held), 7)  # fair share 5 -> capacity 7

    def test_fleet_still_covers_everything_after_one_worker_dies(self):
        redis, clock = FakeRedis(), Clock()
        workers = self._boot(redis, list(range(1, 15)), clock)
        for key, (val, _exp) in list(redis.store.items()):
            if val == "b":
                del redis.store[key]
        clock.advance(1)
        workers["a"].reconcile()
        workers["c"].reconcile()
        clock.advance(31)
        workers["a"].reconcile()
        workers["c"].reconcile()
        covered = workers["a"].held | workers["c"].held
        self.assertEqual(covered, set(range(1, 15)))


class EligibilityTests(unittest.TestCase):
    def test_release_when_a_camera_is_no_longer_eligible(self):
        redis = FakeRedis()
        cams = [1, 2, 3]
        reg = FakeRegistry()
        w = _worker("a", redis, cams, replicas=1, capacity=3, registry=reg)
        w.reconcile()
        self.assertEqual(w.held, {1, 2, 3})

        cams.remove(2)
        w.reconcile()
        self.assertEqual(w.held, {1, 3})
        self.assertIn("cam.2", reg.removed)
        self.assertIsNone(redis.get("track:cam:lease:2"))

    def test_add_failure_hands_the_lease_back(self):
        redis = FakeRedis()
        reg = FakeRegistry()
        reg.fail_on = {"cam.1"}
        w = _worker("a", redis, [1], replicas=1, capacity=1, registry=reg)
        w.reconcile()
        self.assertEqual(w.held, set())
        self.assertIsNone(redis.get("track:cam:lease:1"))

    def test_async_add_failure_hands_the_lease_back(self):
        # The consumer add is deferred onto the worker loop; if it throws
        # there, the camera must not stay "held" with nothing consuming it.
        redis = FakeRedis()
        reg = FakeRegistry()
        reg.fail_async_on = {"cam.1"}
        w = _worker("a", redis, [1], replicas=1, capacity=1, registry=reg)
        w.reconcile()
        self.assertEqual(w.held, {1})  # scheduled successfully...

        reg.fire_async_errors()  # ...but the deferred add failed
        self.assertEqual(w.held, set())
        self.assertIsNone(redis.get("track:cam:lease:1"))


class RenewTests(unittest.TestCase):
    def test_renew_once_keeps_a_held_camera(self):
        redis = FakeRedis()
        w = _worker("a", redis, [1], replicas=1, capacity=1)
        w.reconcile()
        w.renew_once()
        self.assertEqual(w.held, {1})

    def test_renew_once_releases_a_camera_whose_lease_was_taken(self):
        redis = FakeRedis()
        reg = FakeRegistry()
        w = _worker("a", redis, [1], replicas=1, capacity=1, registry=reg)
        w.reconcile()
        self.assertEqual(w.held, {1})

        # Another worker now owns camera 1 (this worker's TTL lapsed).
        redis.store["track:cam:lease:1"] = ("b", redis.now + 100)
        w.renew_once()
        self.assertEqual(w.held, set())
        self.assertIn("cam.1", reg.removed)

    def test_transient_failures_count_per_pass_not_per_camera(self):
        # One Redis blip fails every camera's renew in the same pass; that must
        # count as ONE bad interval, not one per camera, or a worker at fair
        # share would exit on the blip's first tick.
        redis = FakeRedis()
        w = _worker("a", redis, [1, 2, 3], replicas=1, capacity=3)
        w.reconcile()
        self.assertEqual(w.held, {1, 2, 3})

        redis.fail = True
        w.renew_once()  # three cameras fail transiently, one pass
        self.assertEqual(w._transient_renew_fails, 1)
        self.assertEqual(w.held, {1, 2, 3})  # nothing torn down

    def test_a_clean_pass_resets_the_transient_counter(self):
        redis = FakeRedis()
        w = _worker("a", redis, [1], replicas=1, capacity=1)
        w.reconcile()
        w._transient_renew_fails = 1
        w.renew_once()
        self.assertEqual(w._transient_renew_fails, 0)


class ReleaseAllTests(unittest.TestCase):
    def test_release_all_hands_every_lease_back(self):
        redis = FakeRedis()
        reg = FakeRegistry()
        w = _worker("a", redis, [1, 2, 3], replicas=1, capacity=3, registry=reg)
        w.reconcile()
        self.assertEqual(w.held, {1, 2, 3})

        w.release_all()
        self.assertEqual(w.held, set())
        for camera_id in (1, 2, 3):
            self.assertIsNone(redis.get(f"track:cam:lease:{camera_id}"))
            self.assertIn(f"cam.{camera_id}", reg.removed)


class UnclaimedWarningTests(unittest.TestCase):
    def test_warns_when_cameras_stay_unowned_after_the_grace(self):
        redis, clock = FakeRedis(), Clock()
        w = _worker("a", redis, list(range(1, 15)), replicas=3, capacity=7, clock=clock)
        messages = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            w.reconcile()  # claims 5; records the rest as unowned now
            clock.advance(31)
            w.reconcile()  # grows to 7; 7 remain unowned -> warn
        finally:
            logger.remove(sink_id)
        self.assertTrue(
            any("no owner" in m for m in messages),
            f"expected an uncovered-cameras warning, got: {messages}",
        )

    def test_repeated_adopt_failure_still_trips_the_uncovered_warning(self):
        # _adopt failing must NOT reset the unowned timer: a camera whose
        # consumer can never be added is exactly a coverage gap the warning
        # exists to surface, so it has to be reachable.
        redis, clock = FakeRedis(), Clock()
        reg = FakeRegistry()
        reg.fail_on = {"cam.1"}
        w = _worker(
            "a", redis, [1], replicas=1, capacity=1, registry=reg, clock=clock
        )
        messages = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            for _ in range(5):  # 50s > the 30s failover grace
                clock.advance(10)
                w.reconcile()
        finally:
            logger.remove(sink_id)
        self.assertEqual(w.held, set())
        self.assertTrue(
            any("no owner" in m for m in messages),
            f"expected an uncovered-cameras warning, got: {messages}",
        )

    def test_warning_lists_only_orphans_not_fair_share_skips(self):
        # A camera unowned merely because this worker is at capacity is not
        # orphaned yet — it must not appear in the incident-facing list until
        # it has actually been unowned past the grace.
        redis, clock = FakeRedis(), Clock()
        w = _worker("a", redis, [1, 2, 3, 4], replicas=1, capacity=3, clock=clock)
        messages = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            w.reconcile()  # holds 1-3; 4 is unowned but fresh
            self.assertFalse([m for m in messages if "Orphaned:" in m])
            clock.advance(31)
            w.reconcile()  # 4 is now past the grace
        finally:
            logger.remove(sink_id)
        orphans = [m for m in messages if "Orphaned:" in m]
        self.assertTrue(orphans, f"expected an orphan warning, got: {messages}")
        self.assertIn("Orphaned: [4]", orphans[-1])


if __name__ == "__main__":
    unittest.main()
