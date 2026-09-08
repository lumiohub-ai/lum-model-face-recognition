"""Unit tests for CameraLeaseManager's claim/renew/release semantics.

Mirrors test_action_worker.py's FakeRedis pattern (an explicit clock, not
real time, so expiry tests stay instant and deterministic) but extends it
with script_load/evalsha, since renew/release use a Lua compare-and-act
script rather than a plain command.

Run: PYTHONPATH=src python tests/test_camera_lease.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from pipeline.camera_lease import (  # noqa: E402
    CameraLeaseManager,
    _RELEASE_SCRIPT,
    _RENEW_SCRIPT,
)


class FakeRedis:
    """Enough of the redis client for CameraLeaseManager: SET NX EX, GET,
    DELETE, EXISTS, and evalsha for the two compare-and-act scripts."""

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


def _manager(worker_id="worker-a", ttl=10, redis=None):
    return CameraLeaseManager(
        ttl_seconds=ttl, worker_id=worker_id, redis_client=redis or FakeRedis()
    )


class ClaimTests(unittest.TestCase):
    def test_claim_succeeds_for_an_unclaimed_camera(self):
        self.assertTrue(_manager().claim(1))

    def test_a_second_worker_cannot_claim_an_already_held_camera(self):
        redis = FakeRedis()
        self.assertTrue(_manager("worker-a", redis=redis).claim(1))
        self.assertFalse(_manager("worker-b", redis=redis).claim(1))

    def test_claim_succeeds_again_once_the_ttl_has_elapsed(self):
        """A crashed worker's lease must expire, not linger forever — this
        is the entire failover mechanism, with no reconciler needed."""
        redis = FakeRedis()
        _manager("worker-a", ttl=10, redis=redis).claim(1)
        redis.now += 11
        self.assertTrue(_manager("worker-b", redis=redis).claim(1))

    def test_claim_declines_on_a_redis_failure_rather_than_raising(self):
        """A Redis outage must never look like a successful claim — two
        workers both deciding they own a camera is the exact corruption
        per-camera ownership exists to prevent."""
        manager = _manager(redis=FakeRedis(fail=True))
        self.assertFalse(manager.claim(1))
        self.assertTrue(manager.degraded)


class RenewTests(unittest.TestCase):
    def test_renew_succeeds_while_still_owning_the_lease(self):
        redis = FakeRedis()
        manager = _manager("worker-a", redis=redis)
        manager.claim(1)
        self.assertTrue(manager.renew(1))

    def test_renew_actually_extends_the_ttl(self):
        redis = FakeRedis()
        manager = _manager("worker-a", ttl=10, redis=redis)
        manager.claim(1)
        redis.now += 8
        self.assertTrue(manager.renew(1))
        redis.now += 8  # 16s since claim, but only 8s since renew
        self.assertEqual(redis.get("decode:lease:cam:1"), "worker-a")

    def test_renew_fails_once_another_worker_has_claimed_the_lease(self):
        """The critical safety property: a worker that lost the renew race
        must find out, so it stops decoding rather than running alongside
        whoever now legitimately owns the camera."""
        redis = FakeRedis()
        a = _manager("worker-a", ttl=10, redis=redis)
        a.claim(1)
        redis.now += 11  # a's lease lapses
        b = _manager("worker-b", redis=redis)
        b.claim(1)
        self.assertFalse(a.renew(1))

    def test_renew_fails_for_a_camera_this_worker_never_held(self):
        self.assertFalse(_manager().renew(999))

    def test_renew_declines_on_a_redis_failure(self):
        manager = _manager(redis=FakeRedis(fail=True))
        self.assertFalse(manager.renew(1))
        self.assertTrue(manager.degraded)

    def test_degraded_clears_once_redis_recovers(self):
        redis = FakeRedis(fail=True)
        manager = _manager(redis=redis)
        manager.claim(1)
        self.assertTrue(manager.degraded)
        redis.fail = False
        manager.claim(1)
        self.assertFalse(manager.degraded)


class ReleaseTests(unittest.TestCase):
    def test_release_removes_a_lease_this_worker_owns(self):
        redis = FakeRedis()
        manager = _manager("worker-a", redis=redis)
        manager.claim(1)
        manager.release(1)
        self.assertIsNone(redis.get("decode:lease:cam:1"))

    def test_release_does_not_touch_a_lease_another_worker_now_owns(self):
        """If this worker's lease already lapsed and someone else claimed
        the camera, a slow release must not delete THEIR lease."""
        redis = FakeRedis()
        a = _manager("worker-a", ttl=10, redis=redis)
        a.claim(1)
        redis.now += 11
        b = _manager("worker-b", redis=redis)
        b.claim(1)
        a.release(1)  # a's stale release
        self.assertEqual(redis.get("decode:lease:cam:1"), "worker-b")

    def test_release_of_an_unheld_camera_does_not_raise(self):
        _manager().release(1)  # no exception

    def test_release_swallows_a_redis_failure(self):
        manager = _manager(redis=FakeRedis(fail=True))
        manager.release(1)  # no exception


class IsClaimedTests(unittest.TestCase):
    def test_false_when_nobody_holds_the_lease(self):
        self.assertFalse(_manager().is_claimed(1))

    def test_true_once_any_worker_holds_it(self):
        redis = FakeRedis()
        _manager("worker-a", redis=redis).claim(1)
        self.assertTrue(_manager("worker-b", redis=redis).is_claimed(1))

    def test_assumes_claimed_on_a_redis_failure(self):
        """A transient outage must not itself trigger a false 'nobody owns
        this camera' alert — assume claimed (silent) over assume free
        (a spurious page)."""
        manager = _manager(redis=FakeRedis(fail=True))
        self.assertTrue(manager.is_claimed(1))


class WorkerIdTests(unittest.TestCase):
    def test_a_generated_worker_id_is_non_empty(self):
        manager = CameraLeaseManager(ttl_seconds=10, redis_client=FakeRedis())
        self.assertTrue(manager.worker_id)

    def test_two_managers_get_different_generated_ids(self):
        a = CameraLeaseManager(ttl_seconds=10, redis_client=FakeRedis())
        b = CameraLeaseManager(ttl_seconds=10, redis_client=FakeRedis())
        self.assertNotEqual(a.worker_id, b.worker_id)


if __name__ == "__main__":
    unittest.main()
