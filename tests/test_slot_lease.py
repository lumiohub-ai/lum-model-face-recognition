"""Unit tests for SlotLeaseManager (LSO-186).

Reuses test_camera_lease.py's FakeRedis pattern (explicit clock, script_load/
evalsha for the compare-and-act Lua). The behaviour that matters: N identical
replicas each end up on a DISTINCT slot, a crashed replica's slot frees on TTL,
and claim_free_slot fails (None) rather than double-assigning when every slot
is taken.

Run: PYTHONPATH=src python tests/test_slot_lease.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from pipeline.slot_lease import (  # noqa: E402
    SlotLeaseManager,
    _RELEASE_SCRIPT,
    _RENEW_SCRIPT,
)


class FakeRedis:
    """Same shape as test_camera_lease.py's FakeRedis, matching slot_lease's
    own copies of the renew/release scripts."""

    def __init__(self, fail=False):
        self.store = {}
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


def _mgr(worker_id, n=2, ttl=10, redis=None):
    return SlotLeaseManager(
        n_slots=n, ttl_seconds=ttl, worker_id=worker_id, redis_client=redis
    )


class ClaimTests(unittest.TestCase):
    def test_first_replica_takes_slot_zero(self):
        self.assertEqual(_mgr("a", redis=FakeRedis()).claim_free_slot(), 0)

    def test_replicas_get_distinct_slots(self):
        redis = FakeRedis()
        self.assertEqual(_mgr("a", n=2, redis=redis).claim_free_slot(), 0)
        self.assertEqual(_mgr("b", n=2, redis=redis).claim_free_slot(), 1)

    def test_surplus_replica_gets_none_when_all_slots_taken(self):
        """replicas > N: the extra worker must fail (None), never share a slot
        — sharing would split a camera's frames across two trackers."""
        redis = FakeRedis()
        self.assertEqual(_mgr("a", n=2, redis=redis).claim_free_slot(), 0)
        self.assertEqual(_mgr("b", n=2, redis=redis).claim_free_slot(), 1)
        self.assertIsNone(_mgr("c", n=2, redis=redis).claim_free_slot())

    def test_freed_slot_is_reclaimed_after_ttl(self):
        """A crashed replica stops renewing → its slot expires → a restarted
        replica reclaims it. The whole self-healing mechanism."""
        redis = FakeRedis()
        self.assertEqual(_mgr("a", n=2, ttl=10, redis=redis).claim_free_slot(), 0)
        self.assertEqual(_mgr("b", n=2, ttl=10, redis=redis).claim_free_slot(), 1)
        redis.now += 11  # a's lease lapses
        self.assertEqual(_mgr("c", n=2, redis=redis).claim_free_slot(), 0)

    def test_claim_returns_none_on_redis_failure(self):
        self.assertIsNone(_mgr("a", redis=FakeRedis(fail=True)).claim_free_slot())


class RenewTests(unittest.TestCase):
    def test_renew_extends_a_held_slot(self):
        redis = FakeRedis()
        m = _mgr("a", ttl=10, redis=redis)
        slot = m.claim_free_slot()
        redis.now += 8
        self.assertTrue(m.renew(slot))
        redis.now += 8  # 16s since claim, 8s since renew → still held
        self.assertEqual(redis.get("track:slot:lease:0"), "a")

    def test_renew_fails_after_another_worker_took_the_slot(self):
        redis = FakeRedis()
        a = _mgr("a", n=1, ttl=10, redis=redis)
        a.claim_free_slot()
        redis.now += 11
        b = _mgr("b", n=1, redis=redis)
        b.claim_free_slot()
        self.assertFalse(a.renew(0))

    def test_renew_returns_none_on_redis_failure(self):
        # None (not False) so the caller can tell a transient blip apart from
        # a real ownership loss and not restart on the former.
        self.assertIsNone(_mgr("a", redis=FakeRedis(fail=True)).renew(0))

    def test_renew_returns_false_when_slot_reassigned(self):
        redis = FakeRedis()
        a = _mgr("a", n=1, ttl=10, redis=redis)
        a.claim_free_slot()
        redis.now += 11
        _mgr("b", n=1, redis=redis).claim_free_slot()
        self.assertIs(a.renew(0), False)


class ReleaseTests(unittest.TestCase):
    def test_release_frees_a_slot_immediately(self):
        redis = FakeRedis()
        m = _mgr("a", redis=redis)
        slot = m.claim_free_slot()
        m.release(slot)
        self.assertIsNone(redis.get("track:slot:lease:0"))

    def test_release_does_not_touch_another_workers_slot(self):
        redis = FakeRedis()
        a = _mgr("a", n=1, ttl=10, redis=redis)
        a.claim_free_slot()
        redis.now += 11
        b = _mgr("b", n=1, redis=redis)
        b.claim_free_slot()
        a.release(0)  # stale release must not delete b's lease
        self.assertEqual(redis.get("track:slot:lease:0"), "b")

    def test_release_swallows_redis_failure(self):
        _mgr("a", redis=FakeRedis(fail=True)).release(0)  # no raise


class ConstructionTests(unittest.TestCase):
    def test_rejects_zero_slots(self):
        with self.assertRaises(ValueError):
            SlotLeaseManager(n_slots=0, ttl_seconds=10, redis_client=FakeRedis())


if __name__ == "__main__":
    unittest.main()
