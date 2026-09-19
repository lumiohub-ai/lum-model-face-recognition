"""Unit tests for the worker-liveness checks (LSO-214).

The check_* functions are the logic that decides UP/DOWN per component; they call
the module's _consumed_queues() / _redis() helpers, which we patch so no broker
or Redis is needed. worker_liveness only imports config.settings + loguru at
module level (celery/redis are lazy-imported inside the helpers), so this runs
without the model stack.

Run: PYTHONPATH=src python -m pytest tests/test_worker_liveness.py
"""

import importlib.util
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

# Importing workers.worker_liveness pulls workers/__init__ -> celery_app ->
# config/__init__ -> config.vision -> lum_vision (eager), so this can't import
# without the model stack — same skip pattern as test_engine_reload.py.
_IMPORTABLE = importlib.util.find_spec("lum_vision") is not None


class _FakeRedis:
    def __init__(self, kv):
        self._kv = dict(kv)

    def scan_iter(self, match=None, count=None):
        import fnmatch
        return [k for k in self._kv if fnmatch.fnmatch(k, match or "*")]

    def get(self, k):
        return self._kv.get(k)


@unittest.skipUnless(_IMPORTABLE, "loguru not importable")
class CeleryComponentTests(unittest.TestCase):
    def _run(self, consumed):
        from workers import worker_liveness as wl
        with mock.patch.object(wl, "_consumed_queues", return_value=consumed), \
             mock.patch.object(wl.settings, "camera_slot_count", 2):
            return wl.check_celery_components()

    def test_all_up_when_every_queue_consumed(self):
        consumed = {"cam-slot-0", "cam-slot-1", "yolo", "face", "reid",
                    "globaltrack", "embeddings", "detections"}
        res = self._run(consumed)
        self.assertTrue(all(ok for ok, _ in res.values()), res)

    def test_camera_worker_down_if_a_slot_queue_unconsumed(self):
        consumed = {"cam-slot-0", "yolo", "face", "reid", "globaltrack",
                    "embeddings", "detections"}  # cam-slot-1 missing
        res = self._run(consumed)
        self.assertFalse(res["camera-worker"][0])
        self.assertIn("cam-slot-1", res["camera-worker"][1])
        self.assertTrue(res["yolo-worker"][0])

    def test_all_down_when_broker_unreachable(self):
        res = self._run(None)
        self.assertTrue(all(not ok for ok, _ in res.values()), res)


@unittest.skipUnless(_IMPORTABLE, "loguru not importable")
class SlotLeaseTests(unittest.TestCase):
    def _run(self, kv):
        from workers import worker_liveness as wl
        with mock.patch.object(wl, "_redis", return_value=_FakeRedis(kv)), \
             mock.patch.object(wl.settings, "camera_slot_count", 2):
            return wl.check_slot_lease()

    def test_healthy_two_slots_two_distinct_holders(self):
        ok, _ = self._run({"track:slot:lease:0": "wA", "track:slot:lease:1": "wB"})
        self.assertTrue(ok)

    def test_starvation_one_slot_unclaimed(self):
        ok, msg = self._run({"track:slot:lease:0": "wA"})  # slot 1 missing
        self.assertFalse(ok)
        self.assertIn("got 1 keys", msg)

    def test_collision_same_holder_two_slots(self):
        ok, msg = self._run({"track:slot:lease:0": "wA", "track:slot:lease:1": "wA"})
        self.assertFalse(ok)  # 2 keys but only 1 distinct holder


@unittest.skipUnless(_IMPORTABLE, "loguru not importable")
class DecodeTests(unittest.TestCase):
    def _run(self, kv):
        from workers import worker_liveness as wl
        with mock.patch.object(wl, "_redis", return_value=_FakeRedis(kv)):
            return wl.check_decode()

    def test_up_when_decode_leases_present(self):
        ok, _ = self._run({"decode:lease:cam:4": "w", "decode:lease:cam:5": "w"})
        self.assertTrue(ok)

    def test_down_when_no_decode_leases(self):
        ok, _ = self._run({"track:slot:lease:0": "w"})  # unrelated key only
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
