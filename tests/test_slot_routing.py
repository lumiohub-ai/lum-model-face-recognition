"""Unit tests for hash-to-slot camera routing (LSO-186).

The safety property that matters: `camera_slot` must be deterministic across
processes (decode and yolo compute it independently and MUST agree), and
stable across runs — so the golden-value tests below pin exact outputs. If
crc32 is ever swapped for something process-salted (e.g. builtin hash()),
these break, which is the point.

Run: PYTHONPATH=src python tests/test_slot_routing.py
"""

import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from workers.celery_app import (  # noqa: E402
    camera_slot,
    camera_queue_name,
    slot_queue_name,
)


class DeterminismTests(unittest.TestCase):
    def test_same_id_same_slot_every_call(self):
        self.assertEqual(camera_slot(5, 4), camera_slot(5, 4))

    def test_int_and_str_id_collide_to_one_slot(self):
        # decode passes an int, a stray caller might pass a str — both must
        # route identically or the camera splits across two workers.
        self.assertEqual(camera_slot(5, 4), camera_slot("5", 4))  # type: ignore[arg-type]

    def test_golden_values_are_pinned(self):
        # Exact crc32(str(id)) % N. These are the cross-process contract; a
        # change here means decode and yolo could disagree.
        self.assertEqual(camera_slot(5, 2), 0)
        self.assertEqual(camera_slot(23, 2), 0)
        self.assertEqual(camera_slot(2, 2), 1)
        self.assertEqual(camera_slot(12, 2), 1)

    def test_slot_is_in_range(self):
        for cam in range(1, 200):
            self.assertIn(camera_slot(cam, 3), (0, 1, 2))


class DistributionTests(unittest.TestCase):
    def test_spread_is_roughly_even(self):
        n = 4
        counts = Counter(camera_slot(cam, n) for cam in range(1, 401))
        # 400 ids over 4 slots ≈ 100 each; assert none is wildly starved.
        for slot in range(n):
            self.assertGreater(counts[slot], 60, f"slot {slot} starved: {counts}")


class QueueNameTests(unittest.TestCase):
    def test_queue_name_matches_slot_name(self):
        # yolo's producer side and the worker's consumer side must build the
        # identical queue string for the same slot.
        n = 2
        for cam in (2, 5, 12, 23, 30):
            slot = camera_slot(cam, n)
            self.assertEqual(camera_queue_name(cam), slot_queue_name(slot))

    def test_queue_name_uses_configured_slot_count(self):
        from config.settings import settings

        # camera_queue_name reads settings.camera_slot_count live.
        original = settings.camera_slot_count
        try:
            settings.camera_slot_count = 1
            # N=1 → everything routes to the single slot 0.
            self.assertEqual(camera_queue_name(999), "cam-slot-0")
        finally:
            settings.camera_slot_count = original


if __name__ == "__main__":
    unittest.main()
