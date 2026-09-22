"""Unit tests for per-camera queue routing (LSO-218).

Replaces the LSO-186 hash-to-slot tests. A camera now routes to its own
`cam.<id>` queue; WHICH worker consumes it is decided by a short-lived Redis
lease at runtime (pipeline/camera_lease.py + camera_boot.py), not by a crc32
hash. So the only routing contract left to pin here is that the producer and
the consumer build the same queue string for a given camera, and that the
mapping is stable and injective.

Run: PYTHONPATH=src python -m pytest tests/test_camera_routing.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from workers.celery_app import camera_queue_name  # noqa: E402


class CameraQueueNameTests(unittest.TestCase):
    def test_is_per_camera(self):
        self.assertEqual(camera_queue_name(4), "cam.4")

    def test_same_camera_id_always_same_queue(self):
        # Sticky by construction: same id -> same queue -> the same lease
        # owner, so per-camera tracker state stays on one process.
        first = camera_queue_name(42)
        for _ in range(5):
            self.assertEqual(camera_queue_name(42), first)

    def test_distinct_cameras_get_distinct_queues(self):
        # The whole point of per-camera queues: no two cameras share one, so a
        # slow camera can't back up another's frames.
        queues = {camera_queue_name(c) for c in (2, 3, 12, 22, 24, 29)}
        self.assertEqual(len(queues), 6)

    def test_str_and_int_ids_agree(self):
        # decode passes an int; a stray caller might pass a str. Both must
        # name the same queue or the camera's frames split across workers.
        self.assertEqual(camera_queue_name(5), camera_queue_name("5"))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
