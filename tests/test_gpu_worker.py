"""Unit tests for GPUInferenceWorker's queue, batching and correlation logic.

The arcface inference tests that used to live here moved to
tests/test_face_tasks.py when that code moved into its own Celery worker
(LSO-67 Stage 2). What remains is what stayed in this class: per-camera
queue keying by DB id (LSO-130) and the request/response correlation that
keeps a timed-out or dropped frame from permanently desyncing a camera
(LSO-138).

Run: PYTHONPATH=src python tests/test_gpu_worker.py
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np

from pipeline.gpu_worker import GPUInferenceWorker  # noqa: E402


class TestCameraSetKeying(unittest.TestCase):
    """LSO-130: queues are keyed by DB camera id, not list position.

    With positional keys, removing a camera from the middle of the set shifted
    every later camera's index by one — silently re-pointing their GPU queues
    at a different camera. That is why the engine rebuilt everything on a set
    change instead of removing one camera.
    """

    def _worker(self, ids):
        return GPUInferenceWorker(
            camera_ids=ids, metrics_collector=None
        )

    def test_ids_need_not_be_contiguous_or_ordered(self):
        w = self._worker([7, 22, 5])
        self.assertEqual(sorted(w._frame_in_queues), [5, 7, 22])

    def test_removing_a_middle_camera_leaves_the_others_addressable(self):
        w = self._worker([7, 22, 5])
        q7, q5 = w._frame_in_queues[7], w._frame_in_queues[5]

        w.remove_camera(22)

        # The exact objects must survive — not merely "a queue still exists".
        self.assertIs(w._frame_in_queues[7], q7)
        self.assertIs(w._frame_in_queues[5], q5)
        self.assertNotIn(22, w._frame_in_queues)
        self.assertEqual(w.camera_ids, [7, 5])

    def test_added_camera_gets_its_own_queues(self):
        w = self._worker([7])
        w.add_camera(31)
        for d in (
            w._frame_in_queues,
            w._detection_out_queues,
            w._face_in_queues,
            w._embedding_out_queues,
        ):
            self.assertIn(31, d)
        self.assertIsNot(w._frame_in_queues[31], w._frame_in_queues[7])

    def test_add_and_remove_are_idempotent(self):
        w = self._worker([7])
        w.add_camera(7)          # already present
        self.assertEqual(w.camera_ids, [7])
        w.remove_camera(99)      # never present
        self.assertEqual(w.camera_ids, [7])

    def test_submitting_to_a_removed_camera_is_a_noop(self):
        """A camera worker can still be draining its last cycle after removal —
        that must not raise into its thread."""
        w = self._worker([7])
        w.remove_camera(7)
        self.assertFalse(w.submit_frame(7, np.zeros((4, 4, 3), dtype=np.uint8), 1))
        self.assertIsNone(w.submit_faces(7, [], []))
        self.assertEqual(w.get_detections(7, 1, timeout=0.01), [])
        self.assertEqual(w.get_embeddings(7, 1, timeout=0.01), {})


class CountingMetrics:
    """Records only what LSO-138 asserts on."""

    def __init__(self):
        self.drops = 0
        self.stale = 0

    def record_drop(self, camera_id):
        self.drops += 1

    def record_stale_response(self, camera_id):
        self.stale += 1


class TestRequestResponseCorrelation(unittest.TestCase):
    """LSO-138: responses are matched to their request, not counted.

    Before this, a reply the caller had given up on stayed in the out-queue and
    was served to the next request — so every later frame got the *previous*
    frame's detections, permanently. Stale boxes on a live frame make the
    tracker switch IDs, which is what mints duplicate global tracks.
    """

    CAM = 7

    def _worker(self, metrics=None):
        return GPUInferenceWorker(
            camera_ids=[self.CAM],
            metrics_collector=metrics,
        )

    def _frame(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    # ── detection path ────────────────────────────────────────────────────────

    def test_detections_round_trip(self):
        """The happy path, which had no test before."""
        w = self._worker()
        self.assertTrue(w.submit_frame(self.CAM, self._frame(), 41))
        w._detection_out_queues[self.CAM].put((41, [{"id": "for-41"}]))

        self.assertEqual(
            w.get_detections(self.CAM, 41, timeout=0.5), [{"id": "for-41"}]
        )

    def test_a_stale_reply_is_discarded_not_returned(self):
        """The regression test: a timed-out request's late reply must not be
        handed to the next frame. Fails before the fix with the frame-40 boxes
        applied to frame 41."""
        m = CountingMetrics()
        w = self._worker(m)

        # Frame 40 timed out; its reply lands afterwards, unconsumed.
        w._detection_out_queues[self.CAM].put((40, [{"id": "stale-40"}]))
        w._detection_out_queues[self.CAM].put((41, [{"id": "fresh-41"}]))

        self.assertEqual(
            w.get_detections(self.CAM, 41, timeout=0.5), [{"id": "fresh-41"}]
        )
        self.assertEqual(m.stale, 1)

    def test_backlog_drains_rather_than_persisting(self):
        """Orphaned replies must not each cost one future frame.

        maxsize=2 caps the backlog at one stale reply plus the fresh one, so
        that is the deepest case the detection queue can actually reach.
        """
        w = self._worker()
        q = w._detection_out_queues[self.CAM]
        q.put((40, [{"id": "stale-40"}]))
        q.put((41, [{"id": "fresh-41"}]))
        self.assertTrue(q.full())

        self.assertEqual(
            w.get_detections(self.CAM, 41, timeout=0.5), [{"id": "fresh-41"}]
        )
        self.assertTrue(q.empty())

    def test_a_newer_reply_ends_the_wait_instead_of_burning_the_timeout(self):
        """submit_frame's drop-oldest can evict a frame that is already queued,
        so its reply never comes. Seeing a NEWER sequence proves that happened —
        keep waiting and the caller stalls for the full timeout, then discards
        the next good reply too."""
        m = CountingMetrics()
        w = self._worker(m)
        w._detection_out_queues[self.CAM].put((45, [{"id": "newer-45"}]))

        t0 = time.monotonic()
        self.assertEqual(w.get_detections(self.CAM, 41, timeout=2.0), [])
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual(m.stale, 1)

    def test_timeout_returns_empty_without_consuming_a_later_reply(self):
        w = self._worker()
        self.assertEqual(w.get_detections(self.CAM, 41, timeout=0.01), [])

    def test_a_dropped_frame_reports_failure(self):
        """maxsize=2: the third submit drops the oldest, and the caller must be
        told so it skips its get instead of eating another frame's result."""
        m = CountingMetrics()
        w = self._worker(m)
        self.assertTrue(w.submit_frame(self.CAM, self._frame(), 1))
        self.assertTrue(w.submit_frame(self.CAM, self._frame(), 2))

        # Queue is full; this drops the oldest and still succeeds.
        self.assertTrue(w.submit_frame(self.CAM, self._frame(), 3))
        self.assertEqual(m.drops, 1)

        queued = list(w._frame_in_queues[self.CAM].queue)
        self.assertEqual([n for _f, n in queued], [2, 3])

    # ── embedding path ────────────────────────────────────────────────────────

    def test_submit_faces_issues_increasing_sequences(self):
        w = self._worker()
        first = w.submit_faces(self.CAM, [], [])
        second = w.submit_faces(self.CAM, [], [])
        self.assertIsNotNone(first)
        self.assertEqual(second, first + 1)

    def test_embeddings_round_trip(self):
        w = self._worker()
        seq = w.submit_faces(self.CAM, [], [])
        w._embedding_out_queues[self.CAM].put((seq, {5: {"identity": "ok"}}))

        self.assertEqual(
            w.get_embeddings(self.CAM, seq, timeout=0.5), {5: {"identity": "ok"}}
        )

    def test_stale_embeddings_are_discarded(self):
        """Track ids give the embedding path partial cover — a stale map for a
        track that still exists silently attaches the wrong face to it."""
        m = CountingMetrics()
        w = self._worker(m)
        w.submit_faces(self.CAM, [], [])          # seq 1, timed out
        seq = w.submit_faces(self.CAM, [], [])    # seq 2

        w._embedding_out_queues[self.CAM].put((1, {5: {"identity": "stale"}}))
        w._embedding_out_queues[self.CAM].put((seq, {5: {"identity": "fresh"}}))

        self.assertEqual(
            w.get_embeddings(self.CAM, seq, timeout=0.5), {5: {"identity": "fresh"}}
        )
        self.assertEqual(m.stale, 1)

    def test_sequence_state_is_dropped_with_the_camera(self):
        w = self._worker()
        w.submit_faces(self.CAM, [], [])
        w.remove_camera(self.CAM)
        self.assertNotIn(self.CAM, w._face_seq)

        w.add_camera(self.CAM)
        self.assertEqual(w.submit_faces(self.CAM, [], []), 1)


if __name__ == "__main__":
    unittest.main()
