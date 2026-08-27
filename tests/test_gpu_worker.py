"""Unit tests for GPUInferenceWorker._run_arcface_batch (LSO-117).

Covers the index bookkeeping across detect_and_align() (per-ROI) ->
embed_batch() (one crop per call) -> results[i] (scattered back by index) -
exactly the kind of filter/scatter logic that breaks silently if it regresses.

Run: PYTHONPATH=src python tests/test_gpu_worker.py
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np

from pipeline.gpu_worker import GPUInferenceWorker  # noqa: E402


class FakeFace:
    def __init__(self, bbox, kps, det_score, aligned_crop):
        self.bbox = np.array(bbox, dtype=float)
        self.kps = np.array(kps, dtype=float) if kps is not None else None
        self.det_score = det_score
        self.aligned_crop = aligned_crop


def roi(tag):
    """A distinct, valid (non-empty) ROI stand-in - content doesn't matter,
    only identity/order, so each is tagged in its one pixel for debugging."""
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    a[0, 0, 0] = tag
    return a


def crop(tag):
    return np.full((112, 112, 3), tag, dtype=np.uint8)


class FakeFaceDetector:
    """detect_and_align returns from a caller-supplied, per-call list (one
    entry per detect_and_align() call, in order); embed_batch is a plain
    function over the accumulated crop list. Both call histories are
    recorded so tests can assert exactly how the real detector was used."""

    def __init__(self, detect_results, embed_fn=None, embed_exception=None):
        self._detect_results = list(detect_results)
        self.detect_calls = []
        self.embed_calls = []
        self._embed_fn = embed_fn
        self._embed_exception = embed_exception

    def detect_and_align(self, r):
        self.detect_calls.append(r)
        return self._detect_results[len(self.detect_calls) - 1]

    def embed_batch(self, crops):
        self.embed_calls.append(list(crops))
        if self._embed_exception is not None:
            raise self._embed_exception
        return self._embed_fn(crops)


def make_worker(face_detector):
    return GPUInferenceWorker(
        face_detector=face_detector, camera_ids=[1], metrics_collector=None
    )


class TestRunArcfaceBatch(unittest.TestCase):
    def test_each_crop_is_embedded_in_its_own_fixed_shape_call(self):
        rois = [roi(1), roi(2), roi(3)]
        crops = [crop(1), crop(2), crop(3)]
        faces = [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, c) for c in crops]
        fd = FakeFaceDetector(
            detect_results=[[f] for f in faces],
            embed_fn=lambda cs: [np.full(4, i, dtype=float) for i in range(len(cs))],
        )

        results = make_worker(fd)._run_arcface_batch(rois)

        # One call per crop, each of size 1 - a varying batch size makes ORT
        # re-plan and costs ~30x (see _run_arcface_batch).
        self.assertEqual(len(fd.embed_calls), 3)
        self.assertTrue(all(len(c) == 1 for c in fd.embed_calls))
        self.assertTrue(all(r["face_detected"] for r in results))

    def test_none_and_empty_roi_short_circuit_before_detection(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        rois = [None, empty, roi(1)]
        faces = [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(1))]
        fd = FakeFaceDetector(detect_results=[faces], embed_fn=lambda cs: [np.ones(4)])

        results = make_worker(fd)._run_arcface_batch(rois)

        # detect_and_align is only called for the one real ROI - None/empty
        # never reach it.
        self.assertEqual(len(fd.detect_calls), 1)
        self.assertFalse(results[0]["face_detected"])
        self.assertFalse(results[1]["face_detected"])
        self.assertTrue(results[2]["face_detected"])

    def test_face_without_landmarks_stays_no_face_and_is_not_embedded(self):
        rois = [roi(1), roi(2)]
        # roi(1)'s face has no kps -> detect_and_align would leave aligned_crop
        # None (can't align without landmarks); roi(2)'s face is normal.
        faces = [
            [FakeFace([0, 0, 1, 1], None, 0.9, aligned_crop=None)],
            [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(2))],
        ]
        fd = FakeFaceDetector(detect_results=faces, embed_fn=lambda cs: [np.ones(4)])

        results = make_worker(fd)._run_arcface_batch(rois)

        self.assertFalse(results[0]["face_detected"])
        self.assertTrue(results[1]["face_detected"])
        # Only the alignable face's crop reached embed_batch.
        self.assertEqual(len(fd.embed_calls[0]), 1)

    def test_embed_failure_blanks_only_that_face(self):
        # Embedding fails for crop(1) only; crop(2) must still come back.
        rois = [roi(1), roi(2)]
        faces = [
            [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(1))],
            [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(2))],
        ]
        def flaky(cs):
            if cs[0][0, 0, 0] == 1:
                raise RuntimeError("CUDA OOM")
            return [np.full(4, 2.0)]

        fd = FakeFaceDetector(detect_results=faces, embed_fn=flaky)

        results = make_worker(fd)._run_arcface_batch(rois)

        self.assertFalse(results[0]["face_detected"])
        self.assertIsNone(results[0]["embedding"])
        self.assertTrue(results[1]["face_detected"])
        self.assertTrue(np.array_equal(results[1]["embedding"], np.full(4, 2.0)))

    def test_empty_embed_result_leaves_that_face_unrecognized(self):
        # embed_batch returns nothing for a crop - that face must stay
        # "no face" rather than being paired with a neighbour's embedding.
        rois = [roi(1), roi(2)]
        faces = [
            [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(i))] for i in (1, 2)
        ]

        def empty_for_first(cs):
            return [] if cs[0][0, 0, 0] == 1 else [np.full(4, 2.0)]

        fd = FakeFaceDetector(detect_results=faces, embed_fn=empty_for_first)

        results = make_worker(fd)._run_arcface_batch(rois)

        self.assertFalse(results[0]["face_detected"])
        self.assertTrue(results[1]["face_detected"])
        self.assertTrue(np.array_equal(results[1]["embedding"], np.full(4, 2.0)))

    def test_mixed_hit_and_miss_rois_keep_correct_index_alignment(self):
        # roi(1): no face at all. roi(2): a face. roi(3): no face. roi(4): a face.
        # Regression target: results[i] must match roi i, not just "some ROI."
        rois = [roi(1), roi(2), roi(3), roi(4)]
        faces = [
            [],
            [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(2))],
            [],
            [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(4))],
        ]
        fd = FakeFaceDetector(
            detect_results=faces,
            embed_fn=lambda cs: [np.full(4, float(cs[0][0, 0, 0]))],
        )

        results = make_worker(fd)._run_arcface_batch(rois)

        self.assertFalse(results[0]["face_detected"])
        self.assertTrue(results[1]["face_detected"])
        self.assertFalse(results[2]["face_detected"])
        self.assertTrue(results[3]["face_detected"])
        self.assertTrue(np.array_equal(results[1]["embedding"], np.full(4, 2.0)))
        self.assertTrue(np.array_equal(results[3]["embedding"], np.full(4, 4.0)))



class TestCameraSetKeying(unittest.TestCase):
    """LSO-130: queues are keyed by DB camera id, not list position.

    With positional keys, removing a camera from the middle of the set shifted
    every later camera's index by one — silently re-pointing their GPU queues
    at a different camera. That is why the engine rebuilt everything on a set
    change instead of removing one camera.
    """

    def _worker(self, ids):
        return GPUInferenceWorker(
            face_detector=None, camera_ids=ids, metrics_collector=None
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
            face_detector=None, camera_ids=[self.CAM],
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
