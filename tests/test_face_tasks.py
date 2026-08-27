"""Unit tests for workers.face_tasks.embed_batch_task.

Ported from tests/test_gpu_worker.py's TestRunArcfaceBatch when the arcface
inference moved out of GPUInferenceWorker into its own Celery worker. The
behaviour these protect was originally fixed under LSO-117 and is unchanged
by that move — exactly the kind that breaks silently:

  - one embed_batch() call per crop, never a combined batch (batching all
    crops shipped as 0.3.0 and halved production FPS — ORT re-plans on every
    input-shape change)
  - results[i] corresponds to ROI i, even when some ROIs have no face at all
    — a misalignment here hands one person another person's embedding
  - a failed or empty embedding stays "no face" rather than borrowing a
    neighbour's vector

The task is called directly (not through a broker): these test the task
body, while the transport is covered by tests/test_frame_store.py and the
live end-to-end run.

Run: PYTHONPATH=src python tests/test_face_tasks.py
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np  # noqa: E402


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


def run_task(face_detector, rois):
    """Invoke the task body with the model holder and the shared-memory read
    both faked, so these stay unit tests of the inference logic rather than
    of the transport (which frame_store's own suite covers)."""
    from workers import face_tasks

    packed = [(i, r) for i, r in enumerate(rois)]
    # A handle whose `rois` length matches, so the task's n_rois bookkeeping
    # and its blank-result fallbacks line up with what we hand back.
    handle = {
        "camera_id": -1,
        "seq": 1,
        "rois": [
            {
                "track_id": i,
                "offset": 0,
                "height": 2,
                "width": 2,
                "channels": 3,
            }
            for i in range(len(rois))
        ],
    }

    with mock.patch.object(
        face_tasks.model_holder,
        "ensure_face_detector_loaded",
        return_value=face_detector,
    ), mock.patch(
        "workers.frame_store.attach_and_read_roi_batch", return_value=packed
    ):
        return face_tasks.embed_batch_task(handle)


class EmbedBatchTaskTests(unittest.TestCase):
    def test_each_crop_is_embedded_in_its_own_fixed_shape_call(self):
        rois = [roi(1), roi(2), roi(3)]
        crops = [crop(1), crop(2), crop(3)]
        faces = [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, c) for c in crops]
        fd = FakeFaceDetector(
            detect_results=[[f] for f in faces],
            embed_fn=lambda cs: [np.full(4, i, dtype=float) for i in range(len(cs))],
        )

        results = run_task(fd, rois)

        # One call per crop, each of size 1 - a varying batch size makes ORT
        # re-plan and costs ~30x (see face_tasks' embed loop comment).
        self.assertEqual(len(fd.embed_calls), 3)
        self.assertTrue(all(len(c) == 1 for c in fd.embed_calls))
        self.assertTrue(all(r["face_detected"] for r in results))

    def test_none_and_empty_roi_short_circuit_before_detection(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        rois = [None, empty, roi(1)]
        faces = [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(1))]
        fd = FakeFaceDetector(detect_results=[faces], embed_fn=lambda cs: [np.ones(4)])

        results = run_task(fd, rois)

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

        results = run_task(fd, rois)

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

        results = run_task(fd, rois)

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

        results = run_task(fd, rois)

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

        results = run_task(fd, rois)

        self.assertFalse(results[0]["face_detected"])
        self.assertTrue(results[1]["face_detected"])
        self.assertFalse(results[2]["face_detected"])
        self.assertTrue(results[3]["face_detected"])
        self.assertTrue(np.array_equal(results[1]["embedding"], np.full(4, 2.0)))
        self.assertTrue(np.array_equal(results[3]["embedding"], np.full(4, 4.0)))


class EmbedBatchTaskDegradationTests(unittest.TestCase):
    """The task must never raise into the caller. The GPU loop thread
    blocks on this task's result, so an exception here would
    surface as a task failure and make that thread wait out its whole
    timeout for nothing."""

    def test_empty_batch_returns_empty(self):
        from workers import face_tasks

        self.assertEqual(
            face_tasks.embed_batch_task({"camera_id": -1, "seq": 1, "rois": []}), []
        )

    def test_missing_shared_memory_returns_one_blank_per_roi(self):
        """The producer's slot vanished between write and read - every ROI
        gets the default no-face dict rather than an exception."""
        from workers import face_tasks

        handle = {
            "camera_id": -1,
            "seq": 7,
            "rois": [
                {"track_id": i, "offset": 0, "height": 2, "width": 2, "channels": 3}
                for i in range(3)
            ],
        }
        with mock.patch(
            "workers.frame_store.attach_and_read_roi_batch", return_value=None
        ):
            results = face_tasks.embed_batch_task(handle)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r["face_detected"] is False for r in results))
        self.assertTrue(all(r["embedding"] is None for r in results))

    def test_detector_blowing_up_returns_blanks_not_an_exception(self):
        rois = [roi(1), roi(2)]

        class ExplodingDetector:
            def detect_and_align(self, r):
                raise RuntimeError("model unloaded")

        results = run_task(ExplodingDetector(), rois)

        # detect_and_align's own failure is caught per-ROI, so this degrades
        # to two blank results rather than propagating.
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["face_detected"] is False for r in results))


if __name__ == "__main__":
    unittest.main()
