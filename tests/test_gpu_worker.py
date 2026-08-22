"""Unit tests for GPUInferenceWorker._run_arcface_batch (LSO-117).

Covers the index bookkeeping across detect_and_align() (per-ROI) ->
embed_batch() (one crop per call) -> results[i] (scattered back by index) -
exactly the kind of filter/scatter logic that breaks silently if it regresses.

Run: PYTHONPATH=src python tests/test_gpu_worker.py
"""

import os
import sys
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
        detector=None, face_detector=face_detector, num_cameras=1, metrics_collector=None
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


if __name__ == "__main__":
    unittest.main()
