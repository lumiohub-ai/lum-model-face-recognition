"""Unit tests for GPUInferenceWorker._run_arcface_batch (LSO-117).

Covers the index bookkeeping across detect_and_align() (per-ROI) ->
embed_batch() (one batched call) -> results[i] (scattered back by index) -
exactly the kind of filter/batch/scatter logic that breaks silently if it
regresses, per repeated code review feedback on PR #77.

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
    def test_multiple_rois_batch_into_a_single_embed_batch_call(self):
        rois = [roi(1), roi(2), roi(3)]
        crops = [crop(1), crop(2), crop(3)]
        faces = [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, c) for c in crops]
        fd = FakeFaceDetector(
            detect_results=[[f] for f in faces],
            embed_fn=lambda cs: [np.full(4, i, dtype=float) for i in range(len(cs))],
        )

        results = make_worker(fd)._run_arcface_batch(rois)

        # The whole point of LSO-117: one call for every ROI's face, not one per face.
        self.assertEqual(len(fd.embed_calls), 1)
        self.assertEqual(len(fd.embed_calls[0]), 3)
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

    def test_embed_batch_raising_blanks_the_whole_cycle(self):
        # Both ROIs detect fine; embedding then fails for the whole batch -
        # both must fall back to "no face," not just one.
        rois = [roi(1), roi(2)]
        faces = [
            [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(1))],
            [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(2))],
        ]
        fd = FakeFaceDetector(detect_results=faces, embed_exception=RuntimeError("CUDA OOM"))

        results = make_worker(fd)._run_arcface_batch(rois)

        self.assertFalse(any(r["face_detected"] for r in results))
        self.assertIsNone(results[0]["embedding"])
        self.assertIsNone(results[1]["embedding"])

    def test_embed_batch_length_mismatch_is_discarded_not_misaligned(self):
        # embed_batch violates its "one embedding per crop" contract by
        # returning fewer items than it was given. Must NOT zip() the
        # mismatched lists (which would silently attach face 2's embedding
        # to face 1, etc.) - every face this cycle must come back as
        # "no face" instead, loudly logged elsewhere.
        rois = [roi(1), roi(2), roi(3)]
        faces = [
            [FakeFace([0, 0, 1, 1], [[1, 1]], 0.9, crop(i))] for i in (1, 2, 3)
        ]
        fd = FakeFaceDetector(
            detect_results=faces,
            embed_fn=lambda cs: [np.ones(4)],  # 1 embedding for 3 crops - contract violation
        )

        results = make_worker(fd)._run_arcface_batch(rois)

        self.assertFalse(any(r["face_detected"] for r in results))
        self.assertTrue(all(r["embedding"] is None for r in results))

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
            embed_fn=lambda cs: [np.full(4, 2.0), np.full(4, 4.0)],
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
