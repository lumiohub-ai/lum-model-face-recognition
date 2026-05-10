"""Unit tests for the floor homography solver.

Run with: `python -m unittest tests.test_homography` (or via pytest if available)
from the repo root with PYTHONPATH=src.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from domain.calibration.homography import compute_homography  # noqa: E402


SQUARE = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]


def _apply_h(H: np.ndarray, pts: list) -> list:
    src = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(src, H).reshape(-1, 2).tolist()


class HomographyTests(unittest.TestCase):
    def test_identity_when_src_equals_dst(self):
        result = compute_homography(SQUARE, SQUARE)
        H = np.array(result["homography_matrix"]) / result["homography_matrix"][2][2]
        np.testing.assert_allclose(H, np.eye(3), atol=1e-9)
        self.assertLess(result["reprojection_error"], 1e-6)
        self.assertEqual(result["method"], "DLT")
        self.assertIsNone(result["inlier_mask"])

    def test_recovers_synthetic_homography(self):
        H_true = np.array(
            [[1.2, 0.1, 30.0], [0.05, 0.9, -10.0], [1e-4, 2e-4, 1.0]],
            dtype=np.float64,
        )
        dst = _apply_h(H_true, SQUARE)
        result = compute_homography(SQUARE, dst)
        H_est = np.array(result["homography_matrix"])
        # Normalize both to compare up to scale
        H_true_norm = H_true / H_true[2, 2]
        H_est_norm = H_est / H_est[2, 2]
        np.testing.assert_allclose(H_est_norm, H_true_norm, atol=1e-4)
        self.assertLess(result["reprojection_error"], 1e-4)

    def test_collinear_src_points_raise(self):
        collinear = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
        dst = [[0.0, 0.0], [10.0, 5.0], [20.0, 10.0], [30.0, 15.0]]
        with self.assertRaises(ValueError):
            compute_homography(collinear, dst)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            compute_homography(SQUARE[:3], SQUARE[:3])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_homography(SQUARE, SQUARE[:3])

    def test_ransac_marks_outlier(self):
        H_true = np.array(
            [[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        src = SQUARE + [[50.0, 50.0], [25.0, 75.0]]
        dst = _apply_h(H_true, src)
        # Inject a large outlier on the 5th point
        dst[4] = [dst[4][0] + 200.0, dst[4][1] - 200.0]
        result = compute_homography(src, dst)
        self.assertEqual(result["method"], "RANSAC")
        self.assertIsNotNone(result["inlier_mask"])
        self.assertEqual(result["inlier_mask"][4], 0)


if __name__ == "__main__":
    unittest.main()
