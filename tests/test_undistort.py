"""Unit tests for the shared P = K undistortion helper.

The property under test: undistort_image and undistort_points must land in the
identical coordinate space. Nothing in the live pipeline depends on that yet
(undistort_points has no runtime caller — see the module docstring in
src/domain/calibration/undistort.py), but the two are the halves of the future
runtime-projection fix, and a mismatch between them would reintroduce the
homography drift at the moment it is finally wired up.

Run: PYTHONPATH=src python tests/test_undistort.py
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from domain.calibration.undistort import undistort_image, undistort_points  # noqa: E402


K_FISHEYE = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
D_FISHEYE = np.array([-0.05, 0.01, -0.002, 0.0003])

K_PINHOLE = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
D_PINHOLE = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])


def _distort_fisheye_points(undistorted_pts: np.ndarray) -> np.ndarray:
    """Forward-project P=K undistorted pixels back into raw fisheye pixels."""
    norm = cv2.undistortPoints(
        undistorted_pts.reshape(-1, 1, 2), K_FISHEYE, np.zeros(5)
    ).reshape(-1, 2)
    obj = np.hstack([norm, np.ones((len(norm), 1))]).reshape(-1, 1, 3)
    dist_px, _ = cv2.fisheye.projectPoints(
        obj, np.zeros(3), np.zeros(3), K_FISHEYE, D_FISHEYE
    )
    return dist_px.reshape(-1, 2)


class FisheyeRoundTripTests(unittest.TestCase):
    def test_undistort_points_inverts_distortion_in_K_space(self):
        undistorted = np.array([[600.0, 300.0], [960.0, 540.0], [1500.0, 900.0]])
        distorted = _distort_fisheye_points(undistorted)

        recovered = undistort_points(
            distorted.reshape(-1, 1, 2), K_FISHEYE, D_FISHEYE, "fisheye"
        ).reshape(-1, 2)

        np.testing.assert_allclose(recovered, undistorted, atol=1e-6)

    def test_wrong_new_camera_matrix_would_diverge_significantly(self):
        """Sanity check that the P choice actually matters (motivates the test above)."""
        undistorted = np.array([[600.0, 300.0]])
        distorted = _distort_fisheye_points(undistorted)

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_FISHEYE, D_FISHEYE, (1920, 1080), np.eye(3), balance=0.0
        )
        wrong = cv2.fisheye.undistortPoints(
            distorted.reshape(-1, 1, 2), K_FISHEYE, D_FISHEYE, R=np.eye(3), P=new_K
        ).reshape(-1, 2)

        self.assertGreater(np.linalg.norm(wrong[0] - undistorted[0]), 10.0)


class ImagePointAgreementTests(unittest.TestCase):
    def test_marked_pixel_lands_where_point_path_predicts(self):
        """The test that would catch a P mismatch between the two call sites."""
        w, h = 640, 480
        K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
        D = np.array([-0.15, 0.02, -0.001, 0.0002])

        # Build a raw (distorted) synthetic image with one bright marker pixel.
        undistorted_target = np.array([[420.0, 260.0]])
        raw_target = _mark_via_distortion(undistorted_target, K, D, w, h)

        raw_img = np.zeros((h, w, 3), dtype=np.uint8)
        rx, ry = int(round(raw_target[0][0])), int(round(raw_target[0][1]))
        cv2.circle(raw_img, (rx, ry), 3, (255, 255, 255), -1)

        undistorted_img = undistort_image(raw_img, K, D, "fisheye")

        # Where the point path says the marker should be:
        predicted = undistort_points(
            raw_target.reshape(-1, 1, 2), K, D, "fisheye"
        ).reshape(-1, 2)[0]

        ys, xs = np.where(undistorted_img[:, :, 0] > 200)
        self.assertTrue(len(xs) > 0, "marker not found in undistorted image")
        actual_centroid = np.array([xs.mean(), ys.mean()])

        np.testing.assert_allclose(actual_centroid, predicted, atol=3.0)


def _mark_via_distortion(undistorted_pts, K, D, w, h):
    norm = cv2.undistortPoints(
        undistorted_pts.reshape(-1, 1, 2), K, np.zeros(5)
    ).reshape(-1, 2)
    obj = np.hstack([norm, np.ones((len(norm), 1))]).reshape(-1, 1, 3)
    dist_px, _ = cv2.fisheye.projectPoints(obj, np.zeros(3), np.zeros(3), K, D)
    return dist_px.reshape(-1, 2)


class ModelAliasTests(unittest.TestCase):
    def test_standard_and_pinhole_both_take_pinhole_branch(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 240), 5, (255, 255, 255), -1)

        out_standard = undistort_image(img, K_PINHOLE, D_PINHOLE, "standard")
        out_pinhole = undistort_image(img, K_PINHOLE, D_PINHOLE, "pinhole")

        np.testing.assert_array_equal(out_standard, out_pinhole)

    def test_pinhole_point_path_matches_image_path(self):
        w, h = 640, 480
        undistorted_target = np.array([[400.0, 260.0]])
        norm = cv2.undistortPoints(
            undistorted_target.reshape(-1, 1, 2), K_PINHOLE, np.zeros(5)
        ).reshape(-1, 2)
        obj = np.hstack([norm, np.ones((len(norm), 1))]).reshape(-1, 1, 3)
        raw_target, _ = cv2.projectPoints(
            obj, np.zeros(3), np.zeros(3), K_PINHOLE, D_PINHOLE
        )
        raw_target = raw_target.reshape(-1, 2)

        raw_img = np.zeros((h, w, 3), dtype=np.uint8)
        rx, ry = int(round(raw_target[0][0])), int(round(raw_target[0][1]))
        cv2.circle(raw_img, (rx, ry), 3, (255, 255, 255), -1)

        undistorted_img = undistort_image(raw_img, K_PINHOLE, D_PINHOLE, "pinhole")
        predicted = undistort_points(
            raw_target.reshape(-1, 1, 2), K_PINHOLE, D_PINHOLE, "pinhole"
        ).reshape(-1, 2)[0]

        ys, xs = np.where(undistorted_img[:, :, 0] > 200)
        self.assertTrue(len(xs) > 0, "marker not found in undistorted image")
        actual_centroid = np.array([xs.mean(), ys.mean()])

        np.testing.assert_allclose(actual_centroid, predicted, atol=3.0)


class DistCoeffValidationTests(unittest.TestCase):
    """Both models validate D and neither coerces it."""

    # A standard calibration that reaches the fisheye branch via main.py's
    # 'fisheye' default — the case that used to be sliced to k1..k4 and warped.
    STANDARD_5 = [-0.2, 0.05, 0.0, 0.0, 0.01]

    def test_pinhole_rejects_bad_dist_coeff_length(self):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        for bad in ([0.1, 0.2, 0.3], [0.1] * 6, [0.1] * 7):
            with self.subTest(n=len(bad)):
                with self.assertRaises(ValueError) as ctx:
                    undistort_image(img, K_PINHOLE, bad, "pinhole")
                self.assertIn(str(len(bad)), str(ctx.exception))

    def test_pinhole_accepts_supported_dist_coeff_lengths(self):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        for n in (4, 5, 8, 12, 14):
            with self.subTest(n=n):
                undistort_image(img, K_PINHOLE, [0.0] * n, "pinhole")

    def test_fisheye_requires_exactly_four_dist_coeffs(self):
        # cv2.fisheye asserts on every other length (fisheye.cpp:518).
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        for n in (2, 3, 5, 8):
            with self.subTest(n=n):
                with self.assertRaises(ValueError) as ctx:
                    undistort_image(img, K_FISHEYE, [0.0] * n, "fisheye")
                self.assertIn(str(n), str(ctx.exception))

    def test_fisheye_accepts_exactly_four(self):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        undistort_image(img, K_FISHEYE, [0.0] * 4, "fisheye")

    def test_standard_calibration_is_not_reinterpreted_as_fisheye(self):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            undistort_image(img, K_FISHEYE, self.STANDARD_5, "fisheye")
        undistort_image(img, K_PINHOLE, self.STANDARD_5, "pinhole")

    def test_missing_dist_coeffs_is_rejected(self):
        # np.array(None) is [nan]: an all-zero remap, i.e. a black frame.
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        for model, K in (("fisheye", K_FISHEYE), ("pinhole", K_PINHOLE)):
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    undistort_image(img, K, None, model)

    def test_non_finite_values_are_rejected(self):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        for bad in ([float("nan"), 0.0, 0.0, 0.0], [float("inf"), 0.0, 0.0, 0.0]):
            with self.subTest(v=bad[0]):
                with self.assertRaises(ValueError):
                    undistort_image(img, K_FISHEYE, bad, "fisheye")

    def test_non_finite_camera_matrix_is_rejected(self):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        bad_K = [[float("nan"), 0.0, 60.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        with self.assertRaises(ValueError):
            undistort_image(img, bad_K, [0.0] * 4, "fisheye")


if __name__ == "__main__":
    unittest.main()
