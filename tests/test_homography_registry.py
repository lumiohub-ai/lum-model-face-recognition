"""Unit tests for HomographyRegistry's frame_source-driven space selection.

Per docs/calibration-plan.md item 3: `frame_source` on camera_map_positions
is the source of truth for which space a stored H was fit in, not the
camera's calibration_status. These tests pin that decision (made once at
DB-load time) without touching a real DB — HomographyRepository is mocked.

Run: PYTHONPATH=src python tests/test_homography_registry.py
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from domain.calibration.homography_registry import HomographyRegistry  # noqa: E402
from infrastructure.storage.homography_repository import CameraProjectionRow  # noqa: E402

H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
K = [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
D = [-0.1, 0.02, -0.001, 0.0002]


def _row(**overrides):
    defaults = dict(
        homography_matrix=H,
        map_id=1,
        frame_source=None,
        calibration_status=None,
        calibration_matrix=None,
        distortion_coefficients=None,
        calibration_model=None,
    )
    defaults.update(overrides)
    return CameraProjectionRow(**defaults)


def _load(row):
    registry = HomographyRegistry()
    with patch(
        "domain.calibration.homography_registry.HomographyRepository"
    ) as mock_repo_cls:
        mock_repo_cls.return_value.get_for_camera.return_value = row
        return registry.get("test-slug", 7)


class FrameSourceGatingTests(unittest.TestCase):
    def test_null_frame_source_means_raw(self):
        proj = _load(_row(frame_source=None))
        self.assertFalse(proj.undistort)

    def test_captured_raw_means_raw(self):
        proj = _load(_row(frame_source="captured_raw"))
        self.assertFalse(proj.undistort)

    def test_legacy_value_means_raw(self):
        proj = _load(_row(frame_source="something_old"))
        self.assertFalse(proj.undistort)

    def test_captured_undistorted_with_intrinsics_means_undistort(self):
        proj = _load(
            _row(
                frame_source="captured_undistorted",
                calibration_matrix=K,
                distortion_coefficients=D,
                calibration_model="fisheye",
            )
        )
        self.assertTrue(proj.undistort)
        self.assertIsNotNone(proj.K)
        self.assertIsNotNone(proj.D)
        self.assertEqual(proj.model, "fisheye")

    def test_captured_undistorted_without_intrinsics_falls_back_to_raw(self):
        proj = _load(
            _row(frame_source="captured_undistorted", calibration_matrix=None)
        )
        self.assertFalse(proj.undistort)

    def test_stale_raw_homography_on_calibrated_camera_still_projects_raw(self):
        """Mismatch guard: warn but degrade gracefully, don't break outright."""
        proj = _load(
            _row(
                frame_source="captured_raw",
                calibration_status="calibrated",
                calibration_matrix=K,
                distortion_coefficients=D,
            )
        )
        self.assertFalse(proj.undistort)

    def test_map_id_passed_through(self):
        proj = _load(_row(map_id=42))
        self.assertEqual(proj.map_id, 42)


class InvalidateOnCameraCalibrationSavedTests(unittest.TestCase):
    """Recalibrate on the Lens Calibration tab changes K/D on a camera that
    already has a stable H — no HomographyCalibrated event fires for that.
    CameraCalibrationSaved (docs/calibration-plan.md §5) is a pure invalidation
    signal for exactly this case: forget the cached entry, next get() reloads
    fresh K/D + H together so they can't disagree.
    """

    def test_invalidate_forces_reload_of_new_intrinsics(self):
        registry = HomographyRegistry()
        old_row = _row(
            frame_source="captured_undistorted",
            calibration_matrix=K,
            distortion_coefficients=D,
        )
        new_K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        new_row = _row(
            frame_source="captured_undistorted",
            calibration_matrix=new_K,
            distortion_coefficients=D,
        )

        with patch(
            "domain.calibration.homography_registry.HomographyRepository"
        ) as mock_repo_cls:
            mock_repo_cls.return_value.get_for_camera.return_value = old_row
            proj1 = registry.get("test-slug", 7)
            self.assertEqual(proj1.K.tolist(), K)

            # Simulate: operator hits Recalibrate + Save. No HomographyCalibrated
            # fires (H is untouched) — only CameraCalibrationSaved does.
            mock_repo_cls.return_value.get_for_camera.return_value = new_row

            # Without invalidation the stale cached entry would still be served.
            proj_stale = registry.get("test-slug", 7)
            self.assertEqual(proj_stale.K.tolist(), K)

            registry.invalidate("test-slug", 7)
            proj2 = registry.get("test-slug", 7)
            self.assertEqual(proj2.K.tolist(), new_K)

    def test_invalidate_is_noop_for_unknown_camera(self):
        registry = HomographyRegistry()
        registry.invalidate("test-slug", 999)  # must not raise


if __name__ == "__main__":
    unittest.main()
