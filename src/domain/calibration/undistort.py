"""Undistortion in the P = K convention, shared by CaptureFrame and
TestCalibration so the two operator-facing previews cannot diverge.

`undistort_points` has no runtime caller yet: the pipeline fits and applies
homographies in raw pixel space. Picking src_pts off an undistorted preview
therefore drifts by tens of pixels. The runtime side is implemented at 2993c6b
on `bysh-human-tracking` and still needs porting; keep both functions on the
same P so it lands cleanly.
"""

from typing import List, Union

import cv2
import numpy as np


_PINHOLE_D_LENGTHS = (4, 5, 8, 12, 14)
_FISHEYE_D_LENGTHS = (4,)


def _prepare_intrinsics(camera_matrix, dist_coeffs, model: str):
    """Validate and shape intrinsics. Never coerces: distortion coefficients are
    positional, so fitting one model's vector to another's length silently
    reinterprets its values. main.py defaults the model to 'fisheye', which puts
    a standard 5-coefficient calibration one step from being read as k1..k4.
    Callers fall back to the raw frame on error, which is the safe outcome."""
    K = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
    is_fisheye = model == "fisheye"

    if dist_coeffs is None:
        # np.array(None) is [nan], which survives to an all-zero remap: a black
        # frame published as a successful undistort.
        raise ValueError(f"dist_coeffs is required for model={model!r}, got None")

    D = np.array(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    allowed = _FISHEYE_D_LENGTHS if is_fisheye else _PINHOLE_D_LENGTHS
    if D.shape[0] not in allowed:
        raise ValueError(
            f"dist_coeffs for model={model!r} must have "
            f"{' or '.join(map(str, allowed))} coefficients, "
            f"got {D.shape[0]}"
        )
    if not np.all(np.isfinite(D)):
        raise ValueError(
            f"dist_coeffs for model={model!r} contains non-finite values: "
            f"{D.ravel().tolist()}"
        )
    if not np.all(np.isfinite(K)):
        raise ValueError(f"camera_matrix contains non-finite values: {K.tolist()}")

    return K, D, is_fisheye


def undistort_image(
    image: np.ndarray,
    camera_matrix: Union[List[List[float]], np.ndarray],
    dist_coeffs: Union[List[float], np.ndarray],
    model: str = "fisheye",
) -> np.ndarray:
    """Undistort *image* into the P = K space."""
    K, D, is_fisheye = _prepare_intrinsics(camera_matrix, dist_coeffs, model)
    h, w = image.shape[:2]

    if is_fisheye:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
        )
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    return cv2.undistort(image, K, D, None, K)


def undistort_points(
    points: np.ndarray,
    camera_matrix: Union[List[List[float]], np.ndarray],
    dist_coeffs: Union[List[float], np.ndarray],
    model: str = "fisheye",
) -> np.ndarray:
    """Undistort *points* (shape (N, 1, 2)) into the P = K space.

    Returns an array of the same shape, in pixel coordinates (not normalized).
    """
    K, D, is_fisheye = _prepare_intrinsics(camera_matrix, dist_coeffs, model)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)

    if is_fisheye:
        return cv2.fisheye.undistortPoints(pts, K, D, R=np.eye(3), P=K)

    return cv2.undistortPoints(pts, K, D, R=np.eye(3), P=K)
