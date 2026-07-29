"""Undistortion in the P = K convention.

P = K is the committed convention: undistorted output stays in the original
camera matrix's scale and origin (as opposed to a re-estimated "new camera
matrix" like `estimateNewCameraMatrixForUndistortRectify`). The image path
(undistort_image) and the point path (undistort_points) MUST use the same P
or the homography — fit in one space, applied in the other — silently drifts
by tens of pixels. Do not change P; every stored homography is fit in this
space.
"""

from typing import List, Union

import cv2
import numpy as np


def _prepare_intrinsics(camera_matrix, dist_coeffs, model: str):
    K = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
    is_fisheye = model == "fisheye"
    D = np.array(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    if is_fisheye:
        D = D[:4].reshape(4, 1) if D.shape[0] >= 4 else np.vstack(
            [D, np.zeros((4 - D.shape[0], 1))]
        )
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
