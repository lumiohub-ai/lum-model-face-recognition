"""Undistortion in the P = K convention.

P = K means undistorted output stays in the original camera matrix's scale and
origin, rather than a re-estimated "new camera matrix" like
`estimateNewCameraMatrixForUndistortRectify`. Both callers use it, so the two
operator-facing previews land in the same space and cannot diverge:

- `CaptureFrame`  -> `engine.py::_do_capture_frame`
- `TestCalibration` -> `CameraCalibrator.undistort`

WHAT THIS DOES NOT YET COVER — read before trusting a homography:

`undistort_points` has **no runtime caller**. The live pipeline never
undistorts: `engine.py::_do_compute_homography` fits H from the raw src_pts the
backend sends, and `camera_engine.py::emit_positions` applies H directly to raw
bbox coordinates. Runtime projection was deliberately left out when this module
landed (see commit c4fd426).

So the hazard is real and currently unguarded: if an operator picks src_pts off
the *undistorted* CaptureFrame preview, H is fit in P = K space and then applied
to raw distorted coordinates — drifting by tens of pixels. Wiring the runtime
side (registry-cached intrinsics + a `frame_source` column recording which space
each H was fit in) is implemented at commit 2993c6b on `bysh-human-tracking` and
still needs porting here.

Keep undistort_image and undistort_points on the same P: they are the two halves
of that future fix, and a mismatch between them would reintroduce the drift at
the point it is finally wired up.
"""

from typing import List, Union

import cv2
import numpy as np


# cv2.undistort/undistortPoints accept only these distortion-vector lengths.
# Anything else raises a bare cv2.error deep inside OpenCV; we fail earlier with
# a message that names the actual length.
_PINHOLE_D_LENGTHS = (4, 5, 8, 12, 14)


def _prepare_intrinsics(camera_matrix, dist_coeffs, model: str):
    K = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
    is_fisheye = model == "fisheye"
    D = np.array(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    if is_fisheye:
        D = D[:4].reshape(4, 1) if D.shape[0] >= 4 else np.vstack(
            [D, np.zeros((4 - D.shape[0], 1))]
        )
    elif D.shape[0] not in _PINHOLE_D_LENGTHS:
        # The fisheye branch above pads/truncates to 4; the pinhole path has no
        # equivalent safe coercion (the coefficients are positional), so reject
        # rather than guess. Callers fall back to the raw frame.
        raise ValueError(
            f"dist_coeffs for model={model!r} must have "
            f"{' or '.join(map(str, _PINHOLE_D_LENGTHS))} coefficients, "
            f"got {D.shape[0]}"
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
