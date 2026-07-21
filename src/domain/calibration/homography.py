"""Floor homography solver: maps camera image pixels to floor-plan pixels.

Backend supplies matched src_pts (image) and dst_pts (floor-plan); this module
solves for the 3x3 H such that H · src ≈ dst and reports reprojection error.
"""

from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


def compute_homography(
    src_pts: Sequence[Sequence[float]],
    dst_pts: Sequence[Sequence[float]],
    ransac_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Solve for a 3x3 homography mapping src_pts -> dst_pts.

    Uses plain DLT for exactly 4 point pairs (exact solution) and RANSAC for 5+
    pairs to reject outliers. Returns the matrix, mean reprojection error,
    per-point errors, the method used, and (for RANSAC) an inlier mask.

    Raises ValueError on too-few or degenerate (e.g. collinear) point sets.
    """
    src = np.asarray(src_pts, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 1, 2)
    n = src.shape[0]

    if dst.shape[0] != n:
        raise ValueError(
            f"src_pts and dst_pts length mismatch: {n} vs {dst.shape[0]}"
        )
    if n < 4:
        raise ValueError(f"Need >=4 matching point pairs (got {n})")

    if n == 4:
        H, _ = cv2.findHomography(src, dst, method=0)
        method = "DLT"
        mask: np.ndarray | None = None
    else:
        H, mask = cv2.findHomography(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
        )
        method = "RANSAC"

    if H is None:
        raise ValueError(
            "findHomography returned None — points are likely degenerate (collinear or coincident)"
        )

    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst.reshape(-1, 2), axis=1)

    inlier_mask: List[int] | None = (
        mask.flatten().astype(int).tolist() if mask is not None else None
    )

    return {
        "homography_matrix": H.tolist(),
        "reprojection_error": float(errors.mean()),
        "per_point_errors": errors.tolist(),
        "method": method,
        "inlier_mask": inlier_mask,
    }
