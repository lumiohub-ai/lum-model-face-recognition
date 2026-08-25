"""Charuco-based camera calibrator supporting fisheye and standard lens models."""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
from loguru import logger


class CameraCalibrator:
    """Calibrates cameras using a Charuco board (fisheye or standard model).

    Board spec: 5 cols × 7 rows, square=0.04 m, marker=0.02 m, DICT_4X4_50.
    Supports both old OpenCV API (< 4.7) and new API (>= 4.7).
    """

    COLS = 5
    ROWS = 7
    SQUARE_LENGTH = 0.04  # metres
    MARKER_LENGTH = 0.02  # metres

    def __init__(self, fisheye: bool = True):
        self.fisheye = fisheye

        cv_ver = tuple(int(x) for x in cv2.__version__.split(".")[:2])
        self._new_api = cv_ver >= (4, 7)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        if self._new_api:
            self._board = cv2.aruco.CharucoBoard(
                (self.COLS, self.ROWS),
                self.SQUARE_LENGTH,
                self.MARKER_LENGTH,
                aruco_dict,
            )
            self._detector = cv2.aruco.CharucoDetector(self._board)
        else:
            self._board = cv2.aruco.CharucoBoard_create(
                self.COLS,
                self.ROWS,
                self.SQUARE_LENGTH,
                self.MARKER_LENGTH,
                aruco_dict,
            )
            self._detector_params = cv2.aruco.DetectorParameters_create()
            self._aruco_dict = aruco_dict

        logger.info(
            f"CameraCalibrator ready: {self.COLS}x{self.ROWS} Charuco, "
            f"fisheye={fisheye}, new_api={self._new_api}"
        )

    # ── Corner detection ──────────────────────────────────────────────────────

    def detect_corners(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect Charuco corners in *image*.

        Returns:
            (charuco_corners, charuco_ids) or (None, None) if fewer than 6
            corners were detected.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        try:
            if self._new_api:
                charuco_corners, charuco_ids, _, _ = self._detector.detectBoard(gray)
            else:
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                    gray, self._aruco_dict, parameters=self._detector_params
                )
                if marker_ids is None or len(marker_ids) == 0:
                    return None, None
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self._board
                )
        except Exception as e:
            logger.debug(f"Corner detection error: {e}")
            return None, None

        if charuco_ids is None or len(charuco_ids) < 6:
            return None, None

        return charuco_corners, charuco_ids

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Run calibration on a list of frames.

        Returns a dict with keys:
          success, rms_error, camera_matrix, dist_coeffs, img_size,
          frames_used, model  — or  success=False, error, frames_used.
        """
        all_corners: List[np.ndarray] = []
        all_ids: List[np.ndarray] = []

        for i, frame in enumerate(frames):
            corners, ids = self.detect_corners(frame)
            if corners is not None:
                all_corners.append(corners)
                all_ids.append(ids)
            else:
                logger.debug(f"Frame {i + 1}: no Charuco corners detected, skipping")

        frames_used = len(all_corners)
        logger.info(f"Charuco corners found in {frames_used}/{len(frames)} frames")

        if frames_used < 10:
            return {
                "success": False,
                "error": f"Only {frames_used} frames had detectable Charuco corners (need at least 10)",
                "frames_used": frames_used,
            }

        h, w = frames[0].shape[:2]
        img_size = (w, h)

        if self.fisheye:
            return self._calibrate_fisheye(all_corners, all_ids, img_size, frames_used)
        return self._calibrate_standard(all_corners, all_ids, img_size, frames_used)

    # ── Standard calibration ──────────────────────────────────────────────────

    def _calibrate_standard(
        self,
        all_corners: List[np.ndarray],
        all_ids: List[np.ndarray],
        img_size: Tuple[int, int],
        frames_used: int,
    ) -> Dict[str, Any]:
        try:
            if self._new_api:
                obj_pts_list, img_pts_list = [], []
                for corners, ids in zip(all_corners, all_ids):
                    obj_pts, img_pts = self._board.matchImagePoints(corners, ids)
                    if obj_pts is not None and len(obj_pts) >= 4:
                        obj_pts_list.append(obj_pts)
                        img_pts_list.append(img_pts)
                if len(obj_pts_list) < 4:
                    return {
                        "success": False,
                        "error": "Not enough point matches for standard calibration",
                        "frames_used": frames_used,
                    }
                rms, K, D, _, _ = cv2.calibrateCamera(
                    obj_pts_list, img_pts_list, img_size, None, None
                )
            else:
                rms, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
                    all_corners, all_ids, self._board, img_size, None, None
                )

            logger.info(f"Standard calibration complete: RMS={rms:.4f}")
            return {
                "success": True,
                "rms_error": float(rms),
                "camera_matrix": K.tolist(),
                "dist_coeffs": D.tolist(),
                "img_size": list(img_size),
                "frames_used": frames_used,
                "model": "standard",
            }
        except Exception as e:
            logger.exception(f"Standard calibration failed: {e}")
            return {"success": False, "error": str(e), "frames_used": frames_used}

    # ── Fisheye calibration ───────────────────────────────────────────────────

    def _calibrate_fisheye(
        self,
        all_corners: List[np.ndarray],
        all_ids: List[np.ndarray],
        img_size: Tuple[int, int],
        frames_used: int,
    ) -> Dict[str, Any]:
        try:
            obj_pts_list, img_pts_list = [], []

            for corners, ids in zip(all_corners, all_ids):
                if hasattr(self._board, "matchImagePoints"):
                    obj_pts, img_pts = self._board.matchImagePoints(corners, ids)
                else:
                    # Older OpenCV: manually gather object points
                    board_corners = self._board.chessboardCorners
                    o, p = [], []
                    for i, idx in enumerate(ids.flatten()):
                        if idx < len(board_corners):
                            o.append(board_corners[idx])
                            p.append(corners[i])
                    if len(o) < 4:
                        continue
                    obj_pts = np.array(o, dtype=np.float32)
                    img_pts = np.array(p, dtype=np.float32)

                if obj_pts is None or len(obj_pts) < 4:
                    continue

                obj_pts_list.append(obj_pts.reshape(-1, 1, 3).astype(np.float64))
                img_pts_list.append(img_pts.reshape(-1, 1, 2).astype(np.float64))

            if len(obj_pts_list) < 4:
                logger.warning("Not enough point sets for fisheye; falling back to standard")
                return self._calibrate_standard(all_corners, all_ids, img_size, frames_used)

            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                + cv2.fisheye.CALIB_CHECK_COND
                + cv2.fisheye.CALIB_FIX_SKEW
            )
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                obj_pts_list, img_pts_list, img_size, K, D, flags=flags
            )

            logger.info(f"Fisheye calibration complete: RMS={rms:.4f}")
            return {
                "success": True,
                "rms_error": float(rms),
                "camera_matrix": K.tolist(),
                "dist_coeffs": D.tolist(),
                "img_size": list(img_size),
                "frames_used": frames_used,
                "model": "fisheye",
            }
        except Exception as e:
            logger.warning(f"Fisheye calibration failed ({e}); falling back to standard")
            return self._calibrate_standard(all_corners, all_ids, img_size, frames_used)

    # ── Undistortion ──────────────────────────────────────────────────────────

    def undistort(
        self,
        image: np.ndarray,
        camera_matrix,
        dist_coeffs,
        model: str = "fisheye",
    ) -> np.ndarray:
        """Undistort *image* using the provided calibration parameters.

        Delegates to the shared P = K helper so this always matches the
        runtime point-undistortion path — see domain.calibration.undistort.
        """
        from .undistort import undistort_image

        return undistort_image(image, camera_matrix, dist_coeffs, model)
