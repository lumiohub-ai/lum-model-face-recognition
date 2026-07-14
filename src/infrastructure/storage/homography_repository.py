"""Read homography matrices and lens calibration from the shared backend DB.

Each camera has at most one row in `org_<slug>.camera_map_positions`. The row
holds the 3x3 image→floor matrix, the `map_id` it belongs to, and
`frame_source` — which space the homography was fit in (`captured_raw` vs
`captured_undistorted`). Joined against `org_<slug>.cameras` for the lens
calibration (`calibration_matrix`, `distortion_coefficients`,
`calibration_model`, `calibration_status`), so the runtime position-streaming
pipeline can decide whether to undistort a tracked point before applying `H`.
"""

import json
from dataclasses import dataclass
from typing import Any, List, Optional

from loguru import logger
from sqlalchemy import text

from .db_config import DatabaseConfig
from .validators import schema_name_for, validate_client_slug


def _parse_json_field(value: Any) -> Optional[Any]:
    if value is None:
        return None
    return json.loads(value) if isinstance(value, str) else value


@dataclass
class CameraProjectionRow:
    """Raw DB row: homography + the lens calibration needed to use it."""

    homography_matrix: List[List[float]]
    map_id: int
    frame_source: Optional[str]
    calibration_status: Optional[str]
    calibration_matrix: Optional[List[List[float]]]
    distortion_coefficients: Optional[List[float]]
    calibration_model: Optional[str]


class HomographyRepository:
    """Single-camera homography + lens calibration lookup against the org schema."""

    def __init__(self, client_slug: str):
        self.client_slug = validate_client_slug(client_slug)
        self.schema = schema_name_for(self.client_slug)
        self._db = DatabaseConfig.get_instance()

    def get_for_camera(self, camera_id: int) -> Optional[CameraProjectionRow]:
        """Return the homography + lens calibration row for *camera_id*, or None.

        Returns None when no row exists OR the row's homography_matrix is NULL.
        """
        try:
            with self._db.get_connection() as conn:
                result = conn.execute(
                    text(
                        f'SELECT p.homography_matrix, p.map_id, p.frame_source, '
                        f'c.calibration_status, c.calibration_matrix, '
                        f'c.distortion_coefficients, c.calibration_model '
                        f'FROM "{self.schema}".camera_map_positions p '
                        f'JOIN "{self.schema}".cameras c ON c.id = p.camera_id '
                        f'WHERE p.camera_id = :camera_id '
                        f'AND p.homography_matrix IS NOT NULL '
                        f'LIMIT 1'
                    ),
                    {"camera_id": camera_id},
                )
                row = result.fetchone()
                if row is None:
                    return None
                return CameraProjectionRow(
                    homography_matrix=_parse_json_field(row[0]),
                    map_id=int(row[1]),
                    frame_source=row[2],
                    calibration_status=row[3],
                    calibration_matrix=_parse_json_field(row[4]),
                    distortion_coefficients=_parse_json_field(row[5]),
                    calibration_model=row[6],
                )
        except Exception as e:
            logger.exception(
                f"Failed to fetch homography for camera {camera_id} "
                f"in schema {self.schema}: {e}"
            )
            return None
