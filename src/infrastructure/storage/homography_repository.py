"""Read homography matrices from the shared backend DB.

Each camera has at most one row in `org_<slug>.camera_map_positions`. The row
holds the 3x3 image→floor matrix plus the `map_id` it belongs to. Used by the
runtime position-streaming pipeline to project bbox foot points to floor coords.
"""

import json
from typing import List, Optional, Tuple

from sqlalchemy import text

from .db_config import DatabaseConfig
from .validators import schema_name_for, validate_client_slug


class HomographyRepository:
    """Single-camera homography lookup against the org schema."""

    def __init__(self, client_slug: str):
        self.client_slug = validate_client_slug(client_slug)
        self.schema = schema_name_for(self.client_slug)
        self._db = DatabaseConfig.get_instance()

    def get_for_camera(
        self, camera_id: int
    ) -> Optional[Tuple[List[List[float]], int]]:
        """Return (matrix_3x3, map_id) for the given camera, or None if not calibrated.

        Returns None ONLY when no row exists OR the row's homography_matrix is
        NULL — a genuine "not calibrated" state that HomographyRegistry caches
        as a (long-lived) negative result. Any other failure (DB connection
        error, malformed map_id, etc.) raises instead of returning None, so a
        transient DB error can't get permanently cached as "not calibrated"
        and silently disable position streaming for a camera that's actually
        fine.
        """
        with self._db.get_connection() as conn:
            result = conn.execute(
                text(
                    f'SELECT homography_matrix, map_id '
                    f'FROM "{self.schema}".camera_map_positions '
                    f'WHERE camera_id = :camera_id '
                    f'AND homography_matrix IS NOT NULL '
                    f'LIMIT 1'
                ),
                {"camera_id": camera_id},
            )
            row = result.fetchone()
            if row is None:
                return None
            matrix_raw, map_id = row[0], row[1]
            matrix = (
                json.loads(matrix_raw)
                if isinstance(matrix_raw, str)
                else matrix_raw
            )
            if map_id is None:
                # Data-integrity bug (valid matrix, no map_id) — surface it
                # rather than masking it as "not calibrated".
                raise ValueError(
                    f"camera_map_positions row for camera {camera_id} in "
                    f"schema {self.schema} has a homography_matrix but a "
                    f"NULL map_id"
                )
            return matrix, int(map_id)
