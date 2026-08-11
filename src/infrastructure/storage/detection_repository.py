"""Write repository for AI-owned detection records.

Centralizes all INSERT/UPSERT operations for attendance, unrecognized faces,
activities, and user locations into one place, keeping Celery tasks free of SQL.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from loguru import logger

from .db_config import DatabaseConfig
from .validators import schema_name_for


class DetectionRepository:
    """Write-side repository for detection records.

    Handles all INSERT/UPSERT operations on tables owned by the AI service:
    - attendance_records
    - unrecognized_faces
    - activity_records
    - user_locations
    """

    def __init__(self, client_slug: str):
        self.schema = schema_name_for(client_slug)
        self._db = DatabaseConfig.get_instance()

    def record_attendance(
        self,
        user_id: int,
        camera_id: Optional[int],
        timestamp: datetime,
        status: str,
        proof_image_url: Optional[str] = None,
    ) -> int:
        """Insert an attendance record.

        Returns:
            New record ID
        """
        external_id = str(uuid.uuid4())
        with self._db.get_connection() as conn:
            result = conn.execute(text(f"""
                INSERT INTO {self.schema}.attendance_records
                (external_id, user_id, camera_id, timestamp, status, proof_image_url, source)
                VALUES (:external_id, :user_id, :camera_id, :timestamp, :status, :proof_image_url, :source)
                RETURNING id
            """), {
                'external_id': external_id,
                'user_id': user_id,
                'camera_id': camera_id,
                'timestamp': timestamp,
                'status': status.lower(),
                'proof_image_url': proof_image_url,
                'source': 'ai_detection',
            })
            conn.commit()
            row = result.fetchone()
            return row[0] if row else 0

    def save_unrecognized_face(
        self,
        camera_id: Optional[int],
        camera_name: Optional[str],
        timestamp: datetime,
        status: Optional[str],
        image_url: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Insert an unrecognized face record.

        Returns:
            New record ID
        """
        full_notes = notes or ''
        if camera_name:
            full_notes = f"Camera: {camera_name}" + (f"\n{notes}" if notes else '')

        with self._db.get_connection() as conn:
            result = conn.execute(text(f"""
                INSERT INTO {self.schema}.unrecognized_faces
                (detection_time, status, user_status, image_url, notes, created_at, updated_at)
                VALUES (:detection_time, :status, :user_status, :image_url, :notes, :created_at, :updated_at)
                RETURNING id
            """), {
                'detection_time': timestamp,
                'status': 'pending',
                'user_status': status.lower() if status else None,
                'image_url': image_url,
                'notes': full_notes or None,
                'created_at': timestamp,
                'updated_at': timestamp,
            })
            conn.commit()
            row = result.fetchone()
            return row[0] if row else 0

    def record_activity(
        self,
        user_id: int,
        camera_id: Optional[int],
        activity_type: str,
        timestamp: datetime,
        proof_image_url: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Insert an activity record and upsert the user's current-activity
        row, matching FastAPI's `activity/service.py::async_record` (which
        this call bypasses — the AI service writes directly to the DB).

        Returns:
            New record ID
        """
        external_id = str(uuid.uuid4())
        with self._db.get_connection() as conn:
            result = conn.execute(text(f"""
                INSERT INTO {self.schema}.activity_records
                (external_id, user_id, camera_id, activity_type, timestamp, proof_image_url, confidence_score, created_at)
                VALUES (:external_id, :user_id, :camera_id, :activity_type, :timestamp, :proof_image_url, :confidence_score, :created_at)
                RETURNING id
            """), {
                'external_id': external_id,
                'user_id': user_id,
                'camera_id': camera_id,
                'activity_type': activity_type,
                'timestamp': timestamp,
                'proof_image_url': proof_image_url,
                'confidence_score': confidence,
                'created_at': timestamp,
            })
            row = result.fetchone()
            record_id = row[0] if row else 0

            conn.execute(text(f"""
                INSERT INTO {self.schema}.user_current_activities
                (user_id, camera_id, activity_type, detected_at, confidence_score, updated_at)
                VALUES (:user_id, :camera_id, :activity_type, :detected_at, :confidence_score, :updated_at)
                ON CONFLICT (user_id) DO UPDATE SET
                    camera_id = EXCLUDED.camera_id,
                    activity_type = EXCLUDED.activity_type,
                    detected_at = EXCLUDED.detected_at,
                    confidence_score = EXCLUDED.confidence_score,
                    updated_at = EXCLUDED.updated_at
            """), {
                'user_id': user_id,
                'camera_id': camera_id,
                'activity_type': activity_type,
                'detected_at': timestamp,
                'confidence_score': confidence,
                'updated_at': timestamp,
            })

            conn.commit()
            return record_id

    def save_calibration_frame(
        self,
        camera_id: int,
        frame_url: str,
        frame_index: int,
        captured_at: datetime,
    ) -> int:
        """Insert a calibration frame record.

        Looks up organization_id from public.organizations using the schema
        name so the AI service doesn't need to carry the integer ID around.

        Returns:
            New record ID, or 0 on failure
        """
        now = datetime.utcnow()
        try:
            with self._db.get_connection() as conn:
                # Derive client_slug from schema name (schema = "org_{slug}")
                slug = self.schema[len("org_"):]
                org_row = conn.execute(
                    text("SELECT id FROM public.organizations WHERE slug = :slug"),
                    {'slug': slug},
                ).fetchone()
                organization_id = org_row[0] if org_row else None

                result = conn.execute(text(f"""
                    INSERT INTO {self.schema}.calibration_frames
                    (camera_id, organization_id, frame_url, frame_index, captured_at, created_at, updated_at)
                    VALUES (:camera_id, :organization_id, :frame_url, :frame_index, :captured_at, :created_at, :updated_at)
                    RETURNING id
                """), {
                    'camera_id': camera_id,
                    'organization_id': organization_id,
                    'frame_url': frame_url,
                    'frame_index': frame_index,
                    'captured_at': captured_at,
                    'created_at': now,
                    'updated_at': now,
                })
                conn.commit()
                row = result.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.exception(f"Failed to save calibration frame: {e}")
            return 0

    def get_calibration_frames(self, camera_id: int) -> list:
        """Fetch all calibration frame URLs for a camera, ordered by frame_index.

        Returns:
            List of dicts with keys: id, frame_url, frame_index, captured_at
        """
        try:
            with self._db.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT id, frame_url, frame_index, captured_at
                    FROM {self.schema}.calibration_frames
                    WHERE camera_id = :camera_id
                    ORDER BY frame_index ASC
                """), {'camera_id': camera_id})
                rows = result.fetchall()
                return [
                    {
                        'id': row[0],
                        'frame_url': row[1],
                        'frame_index': row[2],
                        'captured_at': row[3].isoformat() if row[3] else None,
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.exception(f"Failed to get calibration frames: {e}")
            return []

    def update_user_location(
        self,
        user_id: Optional[int],
        camera_id: int,
        timestamp: datetime,
        status: str,
    ) -> int:
        """Upsert a user location record (one row per user).

        Returns:
            Record ID
        """
        with self._db.get_connection() as conn:
            result = conn.execute(text(f"""
                INSERT INTO {self.schema}.user_locations
                (user_id, camera_id, detected_at, status, updated_at)
                VALUES (:user_id, :camera_id, :detected_at, :status, :updated_at)
                ON CONFLICT (user_id) DO UPDATE SET
                    camera_id = EXCLUDED.camera_id,
                    detected_at = EXCLUDED.detected_at,
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
                RETURNING id
            """), {
                'user_id': user_id,
                'camera_id': camera_id,
                'detected_at': timestamp,
                'status': status.lower(),
                'updated_at': timestamp,
            })
            conn.commit()
            row = result.fetchone()
            return row[0] if row else 0
