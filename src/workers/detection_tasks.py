"""
Celery Tasks for Detection Processing (Attendance, Activity, Unrecognized Faces)

These tasks are called by the camera engine and processed by Celery workers.
They write data to AI's PostgreSQL database, then publish events to Pub/Sub for UI.

Flow:
1. Camera engine detects face/activity
2. Camera engine queues Celery task (.delay())
3. Celery Worker processes task:
   - Writes record to AI's PostgreSQL database
   - Publishes event to Backend via Redis Pub/Sub
4. Backend receives event and notifies frontend via Socket.IO

Data Ownership:
- AI owns: attendance_records, unrecognized_faces, activities, user_locations (in AI's DB)
- Events are ephemeral notifications for real-time UI updates
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
from celery import shared_task
from loguru import logger


def get_db_connection():
    """Get database connection for writing AI-owned data."""
    from infrastructure.storage.db_config import DatabaseConfig
    return DatabaseConfig()


def get_event_publisher(client_slug: str):
    """Get event publisher for sending UI notifications to Backend."""
    from messaging.publisher import MDAPublisher
    return MDAPublisher(client_slug)


def get_schema_name(client_slug: str) -> str:
    """Get the schema name for a client."""
    from infrastructure.storage.validators import validate_client_slug
    validated_slug = validate_client_slug(client_slug)
    return f"org_{validated_slug}"


# =============================================================================
# ATTENDANCE TASK
# =============================================================================

@shared_task(bind=True, name='detection.record_attendance', queue='detections', max_retries=3)
def task_record_attendance(
    self,
    client_slug: str,
    user_id: int,
    user_name: str,
    status: str,
    camera_id: Optional[int] = None,
    camera_name: Optional[str] = None,
    proof_image_url: Optional[str] = None,
    recorded_at: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record attendance in AI's database and publish event.

    Args:
        client_slug: Organization slug
        user_id: User ID
        user_name: User's full name
        status: 'in' or 'out'
        camera_id: Camera ID
        camera_name: Camera name/location
        proof_image_url: GCS URL of proof image
        recorded_at: ISO timestamp of detection

    Returns:
        Result dict with record_id
    """
    from sqlalchemy import text

    logger.info(f"[Celery] Recording attendance: {user_name} (id={user_id}) {status}")

    try:
        db_config = get_db_connection()
        schema_name = get_schema_name(client_slug)


        # Parse timestamp
        if recorded_at:
            timestamp = datetime.fromisoformat(recorded_at.replace('Z', '+00:00'))
        else:
            timestamp = datetime.utcnow()

        # Generate external_id for the record
        import uuid
        external_id = str(uuid.uuid4())

        # Write to Backend's attendance_records table
        # Schema: id, external_id, user_id, camera_id, timestamp, status, created_at, proof_image_url, source, added_by
        with db_config.get_connection() as conn:
            result = conn.execute(text(f"""
                INSERT INTO {schema_name}.attendance_records
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
                'source': 'ai_detection'
            })
            conn.commit()

            row = result.fetchone()
            record_id = row[0] if row else 0

        logger.info(f"[Celery] Attendance recorded: id={record_id} user={user_id} {status}")

        # Publish event to Backend for real-time UI update
        publisher = get_event_publisher(client_slug)
        publisher.publish_attendance_recorded(
            record_id=record_id,
            user_id=user_id,
            user_name=user_name,
            status=status,
            camera_id=camera_id,
            camera_name=camera_name,
            proof_image_url=proof_image_url,
            recorded_at=timestamp.isoformat() + 'Z'
        )

        return {
            'status': 'success',
            'record_id': record_id,
            'user_id': user_id,
            'attendance_status': status
        }

    except Exception as e:
        logger.error(f"[Celery] Attendance recording failed for {user_name}: {e}")
        raise self.retry(exc=e, countdown=5)


# =============================================================================
# UNRECOGNIZED FACE TASK
# =============================================================================

@shared_task(bind=True, name='detection.save_unrecognized', queue='detections', max_retries=3)
def task_save_unrecognized_face(
    self,
    client_slug: str,
    camera_id: Optional[int] = None,
    camera_name: Optional[str] = None,
    status: Optional[str] = None,
    image_url: Optional[str] = None,
    notes: Optional[str] = None,
    detected_at: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save unrecognized face in database and publish event.

    Uses Backend's UnrecognizedFace table schema:
    - detection_time: when detected
    - status: 'pending' (for workflow)
    - user_status: 'in' or 'out' (direction)
    - image_url: GCS URL of face image

    Args:
        client_slug: Organization slug
        camera_id: Camera ID
        camera_name: Camera name/location (stored in notes)
        status: 'in' or 'out' (direction)
        image_url: GCS URL of face image
        notes: Optional notes
        detected_at: ISO timestamp of detection

    Returns:
        Result dict with record_id
    """
    from sqlalchemy import text

    logger.info(f"[Celery] Saving unrecognized face from camera {camera_id}")

    try:
        db_config = get_db_connection()
        schema_name = get_schema_name(client_slug)

        # Parse timestamp
        if detected_at:
            timestamp = datetime.fromisoformat(detected_at.replace('Z', '+00:00'))
        else:
            timestamp = datetime.utcnow()

        # Combine camera_name into notes if provided
        full_notes = notes or ''
        if camera_name:
            full_notes = f"Camera: {camera_name}" + (f"\n{notes}" if notes else '')

        # Write to database (matches Backend's UnrecognizedFace model)
        # Note: Backend's table doesn't have camera_id, so we store camera info in notes
        with db_config.get_connection() as conn:
            result = conn.execute(text(f"""
                INSERT INTO {schema_name}.unrecognized_faces
                (detection_time, status, user_status, image_url, notes, created_at, updated_at)
                VALUES (:detection_time, :status, :user_status, :image_url, :notes, :created_at, :updated_at)
                RETURNING id
            """), {
                'detection_time': timestamp,
                'status': 'pending',  # Backend's workflow status
                'user_status': status.lower() if status else None,  # in/out direction
                'image_url': image_url,
                'notes': full_notes if full_notes else None,
                'created_at': timestamp,
                'updated_at': timestamp
            })
            conn.commit()

            row = result.fetchone()
            record_id = row[0] if row else 0

        logger.info(f"[Celery] Unrecognized face saved: id={record_id} camera={camera_id}")

        # Publish event to Backend for real-time UI update
        publisher = get_event_publisher(client_slug)
        publisher.publish_unrecognized_face_saved(
            record_id=record_id,
            camera_id=camera_id,
            camera_name=camera_name,
            image_url=image_url,
            detected_at=timestamp.isoformat() + 'Z'
        )

        return {
            'status': 'success',
            'record_id': record_id,
            'camera_id': camera_id
        }

    except Exception as e:
        logger.error(f"[Celery] Unrecognized face save failed: {e}")
        raise self.retry(exc=e, countdown=5)


# =============================================================================
# ACTIVITY TASK
# =============================================================================

@shared_task(bind=True, name='detection.record_activity', queue='detections', max_retries=3)
def task_record_activity(
    self,
    client_slug: str,
    user_id: int,
    user_name: str,
    activity_type: str,
    camera_id: Optional[int] = None,
    confidence: Optional[float] = None,
    proof_image_url: Optional[str] = None,
    detected_at: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record activity detection in AI's database and publish event.

    Args:
        client_slug: Organization slug
        user_id: User ID
        user_name: User's full name
        activity_type: Type of activity (phone_usage, sleeping, etc.)
        camera_id: Camera ID
        confidence: AI confidence score
        proof_image_url: GCS URL of proof image
        detected_at: ISO timestamp of detection

    Returns:
        Result dict with record_id
    """
    from sqlalchemy import text

    logger.info(f"[Celery] Recording activity: {user_name} - {activity_type}")

    try:
        db_config = get_db_connection()
        schema_name = get_schema_name(client_slug)

        # Parse timestamp
        if detected_at:
            timestamp = datetime.fromisoformat(detected_at.replace('Z', '+00:00'))
        else:
            timestamp = datetime.utcnow()

        # Generate external_id for the record
        import uuid
        external_id = str(uuid.uuid4())

        # Write to Backend's activity_records table
        with db_config.get_connection() as conn:
            result = conn.execute(text(f"""
                INSERT INTO {schema_name}.activity_records
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
                'created_at': timestamp
            })
            conn.commit()

            row = result.fetchone()
            record_id = row[0] if row else 0

        logger.info(f"[Celery] Activity recorded: id={record_id} user={user_id} - {activity_type}")

        # Publish event to Backend for real-time UI update
        publisher = get_event_publisher(client_slug)
        publisher.publish_activity_detected(
            record_id=record_id,
            user_id=user_id,
            user_name=user_name,
            activity_type=activity_type,
            camera_id=camera_id,
            confidence=confidence,
            proof_image_url=proof_image_url,
            detected_at=timestamp.isoformat() + 'Z'
        )

        return {
            'status': 'success',
            'record_id': record_id,
            'user_id': user_id,
            'activity_type': activity_type
        }

    except Exception as e:
        logger.error(f"[Celery] Activity recording failed for {user_name}: {e}")
        raise self.retry(exc=e, countdown=5)


# =============================================================================
# USER LOCATION TASK
# =============================================================================

@shared_task(bind=True, name='detection.update_user_location', queue='detections', max_retries=3)
def task_update_user_location(
    self,
    client_slug: str,
    user_id: Optional[int],
    user_name: str,
    camera_id: int,
    camera_name: str,
    status: str,
    updated_at: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update user location in AI's database and publish event.

    Args:
        client_slug: Organization slug
        user_id: User ID (optional)
        user_name: User's full name
        camera_id: Camera ID
        camera_name: Camera name/location
        status: 'in' or 'out'
        updated_at: ISO timestamp

    Returns:
        Result dict with record_id
    """
    from sqlalchemy import text

    logger.info(f"[Celery] Updating user location: {user_name} at {camera_name}")

    try:
        db_config = get_db_connection()
        schema_name = get_schema_name(client_slug)

        # Parse timestamp
        if updated_at:
            timestamp = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        else:
            timestamp = datetime.utcnow()

        # Write to Backend's user_locations table (upsert - one row per user)
        # Schema: id, user_id, camera_id, detected_at, status, updated_at
        with db_config.get_connection() as conn:
            result = conn.execute(text(f"""
                INSERT INTO {schema_name}.user_locations
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
                'updated_at': timestamp
            })
            conn.commit()

            row = result.fetchone()
            record_id = row[0] if row else 0

        logger.info(f"[Celery] User location updated: id={record_id} user={user_id} camera={camera_id}")

        # Publish event to Backend for real-time UI update
        publisher = get_event_publisher(client_slug)
        publisher.publish_user_location_updated(
            user_id=user_id or 0,
            user_name=user_name,
            camera_id=camera_id,
            camera_name=camera_name,
            status=status,
            updated_at=timestamp.isoformat() + 'Z'
        )

        return {
            'status': 'success',
            'record_id': record_id,
            'user_name': user_name,
            'camera_name': camera_name
        }

    except Exception as e:
        logger.error(f"[Celery] User location update failed for {user_name}: {e}")
        raise self.retry(exc=e, countdown=5)
