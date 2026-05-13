"""
Celery Tasks for Detection Processing (Attendance, Activity, Unrecognized Faces)

These tasks are called by the camera engine and processed by Celery workers.
They write data to AI's PostgreSQL database, then publish events to Pub/Sub for UI.

Flow:
1. Camera engine detects face/activity
2. Camera engine queues Celery task (.delay())
3. Celery Worker writes record via DetectionRepository, then publishes Redis event
4. Backend receives event and notifies frontend via Socket.IO
"""

from typing import Dict, Any, Optional
from datetime import datetime
from workers.celery_app import celery
from loguru import logger


def _parse_timestamp(iso_str: Optional[str]) -> datetime:
    """Parse ISO timestamp string, defaulting to now if not provided."""
    if iso_str:
        return datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
    return datetime.utcnow()


# =============================================================================
# ATTENDANCE TASK
# =============================================================================

@celery.task(bind=True, name='detection.record_attendance', queue='detections', max_retries=3)
def task_record_attendance(
    self,
    client_slug: str,
    user_id: int,
    user_name: str,
    status: str,
    camera_id: Optional[int] = None,
    camera_name: Optional[str] = None,
    proof_image_url: Optional[str] = None,
    recorded_at: Optional[str] = None,
    gender: Optional[str] = None,
    age: Optional[int] = None,
) -> Dict[str, Any]:
    """Record attendance in AI's database and publish event."""
    logger.info(f"[Celery] Recording attendance: {user_name} (id={user_id}) {status}")

    try:
        from infrastructure.storage.detection_repository import DetectionRepository
        from messaging.publisher import MDAPublisher

        timestamp = _parse_timestamp(recorded_at)
        record_id = DetectionRepository(client_slug).record_attendance(
            user_id=user_id,
            camera_id=camera_id,
            timestamp=timestamp,
            status=status,
            proof_image_url=proof_image_url,
            gender=gender,
            age=age,
        )
        logger.info(f"[Celery] Attendance recorded: id={record_id} user={user_id} {status}")

        MDAPublisher(client_slug).publish_attendance_recorded(
            record_id=record_id,
            user_id=user_id,
            user_name=user_name,
            status=status,
            camera_id=camera_id,
            camera_name=camera_name,
            proof_image_url=proof_image_url,
            recorded_at=timestamp.isoformat() + 'Z'
        )

        return {'status': 'success', 'record_id': record_id, 'user_id': user_id, 'attendance_status': status}

    except Exception as e:
        logger.exception(f"[Celery] Attendance recording failed for {user_name}: {e}")
        raise self.retry(exc=e, countdown=5)


# =============================================================================
# UNRECOGNIZED FACE TASK
# =============================================================================

@celery.task(bind=True, name='detection.save_unrecognized', queue='detections', max_retries=3)
def task_save_unrecognized_face(
    self,
    client_slug: str,
    camera_id: Optional[int] = None,
    camera_name: Optional[str] = None,
    status: Optional[str] = None,
    image_url: Optional[str] = None,
    notes: Optional[str] = None,
    detected_at: Optional[str] = None,
    gender: Optional[str] = None,
    age: Optional[int] = None,
) -> Dict[str, Any]:
    """Save unrecognized face in database and publish event."""
    logger.info(f"[Celery] Saving unrecognized face from camera {camera_id}")

    try:
        from infrastructure.storage.detection_repository import DetectionRepository
        from messaging.publisher import MDAPublisher

        timestamp = _parse_timestamp(detected_at)
        record_id = DetectionRepository(client_slug).save_unrecognized_face(
            camera_id=camera_id,
            camera_name=camera_name,
            timestamp=timestamp,
            status=status,
            image_url=image_url,
            notes=notes,
            gender=gender,
            age=age,
        )
        logger.info(f"[Celery] Unrecognized face saved: id={record_id} camera={camera_id}")

        MDAPublisher(client_slug).publish_unrecognized_face_saved(
            record_id=record_id,
            camera_id=camera_id,
            camera_name=camera_name,
            image_url=image_url,
            detected_at=timestamp.isoformat() + 'Z'
        )

        return {'status': 'success', 'record_id': record_id, 'camera_id': camera_id}

    except Exception as e:
        logger.exception(f"[Celery] Unrecognized face save failed: {e}")
        raise self.retry(exc=e, countdown=5)


# =============================================================================
# ACTIVITY TASK
# =============================================================================

@celery.task(bind=True, name='detection.record_activity', queue='detections', max_retries=3)
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
    """Record activity detection in AI's database and publish event."""
    logger.info(f"[Celery] Recording activity: {user_name} - {activity_type}")

    try:
        from infrastructure.storage.detection_repository import DetectionRepository
        from messaging.publisher import MDAPublisher

        timestamp = _parse_timestamp(detected_at)
        record_id = DetectionRepository(client_slug).record_activity(
            user_id=user_id,
            camera_id=camera_id,
            activity_type=activity_type,
            timestamp=timestamp,
            proof_image_url=proof_image_url,
            confidence=confidence,
        )
        logger.info(f"[Celery] Activity recorded: id={record_id} user={user_id} - {activity_type}")

        MDAPublisher(client_slug).publish_activity_detected(
            record_id=record_id,
            user_id=user_id,
            user_name=user_name,
            activity_type=activity_type,
            camera_id=camera_id,
            confidence=confidence,
            proof_image_url=proof_image_url,
            detected_at=timestamp.isoformat() + 'Z'
        )

        return {'status': 'success', 'record_id': record_id, 'user_id': user_id, 'activity_type': activity_type}

    except Exception as e:
        logger.exception(f"[Celery] Activity recording failed for {user_name}: {e}")
        raise self.retry(exc=e, countdown=5)


# =============================================================================
# USER LOCATION TASK
# =============================================================================

@celery.task(bind=True, name='detection.update_user_location', queue='detections', max_retries=3)
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
    """Update user location in AI's database and publish event."""
    logger.info(f"[Celery] Updating user location: {user_name} at {camera_name}")

    try:
        from infrastructure.storage.detection_repository import DetectionRepository
        from messaging.publisher import MDAPublisher

        timestamp = _parse_timestamp(updated_at)
        record_id = DetectionRepository(client_slug).update_user_location(
            user_id=user_id,
            camera_id=camera_id,
            timestamp=timestamp,
            status=status,
        )
        logger.info(f"[Celery] User location updated: id={record_id} user={user_id} camera={camera_id}")

        MDAPublisher(client_slug).publish_user_location_updated(
            user_id=user_id or 0,
            user_name=user_name,
            camera_id=camera_id,
            camera_name=camera_name,
            status=status,
            updated_at=timestamp.isoformat() + 'Z'
        )

        return {'status': 'success', 'record_id': record_id, 'user_name': user_name, 'camera_name': camera_name}

    except Exception as e:
        logger.exception(f"[Celery] User location update failed for {user_name}: {e}")
        raise self.retry(exc=e, countdown=5)
