"""
Event Publisher (MDA)

Publishes events to Backend via Redis Pub/Sub.
Events are ephemeral notifications - data is already persisted in AI's DB.

Events: AI Service → Backend (Redis Pub/Sub)
Commands: Backend → AI Service (Redis Streams - handled by stream_consumer.py)
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

from messaging.redis_client import RedisClient
from messaging.channels import (
    EVENT_CHANNELS,
    EVENT_TYPES,
)

logger = logging.getLogger(__name__)


class MDAPublisher:
    """Publisher for sending messages to Backend via Redis Pub/Sub."""

    def __init__(self, client_slug: str, gcs_uploader=None):
        """
        Initialize the MDA publisher.

        Args:
            client_slug: Organization identifier
            gcs_uploader: Optional GCS uploader for proof images
        """
        self.client_slug = client_slug
        self.gcs_uploader = gcs_uploader
        self.redis = RedisClient.get_instance()

    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        return str(uuid.uuid4())

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + 'Z'

    async def _upload_image_to_gcs(self, image: np.ndarray, prefix: str) -> Optional[str]:
        """
        Upload image to GCS and return URL.

        Args:
            image: numpy array image
            prefix: GCS path prefix

        Returns:
            GCS URL or None if upload fails
        """
        if self.gcs_uploader is None:
            logger.warning("No GCS uploader configured, skipping image upload")
            return None

        try:
            url = await self.gcs_uploader.upload_image(image, prefix)
            return url
        except Exception as e:
            logger.exception(f"Failed to upload image to GCS: {e}")
            return None

    def publish_attendance_recorded(
        self,
        record_id: int,
        user_id: int,
        user_name: str,
        status: str,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        proof_image_url: Optional[str] = None,
        recorded_at: Optional[str] = None,
        age: Optional[int] = None,
        age_confidence: Optional[float] = None,
        gender: Optional[str] = None,
        gender_confidence: Optional[float] = None
    ) -> bool:
        """
        Publish AttendanceRecorded event to Backend.

        Args:
            record_id: Attendance record ID (from AI's DB)
            user_id: User ID
            user_name: User name
            status: 'in' or 'out'
            camera_id: Camera ID
            camera_name: Camera name
            proof_image_url: URL of proof image in GCS
            recorded_at: ISO timestamp

        Returns:
            True if published successfully
        """
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['ATTENDANCE_RECORDED'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'record_id': record_id,
            'user_id': user_id,
            'user_name': user_name,
            'status': status.lower(),
            'camera_id': camera_id,
            'camera_name': camera_name,
            'proof_image_url': proof_image_url,
            'recorded_at': recorded_at or self._get_timestamp(),
            'age': age,
            'age_confidence': age_confidence,
            'gender': gender,
            'gender_confidence': gender_confidence,
        }

        logger.info(f"[Events] Publishing AttendanceRecorded: user {user_id} {status}")
        return self.redis.publish(EVENT_CHANNELS['ATTENDANCE'], event)

    def publish_unrecognized_face_saved(
        self,
        record_id: int,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        image_url: Optional[str] = None,
        detected_at: Optional[str] = None,
        age: Optional[int] = None,
        age_confidence: Optional[float] = None,
        gender: Optional[str] = None,
        gender_confidence: Optional[float] = None,
        user_status: Optional[str] = None
    ) -> bool:
        """
        Publish UnrecognizedFaceSaved event to Backend.

        Args:
            record_id: Record ID from AI's DB
            camera_id: Camera ID
            camera_name: Camera name
            image_url: URL of face image in GCS
            detected_at: ISO timestamp

        Returns:
            True if published successfully
        """
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['UNRECOGNIZED_FACE_SAVED'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'record_id': record_id,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'image_url': image_url,
            'detected_at': detected_at or self._get_timestamp(),
            'age': age,
            'age_confidence': age_confidence,
            'gender': gender,
            'gender_confidence': gender_confidence,
            'user_status': user_status,
        }

        logger.info(f"[Events] Publishing UnrecognizedFaceSaved from camera {camera_id}")
        result = self.redis.publish(EVENT_CHANNELS['UNRECOGNIZED'], event)
        return result

    def publish_activity_detected(
        self,
        record_id: int,
        user_id: int,
        user_name: str,
        activity_type: str,
        camera_id: Optional[int] = None,
        confidence: Optional[float] = None,
        proof_image_url: Optional[str] = None,
        detected_at: Optional[str] = None
    ) -> bool:
        """
        Publish ActivityDetected event to Backend.

        Args:
            record_id: Record ID from AI's DB
            user_id: User ID
            user_name: User name
            activity_type: Type of activity (phone_usage, sleeping, etc.)
            camera_id: Camera ID
            confidence: AI confidence score
            proof_image_url: URL of proof image in GCS
            detected_at: ISO timestamp

        Returns:
            True if published successfully
        """
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['ACTIVITY_DETECTED'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'record_id': record_id,
            'user_id': user_id,
            'user_name': user_name,
            'activity_type': activity_type,
            'camera_id': camera_id,
            'confidence': confidence,
            'proof_image_url': proof_image_url,
            'detected_at': detected_at or self._get_timestamp(),
        }

        logger.info(f"[Events] Publishing ActivityDetected: user {user_id} - {activity_type}")
        return self.redis.publish(EVENT_CHANNELS['ACTIVITY'], event)

    def publish_user_location_updated(
        self,
        user_id: int,
        user_name: str,
        camera_id: int,
        camera_name: str,
        status: str = 'in',
        updated_at: Optional[str] = None
    ) -> bool:
        """
        Publish UserLocationUpdated event to Backend.

        Args:
            user_id: User ID
            user_name: User name
            camera_id: Camera ID
            camera_name: Camera name/location
            status: 'in' or 'out'
            updated_at: ISO timestamp

        Returns:
            True if published successfully
        """
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['USER_LOCATION_UPDATED'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'user_id': user_id,
            'user_name': user_name,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'status': status.lower(),
            'updated_at': updated_at or self._get_timestamp(),
        }

        logger.info(f"[Events] Publishing UserLocationUpdated: {user_name} at {camera_name}")
        return self.redis.publish(EVENT_CHANNELS['LOCATION'], event)

    def publish_embedding_created(
        self,
        command_id: str,
        user_id: int,
        embeddings_created: int = 0
    ) -> bool:
        """
        Publish EmbeddingCreated event to Backend.

        Args:
            command_id: Original command ID
            user_id: User ID
            embeddings_created: Number of embeddings created

        Returns:
            True if published successfully
        """
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['EMBEDDING_CREATED'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'command_id': command_id,
            'user_id': user_id,
            'embeddings_created': embeddings_created,
        }

        logger.info(f"[Events] Publishing EmbeddingCreated: user {user_id}, count={embeddings_created}")
        return self.redis.publish(EVENT_CHANNELS['EMBEDDING'], event)

    def publish_embedding_failed(
        self,
        command_id: str,
        user_id: int,
        error: str
    ) -> bool:
        """
        Publish EmbeddingFailed event to Backend.

        Args:
            command_id: Original command ID
            user_id: User ID
            error: Error message

        Returns:
            True if published successfully
        """
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['EMBEDDING_FAILED'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'command_id': command_id,
            'user_id': user_id,
            'error': error,
        }

        logger.info(f"[Events] Publishing EmbeddingFailed: user {user_id}, error={error}")
        return self.redis.publish(EVENT_CHANNELS['EMBEDDING'], event)

    def publish_frame_captured(
        self,
        command_id: str,
        camera_id: int,
        image_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Publish FrameCaptured event to Backend.

        Args:
            command_id: Original command ID (resolves the waiting HTTP request)
            camera_id: Camera the frame was captured from
            image_url: GCS URL of the uploaded frame
            metadata: Frame metadata (width, height, size_bytes, source)
        """
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['FRAME_CAPTURED'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'command_id': command_id,
            'camera_id': camera_id,
            'frame_url': image_url,
            'signed_url': image_url,
            'captured_at': self._get_timestamp(),
            'metadata': metadata or {},
        }

        logger.info(f"[Events] Publishing FrameCaptured: camera {camera_id}, command {command_id}")
        return self.redis.publish(EVENT_CHANNELS['FRAME_CAPTURE'], event)

    def publish_calibration_complete(self, command_id, camera_id, rms_error, camera_matrix, dist_coeffs, img_size, frames_used, model):
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['CALIBRATION_COMPLETE'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'command_id': command_id,
            'camera_id': camera_id,
            'rms_error': rms_error,
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'img_size': img_size,
            'frames_used': frames_used,
            'model': model,
        }
        logger.info(f"[Events] Publishing CalibrationComplete: camera {camera_id}, RMS={rms_error:.4f}")
        return self.redis.publish(EVENT_CHANNELS['CALIBRATION'], event)

    def publish_calibration_failed(self, command_id, camera_id, error):
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['CALIBRATION_FAILED'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'command_id': command_id,
            'camera_id': camera_id,
            'error': error,
        }
        logger.info(f"[Events] Publishing CalibrationFailed: camera {camera_id}")
        return self.redis.publish(EVENT_CHANNELS['CALIBRATION'], event)

    def publish_test_calibration_complete(self, command_id, camera_id, image_url):
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['TEST_CALIBRATION_COMPLETE'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'command_id': command_id,
            'camera_id': camera_id,
            'frame_url': image_url,
        }
        logger.info(f"[Events] Publishing TestCalibrationComplete: camera {camera_id}")
        return self.redis.publish(EVENT_CHANNELS['CALIBRATION'], event)

    def publish_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """Publish a periodic system metrics snapshot (CPU/GPU/RAM/FPS)."""
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['SYSTEM_METRICS'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'metrics': metrics,
        }
        self.redis.publish(EVENT_CHANNELS['METRICS'], event)

    def publish_system_alert(self, alert_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Publish a critical system alert (low FPS, GPU OOM risk, high RAM)."""
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['SYSTEM_ALERT'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'alert_type': alert_type,
            'message': message,
            'details': details or {},
        }
        logger.warning(f"[Events] SystemAlert: {alert_type} — {message}")
        self.redis.publish(EVENT_CHANNELS['METRICS'], event)

    def publish_zone_session_entry(
        self,
        camera_id: int,
        line_id: str,
        line_type: str,
        track_id: str,
        global_track_id: str,
        entered_at: Optional[str] = None,
        age: Optional[int] = None,
        age_confidence: Optional[float] = None,
        gender: Optional[str] = None,
        gender_confidence: Optional[float] = None,
        image_url: Optional[str] = None
    ) -> bool:
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['ZONE_SESSION_ENTRY'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'camera_id': camera_id,
            'line_id': line_id,
            'line_type': line_type,
            'track_id': track_id,
            'global_track_id': global_track_id,
            'entered_at': entered_at or self._get_timestamp(),
            'age': age,
            'age_confidence': age_confidence,
            'gender': gender,
            'gender_confidence': gender_confidence,
            'image_url': image_url,
        }
        logger.info(f"[Events] Publishing ZoneSessionEntry: camera {camera_id}, line {line_id}, track {global_track_id}")
        return self.redis.publish(EVENT_CHANNELS['ZONE_SESSION'], event)

    def publish_zone_session_exit(
        self,
        camera_id: int,
        line_id: str,
        global_track_id: str,
        exited_at: Optional[str] = None
    ) -> bool:
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['ZONE_SESSION_EXIT'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'camera_id': camera_id,
            'line_id': line_id,
            'global_track_id': global_track_id,
            'exited_at': exited_at or self._get_timestamp(),
        }
        logger.info(f"[Events] Publishing ZoneSessionExit: camera {camera_id}, line {line_id}, track {global_track_id}")
        return self.redis.publish(EVENT_CHANNELS['ZONE_SESSION'], event)

    def publish_zone_room_session(
        self,
        camera_id: int,
        line_id: str,
        line_type: str,
        started_at: str,
        ended_at: str,
        duration_seconds: int,
    ) -> bool:
        event = {
            'event_id': self._generate_message_id(),
            'event_type': EVENT_TYPES['ZONE_ROOM_SESSION'],
            'timestamp': self._get_timestamp(),
            'client_slug': self.client_slug,
            'camera_id': camera_id,
            'line_id': line_id,
            'line_type': line_type,
            'started_at': started_at,
            'ended_at': ended_at,
            'duration_seconds': duration_seconds,
        }
        logger.info(f"[Events] Publishing ZoneRoomSession: camera {camera_id}, line {line_id}, duration {duration_seconds}s")
        return self.redis.publish(EVENT_CHANNELS['ZONE_SESSION'], event)

