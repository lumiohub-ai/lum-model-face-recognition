"""
MDA Publisher

Publishes messages to Backend via Redis Pub/Sub (Pure MDA).
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

from face_recognition.mda.redis_client import get_redis_client

logger = logging.getLogger(__name__)

# Channels
CHANNELS = {
    'ATTENDANCE_RECORDS': 'attendance.records',
    'UNRECOGNIZED_FACES': 'unrecognized.faces',
    'ACTIVITY_RECORDS': 'activity.records',
    'USER_LOCATIONS': 'user.locations',
    'EMBEDDING_RESULTS': 'embedding.results',
}


class MDAPublisher:
    """
    Publisher for sending messages to Backend via Redis Pub/Sub.
    """

    def __init__(self, client_slug: str, gcs_uploader=None):
        """
        Initialize the MDA publisher.

        Args:
            client_slug: Organization identifier
            gcs_uploader: Optional GCS uploader for proof images
        """
        self.client_slug = client_slug
        self.gcs_uploader = gcs_uploader
        self.redis = get_redis_client()

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
            logger.warning("[MDA] No GCS uploader configured, skipping image upload")
            return None

        try:
            url = await self.gcs_uploader.upload_image(image, prefix)
            return url
        except Exception as e:
            logger.error(f"[MDA] Failed to upload image to GCS: {e}")
            return None

    def publish_attendance_record(
        self,
        user_id: int,
        status: str,
        camera_id: Optional[int] = None,
        proof_image_url: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Publish attendance record to Backend.

        Args:
            user_id: User ID
            status: 'in' or 'out'
            camera_id: Camera ID
            proof_image_url: URL of proof image in GCS
            timestamp: ISO timestamp (defaults to now)

        Returns:
            True if published successfully
        """
        message = {
            'message_id': self._generate_message_id(),
            'client_slug': self.client_slug,
            'user_id': user_id,
            'status': status.lower(),
            'camera_id': camera_id,
            'proof_image_url': proof_image_url,
            'timestamp': timestamp or self._get_timestamp(),
        }

        logger.info(f"[MDA] Publishing attendance: user {user_id} {status}")
        return self.redis.publish(CHANNELS['ATTENDANCE_RECORDS'], message)

    def publish_unrecognized_face(
        self,
        camera_id: Optional[int] = None,
        status: str = 'unknown',
        image_url: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Publish unrecognized face to Backend.

        Args:
            camera_id: Camera ID
            status: User status
            image_url: URL of face image in GCS
            notes: Optional notes

        Returns:
            True if published successfully
        """
        message = {
            'message_id': self._generate_message_id(),
            'client_slug': self.client_slug,
            'camera_id': camera_id,
            'status': status.lower(),
            'image_url': image_url,
            'detection_time': self._get_timestamp(),
            'notes': notes,
        }

        logger.info(f"[MDA] Publishing unrecognized face from camera {camera_id}")
        return self.redis.publish(CHANNELS['UNRECOGNIZED_FACES'], message)

    def publish_activity_record(
        self,
        user_id: Optional[int] = None,
        activity_type: str = 'unknown',
        camera_id: Optional[int] = None,
        confidence_score: Optional[float] = None,
        proof_image_url: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Publish activity record to Backend.

        Args:
            user_id: User ID (if identified)
            activity_type: Type of activity (phone_usage, sleeping, etc.)
            camera_id: Camera ID
            confidence_score: AI confidence score
            proof_image_url: URL of proof image in GCS
            timestamp: ISO timestamp

        Returns:
            True if published successfully
        """
        message = {
            'message_id': self._generate_message_id(),
            'client_slug': self.client_slug,
            'user_id': user_id,
            'activity_type': activity_type,
            'camera_id': camera_id,
            'confidence_score': confidence_score,
            'proof_image_url': proof_image_url,
            'timestamp': timestamp or self._get_timestamp(),
        }

        logger.info(f"[MDA] Publishing activity: user {user_id} - {activity_type}")
        return self.redis.publish(CHANNELS['ACTIVITY_RECORDS'], message)

    def publish_user_location(
        self,
        user_name: str,
        camera_name: str,
        status: str = 'in',
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Publish user location to Backend.

        Args:
            user_name: User name
            camera_name: Camera name/location
            status: 'in' or 'out'
            timestamp: ISO timestamp

        Returns:
            True if published successfully
        """
        message = {
            'message_id': self._generate_message_id(),
            'client_slug': self.client_slug,
            'user_name': user_name,
            'camera_name': camera_name,
            'status': status.lower(),
            'timestamp': timestamp or self._get_timestamp(),
        }

        logger.info(f"[MDA] Publishing user location: {user_name} at {camera_name}")
        return self.redis.publish(CHANNELS['USER_LOCATIONS'], message)

    def publish_embedding_result(
        self,
        request_id: str,
        status: str,
        action: str,
        user_id: int,
        embeddings_created: int = 0,
        error: Optional[str] = None
    ) -> bool:
        """
        Publish embedding processing result to Backend.

        Args:
            request_id: Original request message ID
            status: 'success' or 'failure'
            action: 'add_user', 'update_user', or 'delete_user'
            user_id: User ID
            embeddings_created: Number of embeddings created
            error: Error message if failed

        Returns:
            True if published successfully
        """
        message = {
            'message_id': self._generate_message_id(),
            'request_id': request_id,
            'status': status,
            'action': action,
            'client_slug': self.client_slug,
            'user_id': user_id,
            'embeddings_created': embeddings_created,
            'error': error,
            'timestamp': self._get_timestamp(),
        }

        logger.info(f"[MDA] Publishing embedding result: {action} for user {user_id} - {status}")
        return self.redis.publish(CHANNELS['EMBEDDING_RESULTS'], message)
