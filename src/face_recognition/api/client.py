"""API client for SmartOffice backend integration (Pure MDA)."""

import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from loguru import logger

from .auth import AuthenticationService

# Lazy import MDA publisher to avoid circular imports
_mda_publisher = None


def get_mda_publisher(client_slug: str):
    """Get or create the MDA publisher instance."""
    global _mda_publisher
    if _mda_publisher is None:
        try:
            from face_recognition.mda.publisher import MDAPublisher
            _mda_publisher = MDAPublisher(client_slug)
            logger.info(f"[MDA] Publisher initialized for {client_slug}")
        except ImportError as e:
            logger.error(f"[MDA] Failed to import MDAPublisher: {e}")
            raise
    return _mda_publisher


class APIClient:
    """Client for communicating with the SmartOffice backend.

    Uses Pure MDA (Redis Pub/Sub) for:
    - Attendance records
    - Unrecognized faces
    - Activity records
    - User locations

    Uses HTTP only for:
    - Authentication (JWT)
    - Fetching users/cameras (read-only operations)
    - VLM API calls (external service)
    """

    def __init__(
        self,
        api_host: str,
        email: str,
        password: str,
        client_slug: str
    ):
        """Initialize the API client.

        Args:
            api_host: Base URL of the API (e.g., "http://localhost:7091/")
            email: User email for authentication
            password: User password for authentication
            client_slug: Organization slug
        """
        self.base_url = api_host.rstrip('/') + '/api'
        self.client_slug = client_slug

        # Initialize authentication service
        self.auth = AuthenticationService(self.base_url, email, password)
        self.token = self.auth.login()

        if not self.token:
            logger.error("Failed to authenticate with API")

        # Initialize MDA publisher
        self._publisher = get_mda_publisher(client_slug)

    @property
    def session(self) -> requests.Session:
        """Get the authenticated session."""
        return self.auth.get_session()

    def _handle_token_expiry(self, response: requests.Response, retry_callback) -> requests.Response:
        """Handle token expiration by refreshing and retrying the request."""
        if response.status_code == 401:
            logger.warning("Received 401 Unauthorized - token may have expired, attempting refresh...")
            if self.auth.refresh_token():
                self.token = self.auth.get_token()
                logger.info("Retrying request with refreshed token...")
                return retry_callback()
            else:
                logger.error("Token refresh failed, cannot retry request")
        return response

    def get_users(self) -> List[Dict[str, Any]]:
        """Retrieve all users from the API."""
        if not self.auth.is_authenticated():
            logger.error("Not authenticated")
            return []

        url = f"{self.base_url}/org/{self.client_slug}/users"

        def make_request():
            return self.session.get(url, params={"status": "active"})

        try:
            response = make_request()
            response = self._handle_token_expiry(response, make_request)

            if response.status_code == 200:
                users = response.json()
                return [
                    {'name': user.get('full_name'), 'id': user.get('id')}
                    for user in users
                    if user.get('full_name') and user.get('id')
                ]
            elif response.status_code == 404:
                logger.warning("No users found in the database")
                return []
            else:
                logger.error(f"Error fetching users: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            return []

    def get_all_users(
        self,
        current_users: List[str],
        page: int = 1,
        limit: int = 50
    ) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """Retrieve all users from the API and determine new and deleted users."""
        if not self.auth.is_authenticated():
            raise ValueError("Not authenticated. Please login first.")

        def make_request():
            return self.session.get(
                f"{self.base_url}/org/{self.client_slug}/users",
                params={"page": page, "limit": limit, "status": "active"}
            )

        response = make_request()
        response = self._handle_token_expiry(response, make_request)

        new_users = []
        deleted_users = []

        if response.status_code == 200:
            users = response.json()

            user_dict = {
                user.get("full_name"): user.get("id")
                for user in users
                if user.get("full_name")
            }
            path_dict = {
                user.get("id"): user.get("image_url")
                for user in users
                if user.get("id")
            }

            name_to_id = [
                {"name": name, "id": user_id}
                for name, user_id in user_dict.items()
            ]
            id_to_path = [
                {"id": user_id, "path": path}
                for user_id, path in path_dict.items()
            ]

            for user in name_to_id:
                if user['name'] not in current_users:
                    image_path = next(
                        (item['path'] for item in id_to_path if item['id'] == user['id']),
                        None
                    )
                    if not image_path:
                        logger.warning(f"Skipping user '{user['name']}' - no image_url provided")
                        continue
                    new_users.append({'name': user['name'], 'image_path': image_path})

            for user in current_users:
                if user not in user_dict.keys():
                    deleted_users.append(user)

            return new_users, deleted_users, name_to_id

        elif response.status_code == 404:
            logger.warning("No users found in the database")
            return [], [], []
        else:
            logger.critical(f"Error fetching users: {response.status_code} - {response.text}")
            raise Exception(f"Error fetching users: {response.status_code} - {response.text}")

    def get_users_by_status(self, status: str) -> List[str]:
        """Get users by their current status (IN/OUT)."""
        if not self.auth.is_authenticated():
            logger.error("Not authenticated")
            return []

        url = f"{self.base_url}/org/{self.client_slug}/users/{status.lower()}"

        def make_request():
            headers = {"Authorization": f"Bearer {self.token}"}
            return self.session.get(url, headers=headers)

        try:
            response = make_request()
            response = self._handle_token_expiry(response, make_request)
            if response.status_code != 200:
                logger.error(f"Failed to fetch {status} users: {response.status_code}: {response.text}")
                return []
            data = response.json()
            return [user["full_name"] for user in data if "full_name" in user]
        except Exception as e:
            logger.error(f"Error fetching {status} users: {e}")
            return []

    def create_attendance_record(
        self,
        user_id: int,
        status: str,
        camera_id: Optional[int] = None,
        proof_image: Optional[np.ndarray] = None,
        proof_image_url: Optional[str] = None
    ) -> bool:
        """Create an attendance record via MDA.

        Args:
            user_id: ID of the user
            status: Either 'IN' or 'OUT'
            camera_id: ID of the camera that detected the person
            proof_image: Image to upload to GCS (if proof_image_url not provided)
            proof_image_url: GCS URL of proof image (if already uploaded)

        Returns:
            True if published successfully
        """
        status = status.upper()
        if status not in ['IN', 'OUT']:
            logger.warning(f"Invalid status '{status}'. Must be 'IN' or 'OUT'")
            return False

        # Upload proof image to GCS if provided and URL not already set
        if proof_image is not None and proof_image_url is None:
            try:
                from .image_uploader import upload_proof_image
                proof_image_url = upload_proof_image(
                    image=proof_image,
                    prefix="attendance_proofs",
                    client_slug=self.client_slug
                )
            except Exception as e:
                logger.warning(f"Failed to upload attendance proof image: {e}")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        logger.info(f"[MDA] Publishing attendance: user {user_id} {status}")
        return self._publisher.publish_attendance_record(
            user_id=user_id,
            status=status,
            camera_id=camera_id,
            proof_image_url=proof_image_url,
            timestamp=timestamp
        )

    def send_unrecognized_face(
        self,
        face: np.ndarray,
        status: str,
        camera_id: Optional[int] = None,
        notes: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> bool:
        """Send unrecognized face via MDA.

        Args:
            face: Unused (kept for API compatibility)
            status: Status of the user ('IN' or 'OUT')
            camera_id: Camera ID that detected the face
            notes: Optional notes
            image_url: GCS URL of face image

        Returns:
            True if published successfully
        """
        status = status.upper()
        if status not in ['IN', 'OUT']:
            logger.warning(f"Invalid status '{status}'. Must be 'IN' or 'OUT'")
            return False

        logger.info(f"[MDA] Publishing unrecognized face from camera {camera_id}")
        return self._publisher.publish_unrecognized_face(
            camera_id=camera_id,
            status=status,
            image_url=image_url,
            notes=notes
        )

    def send_activities(
        self,
        activity_type: str,
        camera_id: Optional[int] = None,
        user_id: Optional[str] = None,
        confidence_score: Optional[float] = None,
        proof_image: Optional[np.ndarray] = None,
        proof_image_url: Optional[str] = None
    ) -> bool:
        """Send activity record via MDA.

        Args:
            activity_type: Type of activity detected
            camera_id: Camera ID
            user_id: User ID
            confidence_score: AI confidence score
            proof_image: Image to upload to GCS (if proof_image_url not provided)
            proof_image_url: GCS URL of proof image (if already uploaded)

        Returns:
            True if published successfully
        """
        activity_type = activity_type.lower()
        valid_types = ['phone_usage', 'sleeping', 'not_focusing', 'talking', 'working', 'unknown']
        if activity_type not in valid_types:
            logger.warning(f"Invalid activity_type '{activity_type}'. Must be one of {valid_types}")
            return False

        # Upload proof image to GCS if provided and URL not already set
        if proof_image is not None and proof_image_url is None:
            try:
                from .image_uploader import upload_proof_image
                proof_image_url = upload_proof_image(
                    image=proof_image,
                    prefix="activity_proofs",
                    client_slug=self.client_slug
                )
            except Exception as e:
                logger.warning(f"Failed to upload activity proof image: {e}")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        logger.info(f"[MDA] Publishing activity: user {user_id} - {activity_type}")
        return self._publisher.publish_activity_record(
            user_id=int(user_id) if user_id else None,
            activity_type=activity_type,
            camera_id=camera_id,
            confidence_score=confidence_score,
            proof_image_url=proof_image_url,
            timestamp=timestamp
        )

    def get_cameras(self, application: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cameras from the API, optionally filtered by application."""
        if not self.auth.is_authenticated():
            logger.error("Not authenticated")
            return []

        url = f"{self.base_url}/org/{self.client_slug}/cameras"

        def make_request():
            return self.session.get(url)

        try:
            response = make_request()
            response = self._handle_token_expiry(response, make_request)
            response.raise_for_status()

            cameras = response.json()

            if application:
                filtered_cameras = []
                for cam in cameras:
                    cam_apps = cam.get('application', [])
                    if isinstance(cam_apps, list):
                        if application in cam_apps:
                            filtered_cameras.append(cam)
                    elif isinstance(cam_apps, str):
                        if cam_apps == application:
                            filtered_cameras.append(cam)
                cameras = filtered_cameras
                logger.info(f"Filtered {len(cameras)} camera(s) with application='{application}'")

            return cameras
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch cameras: {str(e)}")
            return []

    def send_user_location(
        self,
        user_name: str,
        camera_name: str,
        timestamp: str,
        status: str
    ) -> bool:
        """Send user location via MDA.

        Args:
            user_name: Full name of the user
            camera_name: Name of the camera
            timestamp: ISO 8601 formatted timestamp
            status: 'IN' or 'OUT'

        Returns:
            True if published successfully
        """
        if status.upper() not in ['IN', 'OUT']:
            logger.warning(f"Invalid status '{status}'. Must be 'IN' or 'OUT'")
            return False

        logger.info(f"[MDA] Publishing user location: {user_name} at {camera_name}")
        return self._publisher.publish_user_location(
            user_name=user_name,
            camera_name=camera_name,
            status=status,
            timestamp=timestamp
        )

    def upload_annotated_frame(
        self,
        frame: np.ndarray,
        camera_id: int,
        camera_type: str
    ) -> Optional[requests.Response]:
        """Upload annotated frame to the API (still uses HTTP for streaming)."""
        if not self.auth.is_authenticated():
            logger.debug("Not authenticated for API upload, skipping")
            return None

        url = self.base_url + f'/{self.client_slug}/cameras/{camera_id}/{camera_type}'

        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            logger.error("Image encoding failed")
            return None

        def make_request():
            headers = {'Authorization': f'Bearer {self.token}'}
            files = [
                ('images', ('annotated_frame.jpg', io.BytesIO(encoded_image.tobytes()), 'image/jpeg')),
            ]
            return requests.post(url, headers=headers, files=files)

        try:
            response = make_request()
            response = self._handle_token_expiry(response, make_request)

            if response.status_code not in [200, 201]:
                return None
            return response
        except requests.exceptions.RequestException as e:
            logger.debug(f"API upload request failed: {str(e)}")
            return None
