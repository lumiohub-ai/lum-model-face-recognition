
"""API client for SmartOffice backend integration."""

import base64
import io
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from loguru import logger

from .auth import AuthenticationService


class APIClient:
    """Client for communicating with the SmartOffice backend API.

    This class handles all API operations including:
    - User management (fetching, status checking)
    - Attendance record creation
    - Unrecognized face submission
    - Frame upload for dashboard
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

        # Circuit breaker state for location API
        self._location_circuit_breaker_failures = 0
        self._location_circuit_breaker_open_until = 0
        self._location_circuit_breaker_threshold = 5
        self._location_circuit_breaker_timeout = 60  # seconds

    @property
    def session(self) -> requests.Session:
        """Get the authenticated session."""
        return self.auth.get_session()

    def _handle_token_expiry(self, response: requests.Response, retry_callback) -> requests.Response:
        """Handle token expiration by refreshing and retrying the request.

        Args:
            response: The response that potentially has a 401 error
            retry_callback: Function to retry the request after token refresh

        Returns:
            The response from the retry, or the original response if refresh fails
        """
        if response.status_code == 401:
            logger.warning("Received 401 Unauthorized - token may have expired, attempting refresh...")

            # Refresh the token
            if self.auth.refresh_token():
                # Update the token reference
                self.token = self.auth.get_token()

                # Retry the request
                logger.info("Retrying request with refreshed token...")
                return retry_callback()
            else:
                logger.error("Token refresh failed, cannot retry request")

        return response

    def get_users(self) -> List[Dict[str, Any]]:
        """Retrieve all users from the API.

        Returns:
            List of user dictionaries with 'name' and 'id' fields
        """
        if not self.auth.is_authenticated():
            logger.error("Not authenticated")
            return []

        url = f"{self.base_url}/org/{self.client_slug}/users"

        def make_request():
            return self.session.get(
                url,
                params={"status": "active"}
            )

        try:
            response = make_request()
            response = self._handle_token_expiry(response, make_request)

            if response.status_code == 200:
                users = response.json()
                return [
                    {
                        'name': user.get('full_name'),
                        'id': user.get('id')
                    }
                    for user in users
                    if user.get('full_name') and user.get('id')
                ]
            elif response.status_code == 404:
                logger.warning("No users found in the database")
                return []
            else:
                logger.error(
                    f"Error fetching users: {response.status_code} - {response.text}"
                )
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
        """Retrieve all users from the API and determine new and deleted users.

        Args:
            current_users: List of currently known user names
            page: Page number for pagination
            limit: Number of results per page

        Returns:
            Tuple of (new_users, deleted_users, name_to_id_mappings):
            - new_users: List of dictionaries containing new user information
            - deleted_users: List of user names that have been deleted
            - name_to_id_mappings: List of dictionaries mapping names to IDs
        """
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
            # Use 'image_url' field from backend API (not 'image_path')
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

                    # Skip users without image URLs
                    if not image_path:
                        logger.warning(
                            f"Skipping user '{user['name']}' - no image_url provided by backend API"
                        )
                        continue

                    new_users.append({
                        'name': user['name'],
                        'image_path': image_path
                    })

            for user in current_users:
                if user not in user_dict.keys():
                    deleted_users.append(user)

            return new_users, deleted_users, name_to_id

        elif response.status_code == 404:
            logger.warning("No users found in the database")
            return [], [], []

        else:
            logger.critical(
                f"Error fetching users: {response.status_code} - {response.text}"
            )
            raise Exception(
                f"Error fetching users: {response.status_code} - {response.text}"
            )

    def get_users_by_status(self, status: str) -> List[str]:
        """Get users by their current status (IN/OUT).

        Args:
            status: User status ("in" or "out")

        Returns:
            List of user full names with the specified status
        """
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
                logger.error(
                    f"Failed to fetch {status} users with status "
                    f"{response.status_code}: {response.text}"
                )
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
        proof_image: Optional[np.ndarray] = None
    ) -> Optional[requests.Response]:
        """Create an attendance record.

        Args:
            user_id: ID of the user to create record for
            status: Either 'IN' or 'OUT'
            camera_id: ID of the camera that detected the person
            proof_image: Optional annotated frame with person bbox as proof

        Returns:
            Response object if successful, None otherwise
        """
        if not self.auth.is_authenticated():
            logger.critical("Not authenticated. Please login first.")
            return None

        status = status.upper()
        if status not in ['IN', 'OUT']:
            logger.warning(
                f"Invalid status '{status}'. Status must be either 'IN' or 'OUT'"
            )
            return None

        # Generate timestamp in ISO 8601 format with milliseconds
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        url = f"{self.base_url}/org/{self.client_slug}/attendance-records"

        # Prepare form data
        data = {
            "user_id": str(user_id),
            "status": status.lower(),
            "timestamp": timestamp,
            "source": "auto"
        }

        # Encode proof_image if provided
        encoded_image = None
        if proof_image is not None and proof_image.size > 0:
            success, encoded = cv2.imencode('.jpg', proof_image)
            if success:
                encoded_image = encoded
            else:
                logger.warning("Failed to encode proof_image, sending record without image")

        def make_request():
            if encoded_image is not None:
                # Send as multipart/form-data with image file
                headers = {"Authorization": f"Bearer {self.token}"}
                files = [
                    ('proof_image', ('proof.jpg', io.BytesIO(encoded_image.tobytes()), 'image/jpeg')),
                ]
                return self.session.post(
                    f"{self.base_url}/org/{self.client_slug}/attendance-records",
                    data=record_data,
                    files=files,
                    headers=headers
                )
            else:
                # Send as JSON without image
                return self.session.post(
                    f"{self.base_url}/org/{self.client_slug}/attendance-records",
                    json=record_data,
                    headers={"Content-Type": "application/json"}
                )

        try:
            response = make_request()
            response = self._handle_token_expiry(response, make_request)

            if response.status_code not in [200, 201]:
                logger.error(
                    f"Record creation failed with status {response.status_code}: "
                    f"{response.text}"
                )
                return None

            return response

        except requests.exceptions.RequestException as e:
            logger.error(f"Record creation request failed: {str(e)}")
            return None

    def send_unrecognized_face(
        self,
        face: np.ndarray,
        status: str,
        camera_id: Optional[int] = None,
        notes: Optional[str] = None
    ) -> Optional[requests.Response]:
        """Send unrecognized face image to the API.

        Args:
            face: Detected face image (numpy array)
            status: Status of the user ('IN' or 'OUT')
            camera_id: Optional camera ID that detected the face
            notes: Optional notes about the detection

        Returns:
            Response object if successful, None otherwise
        """
        if not self.auth.is_authenticated():
            logger.critical("Not authenticated. Please login first.")
            return None

        status = status.upper()
        if status not in ['IN', 'OUT']:
            logger.warning(
                f"Invalid status '{status}'. Status must be either 'IN' or 'OUT'"
            )
            return None

        url = self.base_url + f'/org/{self.client_slug}/unrecognized-faces'

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        data = {
            'detection_time': timestamp,
            'user_status': status.lower()
        }

        # Add optional fields if provided
        if camera_id is not None:
            data['camera_id'] = camera_id
        if notes:
            data['notes'] = notes

        if face is None or face.size == 0:
            logger.warning("No face detected to send")
            return None

        success, encoded_image = cv2.imencode('.jpg', face)
        if not success:
            logger.error("Image encoding failed")
            return None

        def make_request():
            headers = {'Authorization': f'Bearer {self.token}'}
            # Prepare file payload - create fresh BytesIO for each retry
            files = [
                ('images', ('cropped_face.jpg', io.BytesIO(encoded_image.tobytes()), 'image/jpeg')),
            ]
            return requests.post(url, headers=headers, files=files, data=data)

        try:
            response = make_request()
            response = self._handle_token_expiry(response, make_request)

            if response.status_code not in [200, 201]:
                logger.error(
                    f"Send unrecognized face failed with status {response.status_code}: "
                    f"{response.text}"
                )
                return None

            return response

        except requests.exceptions.RequestException as e:
            logger.error(f"Send unrecognized face request failed: {str(e)}")
            return None

    def send_activities(
        self,
        activity_type: str,
        camera_id: Optional[int] = None,
        user_id: Optional[str] = None,
        confidence_score: Optional[float] = None,
        proof_image: Optional[np.ndarray]=None,

    ) -> Optional[requests.Response]:
        """Send activity to the API.

        Args:
            face: Detected face image (numpy array)
            status: Status of the user ('IN' or 'OUT')
            camera_id: Optional camera ID that detected the face
            notes: Optional notes about the detection

        Returns:
            Response object if successful, None otherwise
        """
        if not self.auth.is_authenticated():
            logger.critical("Not authenticated. Please login first.")
            return None

        activity_type = activity_type.lower()
        if activity_type not in ['phone_usage', 'sleeping', 'not_focusing','talking', 'working','unknown']:
            logger.warning(
                f"Invalid status '{activity_type}'. Activity type must be in range of 'phone_usage', 'sleeping', 'not_focusing','talking', 'working' or 'unknown'"
            )
            return None

        url = self.base_url + f'/org/{self.client_slug}/activities'

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        data = {
            'user_id': user_id,
            'activity_type': activity_type,
            'timestamp': timestamp,
            'metadata': '{}'
        }

        # Add optional fields if provided
        if confidence_score is not None:
            data['confidence_score'] = confidence_score
        if camera_id is not None:
            data['camera_id'] = camera_id

        # Log the exact payload being sent
        logger.info(f"API Request Details: activity_type='{activity_type}' (len={len(activity_type)}), user_id={user_id} (type={type(user_id).__name__}), camera_id={camera_id}")
        logger.debug(f"Full data payload: {data}")

        image = True
        if proof_image is None or proof_image.size == 0:
            image = False
        if image:
            success, encoded_image = cv2.imencode('.jpg', proof_image)
            if not success:
                logger.error("Image encoding failed")
                return None

        def make_request():
            headers = {'Authorization': f'Bearer {self.token}'}
            # Prepare file payload - create fresh BytesIO for each retry
            # Always send as multipart/form-data (matching API expectation)
            if image:
                files = [
                    ('proof_image', ('proof.jpg', io.BytesIO(encoded_image.tobytes()), 'image/jpeg')),
                ]
            else:
                # Send empty file to maintain multipart/form-data format
                files = [
                    ('proof_image', ('', io.BytesIO(b''), 'application/octet-stream')),
                ]

            logger.debug(f"POST {url} with data keys: {list(data.keys())}, has_image: {image}")
            return requests.post(url, headers=headers, files=files, data=data)

        try:
            response = make_request()
            response = self._handle_token_expiry(response, make_request)

            if response.status_code not in [200, 201]:
                logger.error(
                    f"Send {activity_type} failed with status {response.status_code}: "
                    f"{response.text}"
                )
                return None

            return response

        except requests.exceptions.RequestException as e:
            logger.error(f"Send {activity_type} request failed: {str(e)}")
            return None

    def get_user_action(
        self,
        image: np.ndarray,
        vlm_api_url: str = "http://localhost:8001",
        timeout: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Get user action from VLM API.

        Args:
            image: Person crop image (numpy array, BGR format)
            vlm_api_url: URL of VLM API service
            timeout: Request timeout in seconds

        Returns:
            Dictionary with action result, or None if failed
            Example: {"action": "using phone", "raw_output": "...", "inference_time_ms": 245}
        """
        try:
            # Encode image to base64
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                logger.error("Failed to encode image for VLM API")
                return None

            image_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

            # Prepare request
            payload = {"image": image_b64}
            url = f"{vlm_api_url}/api/recognize-action"

            # Send request (no authentication needed for VLM API)
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code not in [200, 201]:
                logger.error(
                    f"VLM API request failed with status {response.status_code}: "
                    f"{response.text}"
                )
                return None

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"VLM API request failed: {str(e)}")
            return None

    def get_cameras(self, application: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cameras from the API, optionally filtered by application.

        Args:
            application: Optional filter for camera application type
                        (e.g., 'attendance', 'unrecognized', 'activity')
                        Note: Camera application field is a JSONB array

        Returns:
            List of camera configuration dictionaries
        """
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

            # Filter by application if specified
            # Application field is a JSONB array: ["attendance", "unrecognized", "activity"]
            if application:
                filtered_cameras = []
                for cam in cameras:
                    cam_apps = cam.get('application', [])
                    # Handle both array and legacy string format
                    if isinstance(cam_apps, list):
                        if application in cam_apps:
                            filtered_cameras.append(cam)
                    elif isinstance(cam_apps, str):
                        # Legacy format or single string
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
    ) -> Optional[requests.Response]:
        """Send user location data to the API with retry logic and circuit breaker.

        Args:
            user_name: Full name of the user
            camera_name: Name of the camera that detected the user
            timestamp: ISO 8601 formatted timestamp
            status: User status ('IN' or 'OUT')

        Returns:
            Response object if successful, None otherwise
        """
        if not self.auth.is_authenticated():
            logger.error("Not authenticated. Cannot send location data.")
            return None

        if status.upper() not in ['IN', 'OUT']:
            logger.warning(
                f"Invalid status '{status}'. Status must be either 'IN' or 'OUT'"
            )
            return None

        # Check circuit breaker
        if self._is_location_circuit_breaker_open():
            logger.debug(
                f"Location API circuit breaker is open, skipping location update for {user_name}"
            )
            return None

        url = f"{self.base_url}/org/{self.client_slug}/user-locations"

        data = {
            "user_name": user_name,
            "camera_name": camera_name,
            "timestamp": timestamp,
            "status": status.lower()
        }

        def make_request():
            headers = {"Authorization": f"Bearer {self.token}"}
            return self.session.post(url, headers=headers, json=data, timeout=10)

        # Use retry with exponential backoff
        response = self._retry_with_backoff(make_request, max_retries=3)

        if response and response.status_code in [200, 201]:
            self._record_location_success()
            return response
        else:
            self._record_location_failure()
            if response:
                logger.warning(
                    f"Failed to send location data for {user_name} "
                    f"with status {response.status_code}"
                )
            else:
                logger.warning(
                    f"Failed to send location data for {user_name}"
                )
            return None

    def upload_annotated_frame(
        self,
        frame: np.ndarray,
        camera_id: int,
        camera_type: str
    ) -> Optional[requests.Response]:
        """Upload annotated frame to the API.

        Args:
            frame: Annotated frame to send
            camera_id: Camera ID
            camera_type: Camera type (IN/OUT)

        Returns:
            Response object if successful, None otherwise
        """
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

