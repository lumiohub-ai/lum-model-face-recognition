
"""API client for SmartOffice backend integration."""

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

    def _retry_with_backoff(
        self,
        request_func,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        retry_status_codes: set = {502, 503, 504}
    ) -> Optional[requests.Response]:
        """Retry a request with exponential backoff.

        Args:
            request_func: Function that makes the request
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            backoff_factor: Multiplier for delay between retries
            retry_status_codes: HTTP status codes that should trigger a retry

        Returns:
            Response object if successful, None otherwise
        """
        delay = initial_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = request_func()

                # Handle token expiry
                response = self._handle_token_expiry(response, request_func)

                # Success
                if response.status_code in [200, 201]:
                    return response

                # Retry on specific error codes
                if response.status_code in retry_status_codes:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Request failed with status {response.status_code}, "
                            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                        continue
                    else:
                        logger.error(
                            f"Request failed with status {response.status_code} "
                            f"after {max_retries} attempts"
                        )
                        return None
                else:
                    # Don't retry on other status codes
                    return response

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Request timed out, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                    continue
                else:
                    logger.error(f"Request timed out after {max_retries} attempts: {e}")
                    return None

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.error(f"Request failed with exception: {e}")
                return None

        return None

    def _is_location_circuit_breaker_open(self) -> bool:
        """Check if the circuit breaker for location API is open.

        Returns:
            True if circuit breaker is open (should not attempt requests), False otherwise
        """
        current_time = time.time()

        # Check if circuit breaker is open
        if current_time < self._location_circuit_breaker_open_until:
            return True

        # Circuit breaker timeout has passed, reset and allow retry
        if self._location_circuit_breaker_open_until > 0:
            logger.info("Location API circuit breaker timeout passed, attempting to reconnect")
            self._location_circuit_breaker_failures = 0
            self._location_circuit_breaker_open_until = 0

        return False

    def _record_location_failure(self):
        """Record a failure for the location API circuit breaker."""
        self._location_circuit_breaker_failures += 1

        if self._location_circuit_breaker_failures >= self._location_circuit_breaker_threshold:
            self._location_circuit_breaker_open_until = (
                time.time() + self._location_circuit_breaker_timeout
            )
            logger.warning(
                f"Location API circuit breaker opened after {self._location_circuit_breaker_failures} "
                f"consecutive failures. Will retry after {self._location_circuit_breaker_timeout}s"
            )

    def _record_location_success(self):
        """Record a success for the location API circuit breaker."""
        if self._location_circuit_breaker_failures > 0:
            logger.info("Location API recovered, resetting circuit breaker")
        self._location_circuit_breaker_failures = 0
        self._location_circuit_breaker_open_until = 0

    def get_org_unique_id(self) -> Optional[str]:
        """Fetch the unique_id of the organization matching this client's slug.

        Returns:
            The organization's unique_id if found, None otherwise
        """
        if not self.auth.is_authenticated():
            logger.error("Not authenticated")
            return None

        url = f"{self.base_url}/organizations"

        def make_request():
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
            }
            return self.session.get(url, headers=headers, timeout=10)

        try:
            response = make_request()
            response = self._handle_token_expiry(response, make_request)
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"[get_org_unique_id] Request failed: {e}")
            return None

        try:
            data = response.json()
        except ValueError:
            logger.warning("[get_org_unique_id] Invalid JSON response")
            return None

        unique_id = next(
            (org["unique_id"] for org in data if org.get("slug") == self.client_slug),
            None
        )

        if not unique_id:
            logger.warning(
                f"[get_org_unique_id] No organization found for slug '{self.client_slug}'"
            )

        return unique_id

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
            proof_image: Optional recognized frame image (numpy array)

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

        if camera_id is not None:
            data["camera_id"] = str(camera_id)

        def make_request():
            headers = {"Authorization": f"Bearer {self.token}"}

            # If proof_image is provided, send as multipart/form-data
            if proof_image is not None and proof_image.size > 0:
                success, encoded_image = cv2.imencode('.jpg', proof_image)
                if not success:
                    logger.error("Proof image encoding failed")
                    return self.session.post(url, data=data, headers=headers)

                files = [
                    ('proof_image', ('proof_image.jpg', io.BytesIO(encoded_image.tobytes()), 'image/jpeg'))
                ]
                return self.session.post(url, data=data, files=files, headers=headers)
            else:
                # Send without image
                return self.session.post(url, data=data, headers=headers)

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
        status: str
    ) -> Optional[requests.Response]:
        """Send unrecognized face image to the API.

        Args:
            face: Detected face image (numpy array)
            status: Status of the user ('IN' or 'OUT')

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
        data = {'detection_time': timestamp}

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

    def get_cameras(self, application: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cameras from the API, optionally filtered by application.

        Args:
            application: Optional filter for camera application type
                        (e.g., 'attendance')

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
            if application:
                cameras = [
                    cam for cam in cameras
                    if application in cam.get('application', [])
                ]

            return cameras

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch cameras: {str(e)}")
            return []

    def get_face_recognition_camera_configs(self) -> Dict[str, List[Any]]:
        """Fetch and parse camera configurations for face recognition.

        Retrieves cameras with 'attendance' in their application list and parses them
        into the format required by HBFace initialization.

        Returns:
            Dictionary containing camera configuration lists with keys:
            - cam_types: List of camera types (IN/OUT)
            - video_path: List of stream URLs
            - camera_name: List of camera names
            - camera_id: List of camera IDs
            - match_threshold: List of matching thresholds
            - roi: List of ROI tuples (or None if no ROIs)
            - line_points: List of virtual line points (or None if no lines)

        Raises:
            ValueError: If no cameras found with 'attendance' in application list
        """
        # Fetch cameras with 'attendance' in application list
        cameras = self.get_cameras(application='attendance')

        if not cameras:
            raise ValueError("No cameras found with 'attendance' in application list")

        # Parse camera configs into HBFace parameters
        cam_types = []
        video_paths = []
        camera_names = []
        camera_ids = []
        match_thresholds = []
        roi_list = []
        line_points_list = []

        for cam in cameras:
            # Map API fields to HBFace parameters
            cam_types.append(cam.get('camera_type', '').upper())
            camera_ids.append(int(cam.get('id')))
            camera_names.append(cam.get('name', ''))
            video_paths.append(cam.get('stream_url', ''))
            match_thresholds.append(float(cam.get('matching_threshold', 0.5)))

            # Handle optional ROI points: [[x1, y1], [x2, y2]] -> (x1, y1, x2, y2)
            roi_points = cam.get('roi_points')
            if roi_points and len(roi_points) >= 2:
                roi_list.append(tuple(roi_points[0] + roi_points[1]))
            else:
                roi_list.append(None)

            # Handle optional virtual line points: [[x1, y1], [x2, y2]]
            virtual_line = cam.get('virtual_line_points')
            if virtual_line and len(virtual_line) >= 2:
                line_points_list.append([tuple(virtual_line[0]), tuple(virtual_line[1])])
            else:
                line_points_list.append(None)

        logger.info(f"Loaded {len(cameras)} camera configurations from API")

        return {
            'cam_types': cam_types,
            'video_path': video_paths,
            'camera_name': camera_names,
            'camera_id': camera_ids,
            'match_threshold': match_thresholds,
            'roi': roi_list if any(roi_list) else None,
            'line_points': line_points_list if any(line_points_list) else None,
        }

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
