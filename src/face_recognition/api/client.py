
"""API client for SmartOffice backend integration."""

import io
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

    @property
    def session(self) -> requests.Session:
        """Get the authenticated session."""
        return self.auth.get_session()

    def get_org_unique_id(self) -> Optional[str]:
        """Fetch the unique_id of the organization matching this client's slug.

        Returns:
            The organization's unique_id if found, None otherwise
        """
        if not self.auth.is_authenticated():
            logger.error("Not authenticated")
            return None

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        url = f"{self.base_url}/organizations"

        try:
            response = self.session.get(url, headers=headers, timeout=10)
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

        response = self.session.get(
            f"{self.base_url}/org/{self.client_slug}/users",
            params={"page": page, "limit": limit, "status": "active"}
        )

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
                user.get("id"): user.get("image_path")
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
                    new_users.append({
                        'name': user['name'],
                        'image_path': next(
                            (item['path'] for item in id_to_path if item['id'] == user['id']),
                            None
                        )
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

        headers = {"Authorization": f"Bearer {self.token}"}
        url = f"{self.base_url}/org/{self.client_slug}/users/{status.lower()}"

        try:
            response = self.session.get(url, headers=headers)
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
        camera_id: Optional[int] = None
    ) -> Optional[requests.Response]:
        """Create an attendance record.

        Args:
            user_id: ID of the user to create record for
            status: Either 'IN' or 'OUT'
            camera_id: ID of the camera that detected the person

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

        record_data = {
            "user_id": user_id,
            "status": status.lower(),
            "camera_id": camera_id,
            "timestamp": timestamp
        }

        try:
            response = self.session.post(
                f"{self.base_url}/org/{self.client_slug}/attendance-records",
                json=record_data,
                headers={"Content-Type": "application/json"}
            )

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
        headers = {'Authorization': f'Bearer {self.token}'}

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        data = {'detection_time': timestamp}

        if face is None or face.size == 0:
            logger.warning("No face detected to send")
            return None

        success, encoded_image = cv2.imencode('.jpg', face)
        if not success:
            logger.error("Image encoding failed")
            return None

        # Convert to byte stream
        image_bytes = io.BytesIO(encoded_image.tobytes())

        # Prepare file payload
        files = [
            ('images', ('cropped_face.jpg', image_bytes, 'image/jpeg')),
        ]

        try:
            response = requests.post(url, headers=headers, files=files, data=data)

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
                        (e.g., 'FaceRecognision')

        Returns:
            List of camera configuration dictionaries
        """
        if not self.auth.is_authenticated():
            logger.error("Not authenticated")
            return []

        url = f"{self.base_url}/org/{self.client_slug}/cameras"

        try:
            response = self.session.get(url)
            response.raise_for_status()

            cameras = response.json()

            # Filter by application if specified
            if application:
                cameras = [
                    cam for cam in cameras
                    if cam.get('application') == application
                ]

            return cameras

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch cameras: {str(e)}")
            return []

    def get_face_recognition_camera_configs(self) -> Dict[str, List[Any]]:
        """Fetch and parse camera configurations for face recognition.

        Retrieves cameras with application='FaceRecognision' and parses them
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
            ValueError: If no cameras found with application='FaceRecognision'
        """
        # Fetch cameras with application='FaceRecognision'
        cameras = self.get_cameras(application='FaceRecognision')

        if not cameras:
            raise ValueError("No cameras found with application='FaceRecognision'")

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
        headers = {'Authorization': f'Bearer {self.token}'}

        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            logger.error("Image encoding failed")
            return None

        image_bytes = io.BytesIO(encoded_image.tobytes())

        files = [
            ('images', ('annotated_frame.jpg', image_bytes, 'image/jpeg')),
        ]

        try:
            response = requests.post(url, headers=headers, files=files)

            if response.status_code not in [200, 201]:
                return None

            return response

        except requests.exceptions.RequestException as e:
            logger.debug(f"API upload request failed: {str(e)}")
            return None
