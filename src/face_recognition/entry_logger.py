"""Entry logging module for tracking and visualizing person entries and exits."""

from collections import deque
import cv2
import pandas as pd
import os
import io
import json
import requests
from typing import Optional, Dict ,List, Any
from datetime import datetime, timezone

class EntryLogger:
    """Logger for tracking and recording person entries and exits.

    This class handles communication with the API to retrieve user information,
    tracks person status changes, and provides visualization for entry/exit events.
    """
    def __init__(self,
                args,
                max_entries=3):
        """Initialize the entry logger.

        Args:
            args: Configuration arguments
            max_entries: Maximum number of recent entries to display on screen
        """
        self.FR_SLUG = os.getenv("FR_SLUG")
        self.args = args
        self.headers = {"Content-Type": "application/json"}
        self.recent_entries = deque(maxlen=max_entries)
        self.client_slug = args.client_slug
        # Configure a dedicated, rotating sink for the CSV status log
        self.log_file_path = f'/app/volumes/storage/{self.FR_SLUG}/logs/{self.client_slug}/status_info.csv'
        os.makedirs('logs', exist_ok=True)

        # Filter to ensure only CSV-intended messages go to this file
        csv_filter = lambda record: record["extra"].get("is_csv", False)

        # Use the rotation period from config, with a fallback default
        rotation_period = getattr(self.args, 'csv_log_rotation', '2 weeks')

        self.args.logger.add(
            self.log_file_path,
            rotation=rotation_period,
            format="{message}",
            level="INFO",
            filter=csv_filter,
            encoding="utf-8"
        )

        # Write header if the file is new or empty
        if not os.path.exists(self.log_file_path) or os.path.getsize(self.log_file_path) == 0:
            self.args.logger.bind(is_csv=True).info("name,status,time,date")

        self.base_url = args.api_host + 'api'
        self.session = requests.Session()
        self.email = args.email
        self.password = args.password
        self.token = self.login()
        unique_id = self.get_org_unique_id()
        self.current_users = args.db_names
        self.max_track_lifetime_seconds = getattr(args, 'max_track_lifetime_seconds', 120) # Default to 120 seconds

        self.new_users, self.deleted_users, self.name_to_id = self.get_all_users()
        self.person_status = self.get_last_status()


    def get_all_users(self):
        """Retrieve all users from the API and determine new and deleted users.

        Returns:
            Tuple containing lists of new users, deleted users, and name-to-ID mappings
        """
        if not self.token:
            raise ValueError("Not authenticated. Please login first.")
        page = 1
        limit = 50
        response = self.session.get(
                f"{self.base_url}/org/{self.client_slug}/users",
                params={"page": page, "limit": limit, "status": "active"}
            )

        new_users = []
        deleted_users = []

        if response.status_code == 200:
            users = response.json()

            user_dict = {user.get("full_name"): user.get("id") for user in users if user.get("full_name")}
            path_dict = {user.get("id"): user.get("image_path") for user in users if user.get("id")}

            name_to_id = [{"name": name, "id": user_id} for name, user_id in user_dict.items()]
            id_to_path = [{"id": user_id, "path": path} for user_id, path in path_dict.items()]

            for user in name_to_id:
                if user['name'] not in self.current_users:
                    new_users.append(
                        {
                            'name': user['name'],
                            'image_path': next((item['path'] for item in id_to_path if item['id'] == user['id']), None)
                        }
                    )
            for user in self.current_users:
                if user not in user_dict.keys():
                    deleted_users.append(user)

            return new_users, deleted_users, name_to_id

        elif response.status_code == 404:
            self.args.logger.warning("No users found in the database")
            return [], [], []

        else:
            self.args.logger.critical(f"Error fetching users: {response.status_code} - {response.text}")
            raise Exception(f"Error fetching users: {response.status_code} - {response.text}")

    def login(self) -> Dict[str, Any]:
        """
        Authenticate user and obtain access token

        Args:
            password (str): User's EMAIL
            password (str): User's password
            client_slug (str): Client slug for organization

        Returns:
            token (str): Access token for authenticated requests

        Raises:
            requests.exceptions.RequestException: If login fails
        """

        try:
            response = self.session.post(f"{self.base_url}/auth/login",
                                         json={"email": self.email,
                                               "password": self.password})

            response.raise_for_status()

            data = response.json()

            if data.get('success'):
                token = data.get('token')
                # Set authorization header for future requests
                self.session.headers.update({
                    'Authorization': f'Bearer {token}'
                })
                return token
            else:
                raise requests.exceptions.RequestException(f"Login failed: {data.get('error', 'Unknown error')}")

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Login request failed: {str(e)}")

    def get_org_unique_id(self) -> str | None:
        """
        Fetch the unique_id of the organization matching this client's slug.
        Returns
        -------
        str | None
            The organization's unique_id if found, otherwise None.
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        url = f"{self.base_url}/organizations"
        try:
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            # log or re-raise depending on your error handling policy
            self.args.logger.warning(f"[get_org_unique_id] Request failed: {e}")
            return None
        try:
            data = response.json()
        except ValueError:
            self.args.logger.warning("[get_org_unique_id] Invalid JSON response")
            return None
        unique_id = next(
            (org["unique_id"] for org in data if org.get("slug") == self.client_slug),
            None
        )
        if not unique_id:
            self.args.logger.warning(f"[get_org_unique_id] No organization found for slug '{self.client_slug}'")

        return unique_id

    def send_data_to_api(self, name, status):
        """Send person entry/exit data to the API.

        Args:
            name: Name of the person
            status: Entry/exit status (IN/OUT)
        """
        user_id = next((int(i['id']) for i in self.name_to_id if i['name'] == name), None)

        if user_id is None:
            self.args.logger.warning(f'User with ID {user_id} not found in the database')
            return

        response = self.create_record(user_id, status)
        data = response.json()
        if response.status_code == 201:
            self.args.logger.info(f"Success: {data['message']}")
        else:
            self.args.logger.warning(f"Error: {data.get('error', 'Unknown error')}")

    def send_unrecognized_face(self, face, status):
        """Send unrecognized face image to the API.
        Args:
            face: Detected face image (numpy array)
            status: Status of the user ('IN' or 'OUT')
        Raises:
            ValueError: If the status is not 'IN' or 'OUT'
            requests.exceptions.RequestException: If the request to the API fails
        """
         # Check if user is authenticated
         # If not, raise an error
        if not self.token:
            raise ValueError("Not authenticated. Please login first.")
        status = status.upper()
        if status not in ['IN', 'OUT']:
            raise ValueError("Status must be either 'IN' or 'OUT'")

        url = self.base_url + f'/org/{self.client_slug}/unrecognized-faces'
        headers = {'Authorization': f'Bearer {self.token}'}
        # data ={'user_status': status.lower()}
        timestamp = timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        data = {'detection_time':timestamp}

        if face is None or face.size == 0:
            self.args.logger.warning("No face detected to send")
            return

        success, encoded_image = cv2.imencode('.jpg', face)
        if not success:
            raise ValueError("Image encoding failed")

        # Convert to byte stream
        image_bytes = io.BytesIO(encoded_image.tobytes())

        # Prepare file payload
        files = [
            ('images', ('cropped_face.jpg', image_bytes, 'image/jpeg')),
        ]
        try:
            response = requests.post(url, headers=headers, files=files, data=data)

            response.raise_for_status()

            return response

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Send unrecognized face request failed: {str(e)}")

    def log_person_entry(self, name, status, appear_time, camera_name="Unknown"):
        """Log a person's entry or exit.

        Args:
            name: Name of the person
            status: Entry/exit status (IN/OUT)
            appear_time: Time when the person appeared
            camera_name: Name of the camera that detected the person
        Returns:
            bool: True if the status was recorded, False if new status is the same as previous status
        """


        previous_status = self.person_status.get(name)
        recorded = False
        # If status is the same as before, do nothing
        if previous_status == status.upper():
            timestamp = appear_time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {name} | Status: {status} | Camera: {camera_name} (unchanged)")
            return recorded
        recorded = True
        # Update the cached status
        self.person_status[name] = status.upper()

        # Format the appearance time
        today_date = appear_time.strftime("%Y-%m-%d")
        today_time = appear_time.strftime("%H:%M:%S")

        if self.args.production:
            # Send data to the API
            self.send_data_to_api(name, status)

        # ANSI color codes
        BOLD = "\033[1m"
        BLUE = "\033[94m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        if status.upper() == "IN":
            print(f"{BOLD}{BLUE}STATUS   | {name} {status.upper()} at {today_time}{RESET}")
        else:
            print(f"{BOLD}{YELLOW}STATUS   | {name} {status.upper()} at {today_time}{RESET}")

        # Log status change to CSV file using the dedicated sink
        csv_message = f'{name},{status},{today_time},{today_date}'
        self.args.logger.bind(is_csv=True).info(csv_message)


        self.recent_entries.appendleft(f"{name} - {status} @ {today_time}")
        return recorded

    def create_record(self, user_id: int, status: str) -> Dict[str, Any]:
        """
        Create an attendance record

        Args:
            user_id (int): ID of the user to create record for
            status (str): Either 'IN' or 'OUT'

        Returns:
            dict: Record creation response data

        Raises:
            requests.exceptions.RequestException: If record creation fails
        """
        if not self.token:
            raise ValueError("Not authenticated. Please login first.")
        status = status.upper()
        if status not in ['IN', 'OUT']:
            raise ValueError("Status must be either 'IN' or 'OUT'")

        # Generate timestamp in ISO 8601 format with milliseconds
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        record_data = {
            "user_id": user_id,
            "status": status.lower(),  # Use lowercase for consistency
            "camera_id": getattr(self, 'camera_id', None),
            "timestamp": timestamp
        }

        try:
            response = self.session.post(
                f"{self.base_url}/org/{self.client_slug}/attendance-records",
                json=record_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            return response

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Record creation request failed: {str(e)}")

    def visualize_entries(self, frame, max_text_width=0):
        """Visualize recent entries on the frame.

        Args:
            frame: Frame to add visualization to
            max_text_width: Maximum width of the text display
        """
        self.padding = 10
        self.base_y = 30

        for entry in self.recent_entries:
            text_size = cv2.getTextSize(entry, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            max_text_width = max(max_text_width, text_size[0])

        for i, entry in enumerate(self.recent_entries):
            top_right_x = frame.shape[1] - max_text_width - self.padding * 2
            cv2.rectangle(
                frame,
                (top_right_x, self.base_y - 25 + i * 35),
                (frame.shape[1] - 10, self.base_y + i * 35 + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                entry,
                (top_right_x + 5, self.base_y + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    def save_status_info(self, video_name='status_info'):
        """Returns the path to the status information log file.

        Args:
            video_name: Base name for the output CSV file (ignored, kept for compatibility)

        Returns:
            Text message indicating where the status information was saved
        """
        text = f"Status information has been saved continuously to {self.log_file_path}"
        return text

    def fetch_all_history(self, page: int = 1, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch all history records from the API.
        Args:
            all_records: List to accumulate records (used for recursive calls)
            page: Current page number for pagination
            limit: Number of records per page
        Returns:
            List of all history records"""
        all_records = []

        while True:
            response = self.session.get(
                f"{self.base_url}/{self.client_slug}/history",
                headers=self.headers,
                params={"page": page, "limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            if not data.get('success', True):
                raise requests.exceptions.RequestException(f"Failed to get history: {data.get('error', 'Unknown error')}")
            records = data.get("records", [])
            all_records.extend(records)

            if page >= data.get("totalPages", 1):
                break
            page += 1

        self.args.logger.info(f"Fetched {len(all_records)} history records")
        return all_records

    def get_last_status(self):
        """Get the last status of each user from the history records.
        This method fetches all history records, filters OUT deleted users,
        and updates the person_status dictionary with the latest status for each user.
        Users with no records will be marked as 'OUT'.
        """

        headers = {"Authorization": f"Bearer {self.token}"}
        resp_in = requests.get(f"{self.base_url}/org/{self.slug}/users/in", headers=headers)
        resp_in.raise_for_status()
        data_in = resp_in.json()
        in_names = [user["full_name"] for user in data_in if "full_name" in user]

        # 2. Fetch current OUT users
        resp_out = requests.get(f"{self.base_url}/org/{self.slug}/users/out", headers=headers)
        resp_out.raise_for_status()
        data_out = resp_out.json()
        out_names = [user["full_name"] for user in data_out if "full_name" in user]

        person_status = {}

        for name in out_names:
            person_status[name] = "OUT"

        for name in in_names:
            # IN overrides OUT if there's any overlap
            person_status[name] = "IN"

        _, _, name_to_id = self.get_all_users()
        for entry in name_to_id:
            name = entry["name"]
            if name not in person_status:
                # They're "active" in org but not explicitly IN or OUT.
                # Old behavior: treat as OUT and warn.
                self.logger.warning(f"No status from API for user '{name}', defaulting to OUT")
                person_status[name] = "OUT"

        return person_status

    def send_annotated_frame(self, frame, camera_index, camera_type):
        """Send annotated frame to the API.

        Args:
            frame: Annotated frame to send
            camera_index: Camera index (0, 1, 2, etc.)
            camera_type: Camera type (IN/OUT)
        Raises:
            ValueError: If the frame cannot be encoded or if the user is not authenticated
            requests.exceptions.RequestException: If the request to the API fails
        Uses:
            requests: To send the frame to the API
        """
        if not self.token:
            raise ValueError("Not authenticated. Please login first.")

        url = self.base_url + f'/{self.client_slug}/cameras/{camera_index}/{camera_type}'
        headers = {
            'Authorization': f'Bearer {self.token}'
        }

        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            raise ValueError("Image encoding failed")
        image_bytes = io.BytesIO(encoded_image.tobytes())

        # Prepare file payload
        files = [
            ('images', ('annotated_frame.jpg', image_bytes, 'image/jpeg')),
        ]
        try:
            response = requests.post(url, headers=headers, files=files)

            response.raise_for_status()
            if response.status_code != 201:
                self.args.logger.warning(f"Failed to send annotated frame: {response.text}")
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Send annotated frame request failed: {str(e)}")

