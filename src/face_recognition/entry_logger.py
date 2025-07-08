"""Entry logging module for tracking and visualizing person entries and exits."""

from collections import deque
import cv2
import pandas as pd
import os
import io
import json
import requests
from typing import Optional, Dict, Any

clientSlug = 'humblebee'
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
        self.args = args

        self.headers = {"Content-Type": "application/json"}

        self.recent_entries = deque(maxlen=max_entries)
        self.person_status = {}
        self.saving_status_info = []
        self.base_url = args.api_host + 'api'
        self.token = None
        self.client_slug = None
        self.session = requests.Session()

        self.login_response = self.login('humblebee', 'Hbvision2025@', 'humblebee')
        self.current_users = args.db_names

        self.new_users, self.deleted_users, self.name_to_id = self.get_all_users()
        
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
                f"{self.base_url}/{self.client_slug}/users",
                params={"page": page, "limit": limit, "status": "active"}
            )

        new_users = []
        deleted_users = []

        if response.status_code == 200:
            users = response.json()
            
            user_dict = {user['username']: user['id'] for user in users}
            path_dict = {user['id']: user['image_path'] for user in users}

            name_to_id = [{'name': name, 'id': user_id} for name, user_id in user_dict.items()]
            id_to_path = [{'id': user_id, 'path': path} for user_id, path in path_dict.items()]

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
   
    def login(self, username: str, password: str, client_slug: str) -> Dict[str, Any]:
        """
        Authenticate user and obtain access token
        
        Args:
            username (str): User's username
            password (str): User's password
            client_slug (str): Client slug for organization
            
        Returns:
            dict: Login response data
            
        Raises:
            requests.exceptions.RequestException: If login fails
        """
        login_data = {
            "username": username,
            "password": password,
            "client_slug": client_slug
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json=login_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('success'):
                self.token = data.get('token')
                self.client_slug = client_slug
                # Set authorization header for future requests
                self.session.headers.update({
                    'Authorization': f'Bearer {self.token}'
                })
                return data
            else:
                raise requests.exceptions.RequestException(f"Login failed: {data.get('error', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Login request failed: {str(e)}")
        
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
        
        payload = {
            "user_id": user_id,
            "detection_type": status.lower()
        }
        
        response = self.create_record(user_id, status)
        data = response.json()
        if response.status_code == 201:
            self.args.logger.info(f"Success: {data['message']}")
        else:
            self.args.logger.warning(f"Error: {data.get('error', 'Unknown error')}")

    def send_unrecognized_face(self, face, status):
        if not self.token:
            raise ValueError("Not authenticated. Please login first.")
        status = status.lower()
        if status not in ['in', 'out']:
            raise ValueError("Status must be either 'in' or 'out'")
        
        url = self.base_url + f'/{self.client_slug}/unrecognized'
        headers = {
            'Authorization': f'Bearer {self.token}'
        }

        data ={'user_status': status}

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
            if response.status_code == 201:
                self.args.logger.info("Unrecognized face sent successfully")
            else:
                self.args.logger.warning(f"Failed to send unrecognized face: {response.text}")
            
            return response 
            
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Send unrecognized face request failed: {str(e)}")
        
    def log_person_entry(self, name, status, appear_time):
        """Log a person's entry or exit.
        
        Args:
            name: Name of the person
            status: Entry/exit status (IN/OUT)
            appear_time: Time when the person appeared
        """
        previous_status = self.person_status.get(name)

        # If status is the same as before, do nothing
        if previous_status == status:
            return

        # Update the cached status
        self.person_status[name] = status

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

        self.saving_status_info.append({
            'name': name,
            'status': status,
            'time': today_time,
            'date': today_date,
        })

        self.recent_entries.appendleft(f"{name} - {status} @ {today_time}")

    def create_record(self, user_id: int, status: str) -> Dict[str, Any]:
        """
        Create an attendance record
        
        Args:
            user_id (int): ID of the user to create record for
            status (str): Either 'in' or 'out'
            
        Returns:
            dict: Record creation response data
            
        Raises:
            requests.exceptions.RequestException: If record creation fails
        """
        if not self.token:
            raise ValueError("Not authenticated. Please login first.")
        status = status.lower()
        if status not in ['in', 'out']:
            raise ValueError("Status must be either 'in' or 'out'")
        
        record_data = {
            "user_id": user_id,
            "status": status
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/{self.client_slug}/history/create",
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
        """Save status information to a CSV file.
        
        Args:
            video_name: Base name for the output CSV file
            
        Returns:
            Text message indicating where the status information was saved
        """
        os.makedirs('logs', exist_ok=True)

        # Convert saving_status_info to DataFrame and save to CSV
        df = pd.DataFrame(self.saving_status_info)
        df.to_csv(f'logs/{video_name}.csv', index=False)

        text = f"Status information saved to logs/{video_name}.csv"

        return text


