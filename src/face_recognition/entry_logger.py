from collections import deque
import cv2
import pandas as pd
import os
import json
import requests

class EntryLogger:
    def __init__(self, 
                args,
                max_entries=3):
        self.args = args
        
        self.create_user_api_url = "https://api.smart-office.humblebee.ai/api/history/create"
        self.get_all_users_api_url = "https://api.smart-office.humblebee.ai/api/users"

        self.headers = {"Content-Type": "application/json"}

        self.recent_entries = deque(maxlen=max_entries)
        self.person_status = {}
        self.saving_status_info = []

        self.current_users = args.db_names

        self.new_users, self.deleted_users, self.name_to_id = self.get_all_users()
        
    def get_all_users(self):
        response = requests.get(self.get_all_users_api_url)

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

    def send_data_to_api(self, name, status):
        user_id = next((int(i['id']) for i in self.name_to_id if i['name'] == name), None)

        if user_id is None:
            self.args.logger.warning(f'User with ID {user_id} not found in the database')
            return
        
        payload = {
            "user_id": user_id,
            "detection_type": status.lower()
        }
        
        response = requests.post(self.create_user_api_url, json=payload, headers=self.headers)
        
        data = response.json()
        if response.status_code == 201:
            self.args.logger.info(f"Success: {data['message']}")
        else:
            self.args.logger.warning(f"Error: {data.get('error', 'Unknown error')}")


    def log_person_entry(self, name, status, appear_time):
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

    def visualize_entries(self, frame, max_text_width=0):
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
        os.makedirs('logs', exist_ok=True)

        # Convert saving_status_info to DataFrame and save to CSV
        df = pd.DataFrame(self.saving_status_info)
        df.to_csv(f'logs/{video_name}.csv', index=False)

        text = f"Status information saved to logs/{video_name}.csv"

        return text


