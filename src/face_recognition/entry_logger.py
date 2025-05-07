from collections import deque
import cv2
from gql import Client
from gql.transport.requests import RequestsHTTPTransport
from gql import gql
import pandas as pd
import os

LOGIN = gql('''
    mutation Login($input: LoginInput!) {
        login(input: $input) {
            _id
            memberType
            memberStatus
            memberAuthType
            memberPhone
            memberNick
            createdAt
            updatedAt
            accessToken
        }
    }
''')

RECORD_DATA = gql('''
    mutation CreateClientDate($input: DateInput!) {
    createClientDate(input: $input) {
        _id
        clientName
        clientIn
        clientOut
        clientWorkingDate
        clientStatus
        clientId
    }
}
''')

class EntryLogger:
    def __init__(self, backend_url, max_entries=3):
        self.backend_url = backend_url
        self.entry_time = {}
        self.recent_entries = deque(maxlen=max_entries)
        self.base_y = 30
        self.padding = 10

        self.person_status = {}
        self.saving_status_info = []

        # self.auth_client = self.authorize_user()

    def authorize_user(self):
        url = self.backend_url

        client = Client(
            transport=RequestsHTTPTransport(
                url=url,
                use_json=True,
            ),
            fetch_schema_from_transport=True,
        )

        login_variables = {
            "input": {
                "memberNick": "Admin", 
                "memberPassword": "123456"
            }
        }

        loginResponse = client.execute(LOGIN, variable_values=login_variables)

        access_token = loginResponse['login']['accessToken']

        auth_transport = RequestsHTTPTransport(
            url=url,
            headers={'Authorization': f'Bearer {access_token}'},
            use_json=True,
        )

        auth_client = Client(
            transport=auth_transport,
            fetch_schema_from_transport=True,
        )

        return auth_client

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

        # Choose correct API field
        client_action_key = "clientIn" if status.upper() == "IN" else "clientOut"

        # Prepare payload
        payload = {
            "clientName": name,
            client_action_key: today_time,
            "clientStatus": status,
            "clientWorkingDate": today_date,
        }

        # self.auth_client.execute(RECORD_DATA, variable_values={'input': payload})

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
        
        


