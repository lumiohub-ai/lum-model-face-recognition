from datetime import datetime
from collections import deque
import cv2
import requests
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from .mutation import LOGIN, RECORD_DATA

class EntryLogger:
    def __init__(self, logging_path):
        self.entry_time = {}
        self.logging_path = logging_path
        self.recent_entries = deque(maxlen=3)
        self.base_y = 30
        self.padding = 10
        self.person_status = {}
        self.auth_client = self.authorize_user()

    def authorize_user(self):
        url = 'http://localhost:4000/graphql'

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

    def log_person_entry(self, name, status):
        update = False

        now =  datetime.now()
        today_date = now.strftime("%Y-%m-%d")
        today_time = now.strftime("%H:%M:%S")

        if name not in self.person_status.keys():
            # IF new name appears add key to the dictionary and add the status
            self.person_status[name] = status
            update = True


        else:
            # Check whether status has updated or not
            if self.person_status[name] != status:
                self.person_status[name] = status
                update = True

        if update:
            self.entry_time[name] = now
            self.recent_entries.append(f'Person: {name} has {status} at {today_time}')

            if status == 'IN':
                call = 'clientIn'
            else:
                call = 'clientOut'

            # API Call to send the data to the server
            input_variable = {
                "clientName": name,
                f"{call}": str(today_time),
                "clientStatus": status,
                "clientWorkingDate": str(today_date),
            }

            self.auth_client.execute(RECORD_DATA, variable_values={'input': input_variable})

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


