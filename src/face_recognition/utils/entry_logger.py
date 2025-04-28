from collections import deque
import cv2
from gql import Client
from gql.transport.requests import RequestsHTTPTransport
from gql import gql

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
    def __init__(self, backend_url):
        self.backend_url = backend_url
        self.entry_time = {}
        self.recent_entries = deque(maxlen=3)
        self.base_y = 30
        self.padding = 10
        self.person_status = {}
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

    def log_person_entry(self, name, status, track_id, appear_time):
        updated = False
        today_date = appear_time.strftime("%Y-%m-%d")
        today_time = appear_time.strftime("%H:%M:%S")
        
        # Check if the person's status is new or has changed
        if self.person_status.get(name) != status:
            self.person_status[name] = status
            updated = True
        
        if updated:
            # Update the entry time and log the event
            self.entry_time[name] = today_time
            log_message = f"{name} -> {status} -> {today_time}, track_id: {track_id}"
            self.recent_entries.append(log_message)
            
            # Determine the action key based on status
            client_action_key = "clientIn" if status.upper() == "IN" else "clientOut"
            
            # Prepare the payload for the API call
            payload = {
                "clientName": name,
                client_action_key: today_time,
                "clientStatus": status,
                "clientWorkingDate": today_date,
            }

            print(log_message)
            
            # Execute the API call to send the data to the server
            # self.auth_client.execute(RECORD_DATA, variable_values={'input': payload})

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


