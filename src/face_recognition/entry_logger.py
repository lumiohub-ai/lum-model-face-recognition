from collections import deque
import cv2
import pandas as pd
import os
import json
import psycopg2

class EntryLogger:
    def __init__(self, 
                args,
                max_entries=3):
        self.args = args
        self.host = args.host
        self.name = args.name
        self.user = args.user
        self.password = args.password
        self.port = args.port
        self.path_to_db_config = args.path_to_db_config

        self.conn, self.cursor = self.connect_to_db()
        
        with open(args.path_to_db_config, 'r') as f:
            self.name_to_id = json.load(f)

        self.recent_entries = deque(maxlen=max_entries)
        self.person_status = {}
        self.saving_status_info = []
    
    def connect_to_db(self):

        conn = psycopg2.connect(
            host=self.host,
            database=self.name,
            user=self.user,
            password=self.password,
            port=self.port
        )

        cursor = conn.cursor()

        return conn, cursor
    
    def log_into_db(self, name, status, appear_time):
        user_id = next((int(i['id']) for i in self.name_to_id if i['name'] == name), None)
        
        self.cursor.execute("SELECT username FROM users WHERE id = %s", (user_id,))
        result = self.cursor.fetchone()

        if not result:
            self.args.logger.warning(f'User with ID {user_id} not found in the database')
            return
        
        username = result[0]
        status = status.strip().lower()

        if status == 'in':
            self.cursor.execute("""
                INSERT INTO dates (id, username, userIn, clientStatus, deleted)
                VALUES (%s, %s, %s, %s, FALSE)
            """, (user_id, username, appear_time, status))
        elif status == "out":
            self.cursor.execute("""
                INSERT INTO dates (id, username, userOut, clientStatus, deleted)
                VALUES (%s, %s, %s, %s, FALSE)
            """, (user_id, username, appear_time, status))

        self.conn.commit()

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

        self.log_into_db(name, status, appear_time)

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
    
    def close_db_connection(self):
        self.cursor.close()
        self.conn.close()
        


