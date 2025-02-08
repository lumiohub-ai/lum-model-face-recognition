from datetime import datetime
from collections import deque
import cv2

class EntryLogger:
    def __init__(self, logging_path):
        self.entry_time = {}
        self.logging_path = logging_path
        self.recent_entries = deque(maxlen=3)
        self.base_y = 30
        self.padding = 10

    def log_person_entry(self, name):
        if name != "Detecting..." and name not in self.entry_time:
            now = datetime.now().strftime("%H:%M:%S on %d.%m.%Y")
            log_entry = f"{name} entered at {now}"
            self.entry_time[name] = now
            self.recent_entries.append(log_entry)  # Add to the recent entries deque
            #save the information to the csv file
            with open(self.logging_path, 'a') as f:
                f.write(log_entry + '\n')

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


