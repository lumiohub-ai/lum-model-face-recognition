from datetime import datetime
from collections import deque

class EntryLogger:
    def __init__(self, logging_path):
        self.entry_time = {}
        self.logging_path = logging_path
        self.recent_entries = deque(maxlen=3)

    def log_person_entry(self, name):
        if name != "Detecting..." and name not in self.entry_time:
            now = datetime.now().strftime("%H:%M:%S on %d.%m.%Y")
            log_entry = f"{name} entered at {now}"
            self.entry_time[name] = now
            self.recent_entries.append(log_entry)  # Add to the recent entries deque
            #save the information to the csv file
            with open(self.logging_path, 'a') as f:
                f.write(log_entry + '\n')


