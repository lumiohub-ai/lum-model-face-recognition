from collections import defaultdict

class TrackManager:
    def __init__(self, ):
        self.track_frame_count = defaultdict(int)
        self.name_to_track_id = {}
        self.name_to_color = {}