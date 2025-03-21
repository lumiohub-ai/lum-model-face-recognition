import sys
import os
sys.path.append(os.curdir)

import cv2
import yaml

from src.face_recognition.engine import FaceEngine
from src.face_recognition.utils import Visualization, StreamHandler, EntryLogger

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class HBFace(FaceEngine):
    def __init__(self, cam_type=None, 
                 config_path="src/face_recognition/cfg/config.yaml", **kwargs) -> None:
        
        self.cam_type = cam_type
        self.load_cfgs(config_path, **kwargs)

        self.stream = StreamHandler(self.args.video_path)
        self.visualize = Visualization()
        self.entry_logger = EntryLogger()

        self.recognized_names = set()

        super().__init__(args=self.args)

        self.show_configs()

    def show_configs(self) -> None:
        # Show configurations to user with colors printings
        for key, value in self.args.__dict__.items():
            print(f"\033[1m{key}\033[0m: {value}")

    def load_cfgs(self, config_path, **kwargs):
        # Load config from yaml file
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception:
            print("Error loading config file. Using as empty config.")
            config = {}
        
        # Create args object to store all parameters
        self.args = type('Args', (), {})()
        
        # Merge config with kwargs (kwargs take precedence)
        for key, value in config.items():
            if key not in kwargs:  # Only set from config if not explicitly provided
                setattr(self.args, key, value)

        # Add explicitly provided parameters
        for key, value in kwargs.items():
            setattr(self.args, key, value)
            
        return self.args
        
    def run(self) -> None:
        if not self.stream.is_video:
            self.stream.start()

        frame_num = 0
        while True:
            ret, frame = self.stream.read()
            
            if not ret:
                break
        
            frame_num += 1

            # Track faces in the frame
            detections = self.track(frame)

            if detections is None:
                self.visualize.display(frame, window_name=self.cam_type)
                continue

            # Process detections
            annotated_frame = self.process_detections(frame, frame_num, roi=self.args.roi)

            # Process removed tracks and recognize faces
            recognized_persons = self.recognize_tracks(detections, line_points=self.args.line_points,
                                                        last_frame=(frame_num == self.stream.last_frame))

            for name, track_id in recognized_persons.items():
                self.entry_logger.log_person_entry(name, self.cam_type, track_id)
                self.recognized_names.add(name)

            # Draw line points
            if self.args.line_points is not None:
                cv2.line(annotated_frame, self.args.line_points[0], self.args.line_points[1], (0, 255, 0), 2)
            
            self.entry_logger.visualize_entries(annotated_frame)
            self.visualize.display(annotated_frame, window_name=self.cam_type)
            
        self.stream.stop()
        cv2.destroyAllWindows()
