import sys
import os
sys.path.append(os.curdir)

import cv2

from src.face_recognition.engine import FaceEngine
from src.face_recognition.utils import Visualization, StreamHandler, EntryLogger

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class HBFace(FaceEngine):
    def __init__(self, video_path, cam_type, annot=True, eval=False,
                 roi=None, line_points=None) -> None:
        super().__init__(video_path=video_path, eval=eval)
        self.roi = roi
        self.cam_type = cam_type
        self.annot = annot
        self.video_path = video_path
        self.line_points = line_points
        
        if video_path is None:
            self.video_path = self.args.video_path
        
        self.recognized_names = set()

        self.stream = StreamHandler(self.video_path)
        self.visualize = Visualization()
        self.entry_logger = EntryLogger()

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
            annotated_frame = self.process_detections(frame, frame_num, roi=self.roi)

            # Process removed tracks and recognize faces
            recognized_persons = self.recognize_tracks(detections, line_points=self.line_points,
                                                        last_frame=(frame_num == self.stream.last_frame))

            for name, track_id in recognized_persons.items():
                self.entry_logger.log_person_entry(name, self.cam_type, track_id)
                self.recognized_names.add(name)

            # Draw line points
            if self.line_points is not None:
                cv2.line(annotated_frame, self.line_points[0], self.line_points[1], (0, 255, 0), 2)
            
            self.entry_logger.visualize_entries(annotated_frame)
            self.visualize.display(annotated_frame, window_name=self.cam_type)
            
        self.stream.stop()
        cv2.destroyAllWindows()
