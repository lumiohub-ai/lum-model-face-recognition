import os

import cv2
from engine import FaceEngine

from utils import Visualization, StreamHandler, EntryLogger

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class HBFace(FaceEngine):
    def __init__(self, video_path, cam_type, annot=True, eval=False) -> None:
        super().__init__(eval=eval)
        self.cam_type = cam_type
        self.annot = annot
        self.video_path = video_path
        if video_path is None:
            self.video_path = self.args.video_path
        
        self.recognized_names = set()

        self.stream = StreamHandler(self.video_path)
        self.visualize = Visualization()
        self.entry_logger = EntryLogger()

    def run(self) -> None:
        if not self.stream.is_video:
            self.stream.start()

        try:
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
                annotated_frame = self.process_detections(frame, frame_num)

                # Process removed tracks and recognize faces
                recognized_persons = self.recognize_tracks(detections, (frame_num == self.stream.last_frame))

                for name in recognized_persons:
                    self.entry_logger.log_person_entry(name, self.cam_type)
                    self.recognized_names.add(name)
                
                self.entry_logger.visualize_entries(annotated_frame)
                self.visualize.display(annotated_frame, window_name=self.cam_type)
                
        except Exception as e:
            print(e)

        finally:
            self.stream.stop()
            cv2.destroyAllWindows()
