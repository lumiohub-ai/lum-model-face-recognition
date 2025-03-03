import os
import cv2
import threading
import queue
from typing import Any, Tuple, List

from cfg import Config
from engine import FaceRecognitionModel
from utils import EntryLogger, Visualization, ShapeDrawer, VideoStream

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class VideoProcessor:
    def __init__(self, cfg: Config, model: FaceRecognitionModel, 
                 entry_logger: EntryLogger, visualize: Visualization) -> None:
        
        self.cfg = cfg
        self.model = model
        self.entry_logger = entry_logger
        self.visualize = visualize

        self.in_line_points = self.initialize_shape_drawer(self.get_first_frame(cfg.in_camera))
        self.out_line_points = self.initialize_shape_drawer(self.get_first_frame(cfg.out_camera))
        
        self.in_stream = VideoStream(cfg.in_camera)
        self.out_stream = VideoStream(cfg.out_camera)
        
        self.name_to_track_id = {"IN": {}, "OUT": {}}
        self.name_to_color = {}
        self.current_frame_persons = []
        # Ensure ShapeDrawer completes before continuing

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filename = f"combined_output_{timestamp}.mp4"
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = None
    
    @staticmethod
    def get_first_frame(source: str) -> Any:
        cap = cv2.VideoCapture(source)
        _, frame = cap.read()
        cap.release()
        
        return frame

    @staticmethod
    def initialize_shape_drawer(frame):
        return ShapeDrawer(frame).run()[0]

    def process_detections(self, detections: Any, frame: Any, 
                           cam_type: str, track_status: List) -> Any:
        if detections.id is None:
            return frame

        boxes = detections.data.cpu().tolist()
        track_ids = detections.id.cpu().tolist()

        for det, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2, *rest = map(int, det)
            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            face = frame[y1:y2, x1:x2]
            name = self.name_to_track_id.get(cam_type, {}).get(track_id, "Detecting...")

            if name == "Detecting...":
                face_emb = self.model.compute_embeddings(face)
                name = self.model.recognize_face(face_emb)
                if name != "Detecting..." and name not in self.current_frame_persons:
                    self.current_frame_persons.append(name)
                    self.name_to_track_id[cam_type][track_id] = name
            
            if int(track_id) in track_status and name != "Detecting...":
                self.entry_logger.log_person_entry(name, cam_type)

            if name not in self.name_to_color:
                self.name_to_color[name] = (1, 31, 242) if name == "Detecting..." else self.visualize.generate_random_color()
            
            color = self.name_to_color[name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def run(self) -> None:
        if not self.in_stream.is_video:
            self.in_stream.start()
        if not self.out_stream.is_video:
            self.out_stream.start()

        try:
            while True:

                in_ret, in_frame = self.in_stream.read()
                out_ret, out_frame = self.out_stream.read()
                
                if not in_ret and not out_ret:
                    break

                if in_ret:
                    self.model.in_counter.count(in_frame, region=self.in_line_points)
                    in_detections = self.model.in_counter.track_data
                    in_counted_ids = self.model.in_counter.counted_ids
                    in_frame = self.process_detections(in_detections, in_frame, "IN", in_counted_ids)

                if out_ret:
                    self.model.out_counter.count(out_frame, region=self.out_line_points)
                    out_detections = self.model.out_counter.track_data
                    out_counted_ids = self.model.out_counter.counted_ids
                    out_frame = self.process_detections(out_detections, out_frame, "OUT", out_counted_ids)

                # Visualize region areas
                self.visualize.draw_region(in_frame, self.in_line_points)
                self.visualize.draw_region(out_frame, self.out_line_points)
                
                combined_frame = self.visualize.concat_frames(in_frame, out_frame, mode="horizontal")
                combined_frame = cv2.resize(combined_frame, (1920, 800))

                if self.video_writer is None:
                    self.video_writer = cv2.VideoWriter(self.output_filename, self.fourcc, 20.0, (1920, 800))
                
                self.video_writer.write(combined_frame)
                
                self.entry_logger.visualize_entries(combined_frame)
                if self.visualize.display(combined_frame, "Face Recognition System"):
                    break
        except:
            pass

        finally:
            self.in_stream.stop()
            self.out_stream.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"  # UDP can reduce delay

    cfg = Config()
    model = FaceRecognitionModel()
    entry_logger = EntryLogger()
    visualize = Visualization()

    video_processor = VideoProcessor(cfg, model, 
                                     entry_logger, visualize)
    
    video_processor.run()
