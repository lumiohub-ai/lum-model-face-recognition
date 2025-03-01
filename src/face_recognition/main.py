import os
import cv2
import threading
from typing import Any, Tuple, List

from cfg import Config
from engine import FaceRecognitionModel, TrackManager
from utils import EntryLogger, Visualization, ShapeDrawer

import warnings
warnings.filterwarnings("ignore")


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class VideoStream:
    def __init__(self, src: Any) -> None:
        self.cap = cv2.VideoCapture(src)
        self.stopped = False
        self.lock = threading.Lock()
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Unable to read from camera source: {src}")
        self.ret = ret
        self.frame = frame
        self.thread = None  # store reference to thread

    def start(self) -> "VideoStream":
        # Create a non-daemon thread so we can join it on stop.
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        return self

    def update(self) -> None:
        while True:
            # Check if we need to stop, protected by the lock.
            with self.lock:
                if self.stopped:
                    break
            ret, frame = self.cap.read()
            if not ret:
                # Signal to stop and break out of the loop.
                with self.lock:
                    self.stopped = True
                break
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self) -> Tuple[bool, Any]:
        with self.lock:
            return self.ret, self.frame
        
    def get_first_frame(self) -> Any:
        return self.frame

    def stop(self) -> None:
        # Signal stop under lock.
        with self.lock:
            if self.stopped:
                return
            self.stopped = True
        # Wait for the update thread to finish.
        if self.thread is not None:
            self.thread.join()
        # Now it is safe to release the VideoCapture.
        self.cap.release()

    def __enter__(self) -> "VideoStream":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()



class VideoProcessor:
    def __init__(
        self,
        cfg: Config,
        model: FaceRecognitionModel,
        entry_logger: EntryLogger,
        track_in: TrackManager,
        track_out: TrackManager,
        visualize: Visualization
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.entry_logger = entry_logger
        self.track_in = track_in
        self.track_out = track_out
        self.visualize = visualize

        self.in_stream = VideoStream(cfg.in_camera)
        self.out_stream = VideoStream(cfg.out_camera)

        self.name_to_track_id = {
            "IN": {},
            "OUT": {}
        }
        self.name_to_color = {}

        # Define ROI once using ShapeDrawer
        in_frame = self.in_stream.get_first_frame()
        out_frame = self.out_stream.get_first_frame()

        # # ROI
        self.in_bbox = [372, 30, 1241, 720]
        self.out_bbox = [2, 114, 547, 717]

        # # Mapped ROI
        in_frame = in_frame[self.in_bbox[1]:self.in_bbox[3], self.in_bbox[0]:self.in_bbox[2]]
        out_frame = out_frame[self.out_bbox[1]:self.out_bbox[3], self.out_bbox[0]:self.out_bbox[2]]

        # self.in_line_points = ShapeDrawer(in_frame).run()[0]
        # self.out_line_points = ShapeDrawer(out_frame).run()[0]

        self.in_line_points = [(196, 13), (171, 640)]
        self.out_line_points = [(321, 101), (61, 565)]

        self.current_frame_persons = []

    def process_detections(
        self, detections: Any, frame: Any, cam_type: str, track_status: List
    ) -> Any:

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
            box_width = x2 - x1
            font_scale = max(0.5, box_width / 300)
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)[0]
            text_x = min(x2 - text_size[0] - 5, x1 + 5)
            text_y = max(0, y1 - 10)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2)
            cv2.putText(frame, str(track_id), (text_x - 10, text_y - 10), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2)


        return frame

    def run(self) -> None:
        self.in_stream.start()
        self.out_stream.start()
        frame_number = 0

        in_bbox = [372, 30, 1241, 720]
        out_bbox = [2, 114, 547, 717]

        try:
            while True:
                frame_number += 1
                if frame_number % self.cfg.skip_frames != 0:
                    continue

                self.model.check_new_faces()

                in_ret, in_frame = self.in_stream.read()
                out_ret, out_frame = self.out_stream.read()

                if not in_ret or not out_ret:
                    break

                # # Map bbox to the frame
                in_frame = in_frame[in_bbox[1]:in_bbox[3], in_bbox[0]:in_bbox[2]]
                out_frame = out_frame[out_bbox[1]:out_bbox[3], out_bbox[0]:out_bbox[2]]

                ### IN 
                self.model.in_counter.count(
                    in_frame, region=self.in_line_points
                )

                in_detections = self.model.in_counter.track_data
                in_counted_ids = self.model.in_counter.counted_ids
            
                ### OUT
                self.model.out_counter.count(
                    out_frame, region=self.out_line_points
                )

                out_detections = self.model.out_counter.track_data
                out_counted_ids = self.model.out_counter.counted_ids


                in_frame_dets = self.process_detections(in_detections, in_frame, "IN", in_counted_ids)
                out_frame_dets = self.process_detections(out_detections, out_frame, "OUT", out_counted_ids)

                self.current_frame_persons = []
                
                # Visualize region areas
                self.visualize.draw_region(in_frame_dets, self.in_line_points)
                self.visualize.draw_region(out_frame_dets, self.out_line_points)
            
                # Concatenate the two frames for a combined view.
                combined_frame = self.visualize.concat_frames(in_frame_dets, 
                                                              out_frame_dets, mode="horizontal")

                # Resize combined frame to display
                combined_frame = cv2.resize(combined_frame, (1920, 800))

                self.entry_logger.visualize_entries(combined_frame)

                if self.visualize.display(combined_frame, "Face Recognition System"):
                    break


        finally:
            self.in_stream.stop()
            self.out_stream.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config()
    
    model = FaceRecognitionModel(cfg.device, cfg.face_crops_path,
                                in_region_points=cfg.in_region_points, out_region_points=cfg.out_region_points,
                                match_threshold=cfg.match_threshold 
                                )
    entry_logger = EntryLogger(cfg.logging_path)

    track_in = TrackManager()
    track_out = TrackManager()
    visualize = Visualization()

    video_processor = VideoProcessor(cfg, model, entry_logger, track_in, track_out, visualize)
    video_processor.run()
