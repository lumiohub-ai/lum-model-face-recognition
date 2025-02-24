import os
import cv2
import threading
import numpy as np
from typing import Any, Tuple

from cfg import Config
from engine import FaceRecognitionModel, TrackManager
from utils import EntryLogger, generate_random_color, display, concat_frames

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

    def start(self) -> "VideoStream":
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self) -> None:
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self) -> Tuple[bool, Any]:
        with self.lock:
            return self.ret, self.frame

    def stop(self) -> None:
        self.stopped = True
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
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.entry_logger = entry_logger
        self.track_in = track_in
        self.track_out = track_out

        self.in_stream = VideoStream(cfg.in_camera)
        self.out_stream = VideoStream(cfg.out_camera)

    def process_detections(
        self, detections: Any, frame: Any, cam_type: str, track: TrackManager
    ) -> Any:
        if not detections or detections[0].boxes.id is None:
            return frame

        boxes = detections[0].boxes.data.cpu().tolist()
        track_ids = detections[0].boxes.id.cpu().tolist()

        for det, track_id in zip(boxes, track_ids):
            if track_id not in track.track_frame_count:
                track.track_frame_count[track_id] = 0

            x1, y1, x2, y2, *rest = map(int, det)
            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            track.track_frame_count[track_id] += 1

            name = track.name_to_track_id.get(track_id, "Detecting...")

            if name == "Detecting...":
                face_emb = self.model.compute_embeddings(face)
                name = self.model.recognize_face(face_emb)
                if name != "Detecting...":
                    track.name_to_track_id[track_id] = name
                    self.entry_logger.log_person_entry(name, cam_type)

            if name not in track.name_to_color:
                track.name_to_color[name] = (0, 0, 0) if name == "Detecting..." else generate_random_color()

            color = track.name_to_color[name]
            box_width = x2 - x1
            font_scale = max(0.5, box_width / 300)
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)[0]
            text_x = min(x2 - text_size[0] - 5, x1 + 5)
            text_y = max(0, y1 - 10)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2)

        return frame

    def run(self) -> None:
        self.in_stream.start()
        self.out_stream.start()
        frame_number = 0

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

                in_detections = self.model.in_detector.track(
                    in_frame,
                    conf=self.cfg.detection_threshold,
                    verbose=False,
                    imgsz=self.cfg.imgsz,
                    persist=True,
                    tracker="bytetrack.yaml",
                )

                out_detections = self.model.out_detector.track(
                    out_frame,
                    conf=self.cfg.detection_threshold,
                    verbose=False,
                    imgsz=self.cfg.imgsz,
                    persist=True,
                    tracker="bytetrack.yaml",
                )

                in_frame_dets = self.process_detections(in_detections, in_frame, "IN", self.track_in)
                out_frame_dets = self.process_detections(out_detections, out_frame, "OUT", self.track_out)

                

                # Concatenate the two frames for a combined view.
                combined_frame = concat_frames(in_frame_dets, out_frame_dets, mode="horizontal")
                # Resize combined frame to display
                combined_frame = cv2.resize(combined_frame, (1900, 720))

                self.entry_logger.visualize_entries(combined_frame)

                if display(combined_frame, "Combined"):
                    break


        finally:
            self.in_stream.stop()
            self.out_stream.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config()
    model = FaceRecognitionModel(cfg.device, cfg.face_crops_path, cfg.match_threshold)
    entry_logger = EntryLogger(cfg.logging_path)
    track_in = TrackManager()
    track_out = TrackManager()

    video_processor = VideoProcessor(cfg, model, entry_logger, track_in, track_out)
    video_processor.run()
