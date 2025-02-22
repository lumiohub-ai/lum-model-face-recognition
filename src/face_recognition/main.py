
from cfg import Config
import cv2
from engine import FaceRecognitionModel, TrackManager
from utils import EntryLogger, generate_random_color, display
import os

import sys
import os
sys.path.append(os.curdir)

from ultralytics import solutions
import numpy as np



def main(cfg):
    cap = cv2.VideoCapture(cfg.video_path)

    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )
    out = cv2.VideoWriter(
        cfg.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (714, 659)
    )

    stop = False

    line_points = [(103, 171), (397, 158)]
    # [(159, 137), (162, 56), (254, 62), (285, 155), (277, 215), (215, 237), (160, 135)]

    counter = solutions.ObjectCounter(
        show=False,
        region=line_points,
        model="yolov8m-face.pt",
        classes=[0],
        show_in=True, 
        show_out=True,
        line_width=2,
        persist=True,
        verbose=False
    )

    while True:
        model.check_new_faces()

        ret, frame = cap.read()
        if not ret or stop:
            break

        # x, y, w, h = cfg.camera_roi_coordinates
        # frame = frame[y : y + h, x : x + w]

        # detections = model.detector.track(
        #     frame,
        #     conf=cfg.detection_threshold,
        #     verbose=False,
        #     imgsz=cfg.imgsz,
        #     persist=True,
        # )

        counter.count(np.ascontiguousarray(frame))
        detections = counter.track_data

        if detections.id is None:
            stop = display(frame, out)
            continue

        boxes = detections.data.cpu().tolist()
        track_ids = detections.id.cpu().tolist()

        for det, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2, _, _, _ = map(int, det)

            status = counter.track_status.get(int(track_id), None)
            if status is not None:
                print(f"Person {track_id} is {status}")

            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            track.track_frame_count[track_id] += 1

            if track.track_frame_count[track_id] >= cfg.check_interval:
                track.name_to_track_id.pop(track_id, None)
                track.track_frame_count[track_id] = 0

            name = track.name_to_track_id.get(track_id, "Detecting...")

            if name == "Detecting...":
                face_emb = model.compute_embeddings(face)
                name = model.recognize_face(face_emb)
                if name != "Detecting...":
                    track.name_to_track_id[track_id] = name

            entry_logger.log_person_entry(name, status)

            if name not in track.name_to_color:
                track.name_to_color[name] = (
                    (0, 0, 0) if name == "Detecting..." else generate_random_color()
                )

            color = track.name_to_color[name]
            box_width = x2 - x1
            font_scale = max(0.5, box_width / 300)
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)[0]
            text_x = min(x2 - text_size[0] - 5, x1 + 5)
            text_y = max(0, y1 - 10)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame, name, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2
            )

        entry_logger.visualize_entries(frame)

        stop = display(frame, out)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;10485760"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cfg = Config()

    model = FaceRecognitionModel(cfg.device, cfg.face_crops_path, cfg.match_threshold)
    entry_logger = EntryLogger(cfg.logging_path)
    track = TrackManager()

    main(cfg)
