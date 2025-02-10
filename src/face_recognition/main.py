from cfg import Config
import cv2
from engine import FaceRecognitionModel, TrackManager
from utils import EntryLogger, generate_random_color, display
import os

from ultralytics import YOLO

import time

def save_body_crop(frame, person_det, person_name, save_dir="body_crops"):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract coordinates for the body bounding box
    x1_p, y1_p, x2_p, y2_p, _, _, _ = map(int, person_det)
    
    # Crop the body from the frame
    body_crop = frame[y1_p:y2_p, x1_p:x2_p]

    # Generate a unique filename using name and timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{person_name}_{timestamp}.jpg"

    # Save the cropped image
    cv2.imwrite(os.path.join(save_dir, filename), body_crop)
    print(f"Saved body crop for {person_name} as {filename}")


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

    while True:
        model.check_new_faces()

        ret, frame = cap.read()
        if not ret or stop:
            break

        person_tracks_to_name = {}

        x, y, w, h = cfg.camera_roi_coordinates
        frame = frame[y : y + h, x : x + w]

        detections = model.detector.track(
            frame,
            conf=cfg.detection_threshold,
            verbose=False,
            imgsz=cfg.imgsz,
            persist=True,
        )

        person_detections = person_detector.track(
            frame,
            conf=cfg.detection_threshold,
            verbose=False,
            imgsz=cfg.imgsz,
            persist=True,
        )

        if detections[0].boxes.id is None:
            stop = display(frame, out)
            continue

        boxes = detections[0].boxes.data.cpu().tolist()
        track_ids = detections[0].boxes.id.cpu().tolist()

        for det, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2, _, _, _ = map(int, det)
            # Get the center of the rectangle
            center = (x1 + x2) // 2, (y1 + y2) // 2

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
                    if person_detections[0].boxes.id is None:
                        # stop = display(frame, out)
                        continue

                    person_boxes = person_detections[0].boxes.data.cpu().tolist()
                    person_track_ids = person_detections[0].boxes.id.cpu().tolist()

                    for person_det, person_track_id in zip(person_boxes, person_track_ids):
                        x1_p, y1_p, x2_p, y2_p, _, _, _ = map(int, person_det)

                        if person_track_id in person_tracks_to_name.values():
                            person_detection_name = track.name_to_track_id.get(person_track_id, "Detecting...")
                            if person_detection_name != 'Detecting...':
                                save_body_crop(frame, person_det, person_detection_name)

                            cv2.rectangle(frame, (x1_p, y1_p), (x2_p, y2_p), (0, 255, 0), 3)
                            cv2.putText(
                                frame, person_detection_name, (x1_p, y1_p - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2
                            )

                            continue
            
                        x1_p, y1_p, x2_p, y2_p, _, _, _ = map(int, person_det)
                        # if face center point is inside the person bounding box make 
                        # that person the owner of the face and save the name
                        if x1_p < center[0] < x2_p and y1_p < center[1] < y2_p:
                            person_tracks_to_name[name] = person_track_id
                            person_detection_name = track.name_to_track_id.get(person_track_id, "Detecting...")
                            if person_detection_name != 'Detecting...':
                                save_body_crop(frame, person_det, person_detection_name)
                        
                        # cv2.rectangle(frame, (x1_p, y1_p), (x2_p, y2_p), (0, 255, 0), 3)
                        # cv2.putText(
                        #     frame, person_detection_name, (x1_p, y1_p - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2
                        # )

            entry_logger.log_person_entry(name)

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
    person_detector = YOLO('yolov8m.pt')
    model = FaceRecognitionModel(cfg.device, cfg.face_crops_path, cfg.match_threshold)
    entry_logger = EntryLogger(cfg.logging_path)
    track = TrackManager()

    main(cfg)
