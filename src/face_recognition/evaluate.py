import sys
import os
sys.path.append(os.curdir)

from cfg import Config
import cv2
from engine import FaceRecognitionModel, TrackManager
from utils import EntryLogger, Visualization

from ultralytics import YOLO

def main(cfg, model):
    cap = cv2.VideoCapture(cfg.video_path)
    detector = YOLO('/home/hb-nano/mirsaid/face-recognition/models/yolov8n-face.engine')

    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )
    person_names = []

    stop = False
    frame_count = 0
    while True:    
        model.check_new_faces()

        ret, frame = cap.read()
        if not ret or stop:
            break

        # Add frame skipping
        frame_count += 1
        if frame_count % 3 != 0:
             cv2.imshow("Face Recognition", frame)
             continue
        
        # frame = frame[y : y + h, x : x + w]

        det_num = 0
        detections = detector(
            frame,
            conf=cfg.detection_threshold,
            verbose=False,
            imgsz=cfg.imgsz,
        )

        # if detections[0].boxes.id is None:
        #     cv2.imshow("Face Recognition", frame)
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break  
        #     continue

        boxes = detections[0].boxes.data.cpu().tolist()
        # track_ids = detections[0].boxes.id.cpu().tolist()

        for det in boxes:
            det_num += 1
            x1, y1, x2, y2, _, _ = map(int, det)
            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            #track.track_frame_count[track_id] += 1

            # if track.track_frame_count[track_id] >= cfg.check_interval:
            #     track.name_to_track_id.pop(track_id, None)
            #     track.track_frame_count[track_id] = 0

            # name = track.name_to_track_id.get(track_id, "Detecting...")

            face_emb = model.compute_embeddings(face)
            name = model.recognize_face(face_emb)

            if name != "Detecting...":
                #track.name_to_track_id[track_id] = name
                person_path = f'/home/hb-nano/mirsaid/face-recognition/data/images/{name}.jpg'
                person_names.append(name)
                # person crop vs face crop comparison and save it
                person = cv2.imread(person_path)

                if person is not None and face is not None:
                    # Resize face crop to match height of the person image
                    face_resized = cv2.resize(face, (person.shape[1], person.shape[0]))

                    # Concatenate the images horizontally
                    comparison_image = cv2.hconcat([person, face_resized])

                    # Save the concatenated image
                    save_path = f'/home/hb-nano/mirsaid/face-recognition/comparison_7_fps_without_track_hb/{name}_comparison_{det_num}_{frame_count}.jpg'
                    cv2.imwrite(save_path, comparison_image)

            # if name == "Detecting...":
            #     face_emb = model.compute_embeddings(face)
            #     name = model.recognize_face(face_emb)
               
                    

            # entry_logger.log_person_entry(name)
            # if name not in track.name_to_color:
            #     track.name_to_color[name] = (
            #         (0, 0, 0) if name == "Detecting..." else Visualization.generate_random_color()
            #     )

            box_width = x2 - x1
            font_scale = max(0.5, box_width / 300)
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)[0]
            text_x = min(x2 - text_size[0] - 5, x1 + 5)
            text_y = max(0, y1 - 10)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.putText(
                frame, name, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2
            )

        #entry_logger.visualize_entries(frame)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  

    print(person_names)
    print(len(person_names))
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":


    cfg = Config()

    model = FaceRecognitionModel(cfg.device, cfg.face_crops_path, cfg.match_threshold)
    entry_logger = EntryLogger(cfg.logging_path)
    track = TrackManager()

    main(cfg, model)



