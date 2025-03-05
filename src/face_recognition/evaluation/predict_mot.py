import cv2
import sys
import os

sys.path.append(os.curdir)
sys.path.append(os.path.join(os.curdir, 'src/face_recognition'))

from ultralytics import YOLO
from engine import FaceRecognitionModel


video_path = 'output_11_processed_fps.mp4'
video_name = os.path.basename(video_path).split('.')[0]

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

cfg = {
    'model_arch': 'yolov8n-face.pt',
    'conf' : 0.25,
    'imgsz': 960,
    'persist': True,
    'tracker': 'bytetrack.yaml',
    'match_threshold': 0.7,
}

detector = YOLO(cfg['model_arch'])
model = FaceRecognitionModel()

frame_num = 0
name_to_track_id = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Do something with the frame here
    detections = detector.track(
            frame,
            verbose=False,
            conf=cfg['conf'],
            imgsz=cfg['imgsz'],
            persist=cfg['persist'],
            tracker=cfg['tracker'],
    )

    if detections[0].boxes.id is None:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = detections[0].boxes.data.cpu().tolist()
    track_ids = detections[0].boxes.id.cpu().tolist()

    for det, track_id in zip(boxes, track_ids):
        conf = det[5]
        track_id = int(track_id)
        x1, y1, x2, y2, _, _, _ = map(int, det)
        w = x2 - x1
        h = y2 - y1

        face = frame[y1:y2, x1:x2]
        name = name_to_track_id.get(track_id, "Detecting...")
        
        if name == "Detecting...":
            face_emb = model.compute_embeddings(face)
            name = model.recognize_face(face_emb)

            if name != "Detecting...":
                name_to_track_id[track_id] = name

        if name != "Detecting...":
            recognition_txt = f"{frame_num},{x1},{y1},{w},{h},{name}\n"

            with open(f'{video_name}_recognition.txt', 'a') as f:
                f.write(recognition_txt)


        track_txt = f"{frame_num},{track_id},{x1},{y1},{w},{h},{conf},-1,-1,-1,-1\n"
        with open(f'{video_name}_mot.txt', 'a') as f:
            f.write(track_txt)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

