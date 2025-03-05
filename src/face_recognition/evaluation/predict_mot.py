import cv2
import sys
import os

sys.path.append(os.curdir)

from ultralytics import YOLO


video_path = 'output_10_processed_fps.mp4'

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
    'output_txt_path': 'output_pred.txt',
}

detector = YOLO(cfg['model_arch'])

frame_num = 0
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
        conf = det[4]
        x1, y1, x2, y2, _, _, _ = map(int, det)
        w = x2 - x1
        h = y2 - y1

        saving_txt = f"{frame_num},{track_id},{x1},{y1},{w},{h},{conf},-1,-1,-1,-1\n"

        with open(cfg['output_txt_path'], 'a') as f:
            f.write(saving_txt)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

