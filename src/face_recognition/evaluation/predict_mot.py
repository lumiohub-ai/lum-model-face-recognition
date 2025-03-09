import cv2
import sys
import os
import argparse

sys.path.append(os.curdir)
sys.path.append(os.path.join(os.curdir, 'src/face_recognition'))

from ultralytics import YOLO
from engine import FaceRecognitionModel
from facenet_pytorch import MTCNN

def save_crops(crops, track_id):
    os.makedirs(f'crops/{track_id}', exist_ok=True)
    for frame_num, crop in crops.items():
        cv2.imwrite(f'crops/{track_id}/{frame_num}.jpg', crop)


def set_paths(video_name, alg_name):
    output_recognition_path = f'results/{video_name}_recognition_{alg_name}.txt'
    output_tracking_path = f'TrackEval/data/trackers/mot_challenge/hbface-train/{alg_name}/data/{video_name}.txt'
    os.makedirs(os.path.dirname(output_tracking_path), exist_ok=True)

    if os.path.exists(output_recognition_path):
        os.remove(output_recognition_path)
    if os.path.exists(output_tracking_path):
        os.remove(output_tracking_path)
    return output_recognition_path, output_tracking_path


def predict(video_path, video_name, alg_name,
            model_arch, confidence_threshold, imgsz, tracker, match_threshold,
            show=True
            ):
    output_recognition_path, output_tracking_path= set_paths(video_name, alg_name)
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    cfg = {
        'model_arch': model_arch,
        'conf' : confidence_threshold,
        'imgsz': imgsz,
        'persist': True,
        'tracker': tracker,
        'match_threshold': match_threshold,
    }

    detector = YOLO(cfg['model_arch'])

    model = FaceRecognitionModel(match_threshold=cfg['match_threshold'])

    frame_num = 0
    track_crops_frame = {}
    passed_tracks = []
    name_to_track_id = {}
    all_tracks = []

    last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # save one frame
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
            if show:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        boxes = detections[0].boxes.data.cpu().tolist()
        track_ids = detections[0].boxes.id.cpu().tolist()
        
        for det, track_id in zip(boxes, track_ids):
            track_id = int(track_id)
            
            all_tracks.append(track_id)

            x1, y1, x2, y2, _, _, _ = map(int, det)

            face = frame[y1:y2, x1:x2]
            
            if track_id not in track_crops_frame.keys():
                track_crops_frame[track_id] = {}

            track_crops_frame[track_id][frame_num] = face

        removed_tracks = detections[0].removed_tracks.tolist()
        removed_tracks = [id for id in removed_tracks if id not in passed_tracks]

        for id in removed_tracks:
            if id == 31:
                pass

            passed_tracks.append(id)
            face_embeddings = model.compute_embeddings(track_crops_frame[id].values())
            # After recognition, delete crops to save memory
            del track_crops_frame[id]

            name = model.recognize_face(face_embeddings)
            name_to_track_id[id] = name
            
            text_show = f'{id} recognized as {name}'
            
            if name != "Unknown":
                print(f'{id} recognized as {name}')
                cv2.putText(frame, text_show, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if frame_num == last_frame:
            for id in all_tracks:
                if id not in passed_tracks:
                    face_embeddings = model.compute_embeddings(track_crops_frame[id].values())

                    del track_crops_frame[id]

                    name = model.recognize_face(face_embeddings)
                    name_to_track_id[id] = name
                    text_show = f'{id} recognized as {name}'
                    if name != "Unknown":
                        cv2.putText(frame, text_show, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        if show:    
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, required=True)
    parser.add_argument('-v', type=str, required=True)
    parser.add_argument('-a', type=str, required=True)
    parser.add_argument('-m', type=str, required=True, description='Model architecture')
    parser.add_argument('-c', type=float, required=True, description='Confidence threshold')
    parser.add_argument('-i', type=int, required=True, description='Image size')
    parser.add_argument('-t', type=str, required=True, description='Tracker')
    parser.add_argument('-mt', type=int, required=True, description='Match threshold')
    parser.add_argument('-s', type=bool, default=True, description='Show video')

    args = parser.parse_args()

    predict(args.video_path, args.video_name, args.alg_name,
            args.model_arch, args.confidence_threshold, args.imgsz, 
            args.tracker, args.mt, show=True
            )