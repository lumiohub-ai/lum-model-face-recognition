import sys
import os
import contextlib

sys.path.append(os.curdir)
sys.path.append(os.path.join(os.getcwd(), 'yolo_tracking'))

from src.face_recognition.engine import FaceRecognition
from yolo_tracking.boxmot import DeepOCSORT
import numpy as np

from pathlib import Path

import cv2
from shapely.geometry import LineString
from datetime import datetime
import pytz

from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, args) -> None:
        self.args = args
        self.timezone = pytz.timezone(args.timezone)
        self.initialize_models()
        self.initialize_tracking()

    def initialize_models(self):
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                self.model = FaceAnalysis(name='buffalo_l')
                self.model.prepare(ctx_id=0)

                self.tracker = DeepOCSORT(
                    device='cuda:0',
                    custom_features=True,
                )

                self.face_recognition = FaceRecognition(self.args)

    def initialize_tracking(self):
        """ Initialize tracking variables. """
        self.track_emb_frame_history = {}
        self.track_boxes_frame = {}
        self.track_road_history = {}
        
        self.all_tracks = set()
        
        self.id_appear_time = {}
        self.passed_tracks = []

        self.mot_results = []


    def track(self, frame):
        """ Perform tracking on the given frame."""

        faces = self.model.get(frame)

        boxes = []
        features = []

        if len(faces) == 0:
            return [], []

        for face in faces:
            embedding = face.embedding
            emb = embedding / np.linalg.norm(embedding)
            features.append(emb)

            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = face.det_score
            boxes.append([x1, y1, x2, y2, conf, 0]) # class id 0

        boxes, features = np.array(boxes), np.array(features)

        self.tracker.update(boxes, frame, features) # custom insight face features

        active_tracks = self.tracker.active_tracks # includes all the info about track
        removed_tracks = self.tracker.removed_tracks

        return active_tracks, removed_tracks
    
    def visualize_tracks(self, frame):
        """ Visualize the tracks on the frame. """
        im0 = frame.copy()

        self.tracker.plot_results(im0, show_trajectories=True)

        return im0

    def process_active_tracks(self, tracks, frame_num):
        """ Process the detections and update the tracking information. """
        for track in tracks:
            now = datetime.now(self.timezone)

            emb, track_id = track.emb, track.id

            self.all_tracks.add(track_id)
            self.id_appear_time.setdefault(track_id, now)

            if track.history_observations and len(track.history_observations) > 2:
                    box = track.history_observations[-1]
                    center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

                    self.track_road_history.setdefault(track_id, []).append(center)
                    self.track_boxes_frame.setdefault(track_id, {})[frame_num] = [
                        box[0], box[1], box[2] - box[0], box[3] - box[1], track.conf
                    ]
            else:
                continue 
                               
            self.track_emb_frame_history.setdefault(track_id, {})[frame_num] = emb

    def recognize_removed_tracks(self, removed_tracks, last_frame=False) -> dict:
        """ Recognize faces in the tracked objects. """
        persons_logged = {}
        removed_tracks = removed_tracks if not last_frame else list(self.all_tracks - set(self.passed_tracks))

        for track_id in removed_tracks:
            if track_id in self.passed_tracks:
                continue

            

            self.passed_tracks.append(track_id)

            track_id_embeddings = self.track_emb_frame_history.get(track_id, {})
            if len(track_id_embeddings) == 0:
                continue
            # Concat all embeddings for the track
            track_id_embeddings = np.array(list(track_id_embeddings.values()))

            recognition_info = self.face_recognition.recognize_face(track_id_embeddings)

            name = recognition_info['name']
            sim = recognition_info['similarity']

            
            # if self.args.save_crops:
            #     self.save_crops(track_id, recognition_info)

            if not recognition_info['recognized']:
                print(f"{name} with {track_id} cannot pass threshold with {sim}.")
                continue

            if not self.count_line_passing(track_id):
                print(f"Track {track_id} with {name} has not passed the counting line.")
                continue  

            
            persons_logged[name] = [track_id, self.id_appear_time[track_id]]

            if self.args.eval:
                for frame_num, box in self.track_boxes_frame[track_id].items():
                    self.mot_results.append({
                        'frame': frame_num, 'id': track_id,
                        'x': box[0], 'y': box[1], 'w': box[2], 'h': box[3],
                        'conf': box[4], 'name': name
                    })

        return persons_logged
    
    def is_within_roi(self, box1, box2):
        """Check if the bounding boxes intersect."""
        if box2 is None:
            return True # because user does not want to roi check
        
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Compute the coordinates of the intersection area
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)

        # Check if there is an intersection
        return xA < xB and yA < yB  # Returns True if boxes intersect, else False

    def count_line_passing(self, track_id) -> bool:
        """Check if the track has passed the counting line."""
        # Early returns for cases where counting isn't needed or possible
        if self.args.line_points is None:
            return True  # User doesn't require line passing check
            
        road_points = self.track_road_history.get(track_id, [])
        
        if len(road_points) < 2:
            return False
            
        # Check if track trajectory intersects with counting line
        first_point, last_point = road_points[0], road_points[-1]
        track_line = LineString([first_point, last_point])
        default_line = LineString(self.args.line_points)
        
        return track_line.intersects(default_line)
    
    def save_crops(self, track_id, info) -> None:
        """
        Save the cropped image along with the ground truth image from the database concatenated horizontally.
        """
        crop_image = list(self.track_crops_frame[track_id].values())[info['matched_frame_num']]
        folder_rec = 'recognized' if info['recognized'] else 'not_recognized'
        save_dir = os.path.join(self.args.crops_path, folder_rec)
        
        os.makedirs(save_dir, exist_ok=True)
        
        name = self.face_recognition.db_names[info['best_match_idx']]
        crop_save_path = os.path.join(save_dir, f"{name}_ {str(self.args.cam_type)}_{track_id}.jpg")
        
        if name in self.face_recognition.db_names:
            db_face_image = self.face_recognition.db_images[info['best_match_idx']]

            db_face_image = cv2.resize(db_face_image, (crop_image.shape[1], crop_image.shape[0]))
            combined_img = cv2.hconcat([crop_image, db_face_image])

            cv2.imwrite(crop_save_path, combined_img)

        else:
            cv2.imwrite(crop_save_path, crop_image)

    def save_faces(self, frame_num, track_id, face_crop):
        """ Save the face crops to the specified directory. """
        parent_dir = 'data_collection'
        save_dir = os.path.join(parent_dir, str(self.args.cam_type), str(track_id))
        os.makedirs(save_dir, exist_ok=True)

        random_num = np.random.randint(0, 100000)
        save_path = os.path.join(save_dir, f"{frame_num}_{random_num}.jpg")
        
        cv2.imwrite(save_path, face_crop)
        