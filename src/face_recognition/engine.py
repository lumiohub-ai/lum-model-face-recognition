import sys
import os
import contextlib
import numpy as np
import pytz
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Any
from shapely.geometry import LineString
import cv2

from .recognition import FaceRecognition
from boxmot import DeepOCSORT # type: ignore
from insightface.app import FaceAnalysis # type: ignore

class FaceEngine:
    def __init__(self, args) -> None:
        self.args = args
        self.timezone = pytz.timezone(args.timezone)
        self._initialize_models()
        self._initialize_tracking()
        self._setup_data_collection_folder()

    def _setup_data_collection_folder(self) -> None:
        self.data_collection_path = os.path.join(os.getcwd(), 'data/collection')
        if not os.path.exists(self.data_collection_path):
            os.makedirs(self.data_collection_path)

    def _initialize_models(self) -> None:
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                self.model = FaceAnalysis(name='buffalo_l')
                self.model.prepare(ctx_id=0)

        self.tracker = DeepOCSORT(
            device='cuda:0',
            custom_features=True,
        )

        self.face_recognition = FaceRecognition(self.args)

    def _initialize_tracking(self) -> None:
        self.track_emb_frame_history: Dict[int, Dict[int, np.ndarray]] = {}
        self.track_boxes_frame: Dict[int, Dict[int, List[float]]] = {}
        self.track_road_history: Dict[int, List[Tuple[int, int]]] = {}
        self.track_crop_history = {}

        self.all_tracks: Set[int] = set()
        self.id_appear_time: Dict[int, datetime] = {}
        self.passed_tracks: List[int] = []

        self.mot_results: List[Dict[str, Any]] = []

    def track(self, frame: np.ndarray) -> Tuple[List, List]:
        faces = self.model.get(frame)

        boxes = []
        features = []

        for face in faces:
            embedding = face.embedding
            emb = embedding / np.linalg.norm(embedding)
            features.append(emb)

            x1, y1, x2, y2 = face.bbox.astype(int)

            conf = face.det_score
            boxes.append([x1, y1, x2, y2, conf, 0])  # class id 0 for faces
        
        if len(boxes) == 0:
            # If no faces are detected, we can still update the tracker with dummy values
            self.tracker.update(np.empty((0, 6)), frame, np.empty((0, 512)))
            return [], []

        boxes, features = np.array(boxes), np.array(features)
        self.tracker.update(boxes, frame, features)

        # removed_tracks = [track.id for track in self.tracker.removed_stracks]

        return self.tracker.active_tracks, self.tracker.removed_tracks
    
    def visualize_tracks(self, frame: np.ndarray) -> np.ndarray:
        visualization_frame = frame.copy()
        self.tracker.plot_results(visualization_frame, show_trajectories=True)
        return visualization_frame

    def process_active_tracks(self, tracks: List, frame: np.ndarray, frame_num: int) -> None:
        for track in tracks:
            now = datetime.now(self.timezone)
            emb, track_id = track.emb, track.id

            self.all_tracks.add(track_id)

            if track_id not in self.id_appear_time:
                self.id_appear_time.setdefault(track_id, now)

            if track.history_observations and len(track.history_observations) > 2:
                box = track.history_observations[-1]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                face_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

                self.track_road_history.setdefault(track_id, []).append(center)
                self.track_boxes_frame.setdefault(track_id, {})[frame_num] = [
                  x1, y1, x2, y2, track.conf, 0
                ]
                self.track_crop_history.setdefault(track_id, {})[frame_num] = face_crop
                
            else:
                continue 

            self.track_emb_frame_history.setdefault(track_id, {})[frame_num] = emb
                           
    def recognize_removed_tracks(self, removed_tracks: List[int], last_frame: bool = False) -> Dict[str, List]:
        persons_logged = {}
        
        tracks_to_process = sorted(list(self.all_tracks - set(self.passed_tracks))) if last_frame else removed_tracks
        # Track must not be in passed_tracks
        tracks_to_process = [track_id for track_id in tracks_to_process if track_id not in self.passed_tracks]

        for track_id in tracks_to_process:
            self.passed_tracks.append(track_id)

            track_id_embeddings = self.track_emb_frame_history.get(track_id, {})
            if not track_id_embeddings or not self._count_line_passing(track_id):
                self._delete_cache(track_id)
                # self.args.logger.debug(f"Track {track_id} has no embeddings or has not passed the counting line.")
                continue
                
            track_id_embeddings = np.array(list(track_id_embeddings.values()))
            recognition_info = self.face_recognition.recognize_face(track_id_embeddings)
            self._save_face_crop(track_id, recognition_info)

            if not recognition_info['recognized']:
                self._delete_cache(track_id)
                continue

            name = recognition_info['name']
            sim = recognition_info['similarity']

            persons_logged[name] = [track_id, self.id_appear_time[track_id]]
            self.args.logger.debug(f"Track {track_id} with {name} has recognized with {sim:.2f}.")

            if self.args.eval:
                self._record_evaluation_results(track_id, name)

            self._delete_cache(track_id)           

        return persons_logged
    
    def _delete_cache(self, track_id) -> None:
        """Delete the cache of embeddings and boxes."""
        try:
            for d in [
                    self.track_emb_frame_history,
                    self.track_boxes_frame,
                    self.track_crop_history,
                    self.track_road_history,
                    self.id_appear_time,
                ]:
                    del d[track_id]
        except KeyError:
            pass
        
    def _save_face_crop(self, track_id, recognition_info) -> None:
        """Save a face crop of the recognized person to the data_collection folder."""
        parent_path = 'recognized' if recognition_info['recognized'] else 'unrecognized'
        if not os.path.exists(os.path.join(self.data_collection_path, parent_path)):
            os.makedirs(os.path.join(self.data_collection_path, parent_path))
        
        sim = recognition_info['similarity']

        face_crops_for_track = list(self.track_crop_history.get(track_id, {}).values())
        matched_frame_crop = face_crops_for_track[recognition_info['matched_frame_num']]

        if matched_frame_crop is not None:
            file_name = f"{recognition_info['name']}_{track_id}_{sim:.2f}.jpg"
            file_path = os.path.join(self.data_collection_path, parent_path, file_name)
            cv2.imwrite(file_path, matched_frame_crop)
        else:
            pass
    
    def _count_line_passing(self, track_id: int) -> bool:
        if self.args.line_points is None:
            return True
            
        road_points = self.track_road_history.get(track_id, [])
        
        if len(road_points) < 2:
            return False
            
        first_point, last_point = road_points[0], road_points[-1]
        track_line = LineString([first_point, last_point])
        counting_line = LineString(self.args.line_points)
        
        return track_line.intersects(counting_line)
    
    def _record_evaluation_results(self, track_id: int, name: str) -> None:
        for frame_num, box in self.track_boxes_frame[track_id].items():
            self.mot_results.append({
                'frame': frame_num, 
                'id': track_id,
                'x': box[0], 
                'y': box[1], 
                'w': box[2], 
                'h': box[3],
                'conf': box[4], 
                'name': name
            })