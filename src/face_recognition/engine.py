import sys
import os
import contextlib
import numpy as np
import pytz
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Any
from shapely.geometry import LineString

from .recognition import FaceRecognition
from boxmot import BoTSORT
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, args) -> None:
        self.args = args
        self.timezone = pytz.timezone(args.timezone)
        self._initialize_models()
        self._initialize_tracking()

    def _initialize_models(self) -> None:
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                self.model = FaceAnalysis(name='buffalo_l')
                self.model.prepare(ctx_id=0)

        self.tracker = BoTSORT(
            device='cuda:0',
            custom_features=True,
            track_buffer=1,
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
            w = x2 - x1
            h = y2 - y1

            if h < 100:
                continue

            conf = face.det_score
            boxes.append([x1, y1, x2, y2, conf, 0])  # class id 0 for faces
        
        if len(boxes) == 0:
            boxes = np.array([[1, 1, 10, 10, 1, 0]])  # Dummy box
            features = np.ones((1, 512))  # Dummy feature

            self.tracker.update(boxes, frame, features)

        boxes, features = np.array(boxes), np.array(features)
        self.tracker.update(boxes, frame, features)

        removed_tracks = [track.id for track in self.tracker.removed_stracks]

        return self.tracker.active_tracks, removed_tracks
    
    def visualize_tracks(self, frame: np.ndarray) -> np.ndarray:
        visualization_frame = frame.copy()
        self.tracker.plot_results(visualization_frame, show_trajectories=True)
        return visualization_frame

    def process_active_tracks(self, tracks: List, frame: np.ndarray, frame_num: int) -> None:
        for track in tracks:
            now = datetime.now(self.timezone)
            emb, track_id = track.curr_feat, track.id

            self.all_tracks.add(track_id)

            if track_id not in self.id_appear_time:
                self.id_appear_time.setdefault(track_id, now)

            if track.history_observations and len(track.history_observations) > 2:
                box = track.history_observations[-1]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                # face_crop = frame[y1:y2, x1:x2]

                center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

                self.track_road_history.setdefault(track_id, []).append(center)
                self.track_boxes_frame.setdefault(track_id, {})[frame_num] = [
                  x1, y1, x2, y2, track.conf, 0
                ]
                # self.track_crop_history.setdefault(track_id, {})[frame_num] = face_crop
                
            else:
                continue 

            self.track_emb_frame_history.setdefault(track_id, {})[frame_num] = emb
                           
    def recognize_removed_tracks(self, removed_tracks: List[int], last_frame: bool = False) -> Dict[str, List]:
        persons_logged = {}
        
        tracks_to_process = sorted(list(self.all_tracks - set(self.passed_tracks))) if last_frame else removed_tracks

        for track_id in tracks_to_process:
            if track_id in self.passed_tracks:
                continue

            self.passed_tracks.append(track_id)

            track_id_embeddings = self.track_emb_frame_history.get(track_id, {})
            if not track_id_embeddings:
                continue
                
            track_id_embeddings = np.array(list(track_id_embeddings.values()))
            recognition_info = self.face_recognition.recognize_face(track_id_embeddings)

            name = recognition_info['name']
            sim = recognition_info['similarity']

            if not recognition_info['recognized']:
                self.args.logger.debug(f"{name} with {track_id} cannot pass threshold with {sim}.")
                continue

            if not self._count_line_passing(track_id):
                self.args.logger.debug(f"Track {track_id} with {name} has not passed the counting line.")
                continue  

            # Determine movement direction if possible
            direction = self._determine_movement_direction(track_id)
            if direction and direction != self.args.cam_type:
                self.args.logger.debug(f"Track {track_id} with {name} detected in {self.args.cam_type} camera but moving {direction} direction.")
                continue
            
            persons_logged[name] = [track_id, self.id_appear_time[track_id]]
            self.args.logger.debug(f"Track {track_id} with {name} has recognized with {sim}.")

            if self.args.eval:
                self._record_evaluation_results(track_id, name)

        return persons_logged
    
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
    
    def _determine_movement_direction(self, track_id: int) -> Optional[str]:
        """Determine the movement direction based on track history."""
        road_points = self.track_road_history.get(track_id, [])
        
        if len(road_points) < 5:  # Need enough points to determine direction reliably
            return None
        
        # Get first and last points to determine direction
        first_point = road_points[0]
        last_point = road_points[-1]
        
        # Calculate movement on x-axis
        dx = last_point[0] - first_point[0]
        # Calculate movement on y-axis
        dy = last_point[1] - first_point[1]
        
        # Determine dominant direction based on largest movement
        if abs(dx) > abs(dy):
            # Horizontal movement is dominant
            direction = "OUT" if dx > 0 else "IN"
        else:
            # Vertical movement is dominant
            direction = "OUT" if dy > 0 else "IN"
            
        # Can customize this based on your specific camera setup
        return direction
    
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