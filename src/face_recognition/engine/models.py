from typing import Any, List, Tuple
from cfg import Config
import sys
import os
sys.path.append(os.curdir)
from ultralytics import YOLO
from engine import FaceRecognition
from utils import Visualization
import cv2


class FaceEngine:
    def __init__(self, eval: bool = False) -> None:
        self.args = Config()

        self.device = self.args.device
        self.eval = eval

        self.detector = YOLO(self.args.model_path).to(self.device).eval()
        self.tracker = self.args.tracker

        self.conf = self.args.detection_threshold
        self.imgsz = self.args.imgsz
        self.tracker = self.args.tracker

        self.face_recognition = FaceRecognition(
            db_path=self.args.db_path,
            match_threshold=self.args.match_threshold,
            device=self.device
        )

        # Initialize tracking variables
        self.track_crops_frame = {}
        self.track_boxes_frame = {}

        self.all_tracks = set()
        self.passed_tracks = []
        
        self.name_to_track_id = {}
        self.name_to_consistent_id = {}
        
        self.id_mapping = {}
        self.mot_results = []

        self.current_dets = None

        self.visualize = Visualization()
    
    def track(self, frame) -> None:
        detections = self.detector.track(
                frame,
                verbose=False,
                conf=self.conf,
                imgsz=self.imgsz,
                tracker=self.tracker,
                persist=True,
            )
        
        if detections[0].boxes.id is None:
            return None
        
        self.current_dets = detections[0].boxes.data.cpu().tolist()
        
        return detections[0]
    
    def process_detections(
            self,
            frame: Any,
            frame_num: int,
        ): 
            im0 = frame.copy()
            
            if self.current_dets is None:
                return
            
            for det in self.current_dets:
                x1, y1, x2, y2, track_id, conf, _ = map(int, det)
                w, h = x2 - x1, y2 - y1

                face = frame[y1:y2, x1:x2]
                
                if track_id not in self.track_crops_frame:
                    # Initialize track crops and boxes
                    self.track_crops_frame[track_id], self.track_boxes_frame[track_id] = {}, {}

                self.track_crops_frame[track_id][frame_num] = face
                self.track_boxes_frame[track_id][frame_num] = [x1, y1, w, h, conf]

                color = self.visualize.define_color(track_id)

                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                cv2.putText(im0, str(track_id), (x1, y1), self.visualize.font, self.visualize.font_scale, color, 2)

            return im0
            
    def recognize_tracks(
        self,
        detections: Any,
        last_frame: bool = False
    ) -> List[int]:
        persons_logged = []
        
        removed_tracks = detections.removed_tracks.tolist()
        removed_tracks = [track_id for track_id in removed_tracks if track_id not in self.passed_tracks]

        if last_frame:
            removed_tracks = self.all_tracks - set(self.passed_tracks)

        for track_id in removed_tracks:
            self.passed_tracks.append(track_id)
            
            # Skip if track has no data
            if track_id not in self.track_crops_frame or not self.track_crops_frame[track_id]:
                continue
                
            face_embeddings = self.face_recognition.compute_embeddings(self.track_crops_frame[track_id].values())
            name = self.face_recognition.recognize_face(face_embeddings)
            
            del self.track_crops_frame[track_id] # to save memory leakages
            
            if name != "Unknown":
                if name not in self.name_to_consistent_id:
                    self.name_to_consistent_id[name] = track_id

                    persons_logged.append(name)

                    consistent_id = track_id
                else:
                    consistent_id = self.name_to_consistent_id[name]
                    
                # Map the original track ID to the consistent ID
                self.id_mapping[track_id] = consistent_id
                
                # Use consistent ID for display and result storage
                display_id = consistent_id

                if self.eval:
                    # Add to MOT results with the consistent ID
                    for frame_num in self.track_boxes_frame[track_id]:
                        box = self.track_boxes_frame[track_id][frame_num]
                        self.mot_results.append({
                            'frame': frame_num,
                            'id': display_id,  # Use consistent ID in results
                            'x': box[0],
                            'y': box[1],
                            'w': box[2],
                            'h': box[3],
                            'conf': box[4],
                            'name': name,
                        })

                self.name_to_track_id[track_id] = name

        return persons_logged
    
    

    
        
        
        