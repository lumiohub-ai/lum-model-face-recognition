from typing import Any, List, Tuple
from cfg import Config
import sys
import os
sys.path.append(os.curdir)
from ultralytics import YOLO
from engine import FaceRecognition
from utils import Visualization
import cv2
from shapely.geometry import LineString



class FaceEngine:
    def __init__(self, video_path: str = None, eval: bool = False) -> None:
        self.args = Config()
        self.video_path = video_path
        if video_path is not None:
            self.video_path = video_path

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
        self.track_road_history = {}

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
    
    def count_line_passing(self, track_id, line_points) -> bool:
        # Get first and last point of the road history
        if track_id not in self.track_road_history:
            return False
        
        road_points = self.track_road_history[track_id]

        if len(road_points) < 2 or line_points is None:
            return False
        
        first_point = road_points[0]
        last_point = road_points[-1]

        track_line = LineString([first_point, last_point])
        default_line = LineString(line_points)

        if track_line.intersects(default_line):
            return True
        
    def process_detections(
            self,
            frame,
            frame_num,
            roi=None,
        ): 
            im0 = frame.copy()
            
            if self.current_dets is None:
                return
            
            for det in self.current_dets:
                x1, y1, x2, y2, track_id, conf, _ = map(int, det)
                w, h = x2 - x1, y2 - y1

                if not self.iou((x1, x2, w, h), roi):
                    continue

                face = frame[y1:y2, x1:x2]
                
                if track_id not in self.track_crops_frame:
                    # Initialize track crops and boxes
                    self.track_crops_frame[track_id], self.track_boxes_frame[track_id] = {}, {}

                self.track_crops_frame[track_id][frame_num] = face
                self.track_boxes_frame[track_id][frame_num] = [x1, y1, w, h, conf]

                center = (x1 + w // 2, y1 + h // 2)
                # Add the center of the bounding box to the road history
                self.track_road_history.setdefault(track_id, []).append(center)
                
                
                self.all_tracks.add(track_id)
                color = self.visualize.define_color(track_id)

                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                cv2.putText(im0, str(track_id), (x1, y1), self.visualize.font, self.visualize.font_scale, color, 2)

            # Draw the roi region
            cv2.rectangle(im0, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

            return im0
            
    def recognize_tracks(
        self,
        detections,
        line_points = None,
        last_frame = False
    ) -> List[int]:
        persons_logged = {}
        
        removed_tracks = detections.removed_tracks.tolist()
        removed_tracks = [track_id for track_id in removed_tracks if track_id not in self.passed_tracks]

        if last_frame:
            removed_tracks = self.all_tracks - set(self.passed_tracks)

        for track_id in removed_tracks:
            self.passed_tracks.append(track_id)
            
            # Skip if track do not crosses the line
            if not self.count_line_passing(track_id, line_points):
                continue

            # Skip if track has no data
            if track_id not in self.track_crops_frame or not self.track_crops_frame[track_id]:
                continue
                
            face_embeddings = self.face_recognition.compute_embeddings(self.track_crops_frame[track_id].values())
            name, matched_frame_idx = self.face_recognition.recognize_face(face_embeddings)
            
            if name != "Unknown":
                if name not in self.name_to_consistent_id:
                    self.name_to_consistent_id[name] = track_id
                    persons_logged[name] = track_id
                    consistent_id = track_id
                else:
                    consistent_id = self.name_to_consistent_id[name]

                # Map the original track ID to the consistent ID
                self.id_mapping[track_id] = consistent_id
                # Use consistent ID for display and result storage
                display_id = consistent_id

                # Concat the frame image with database image and save it
                matched_frame_img = list(self.track_crops_frame[track_id].values())[matched_frame_idx]
                matched_frame_img = cv2.resize(matched_frame_img, (160, 160))
                
                matched_database_img = cv2.imread(os.path.join(self.args.db_path, name + ".jpg"))
                matched_database_img = cv2.resize(matched_database_img, (160, 160))

                concat_img = cv2.hconcat([matched_frame_img, matched_database_img])
                cv2.imwrite(os.path.join(self.args.matched_path, 
                                         f'{name}_{display_id}_{os.path.basename(self.video_path)}.jpg'), 
                concat_img)
                
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

            del self.track_crops_frame[track_id] # to save memory leakages
        return persons_logged
    
    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Compute the coordinates of the intersection area
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)

        # Check if there is an intersection
        return xA < xB and yA < yB  # Returns True if boxes intersect, else False

    

    
        
        
        