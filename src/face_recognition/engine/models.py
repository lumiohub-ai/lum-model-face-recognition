import sys
import os
sys.path.append(os.curdir)

from src.face_recognition.engine import FaceRecognition

import numpy as np

import dlib
from ultralytics import YOLO
import cv2

from shapely.geometry import LineString

from sklearn.metrics.pairwise import cosine_similarity

class FaceEngine:
    def __init__(self, args) -> None:
        self.args = args

        self.detector = YOLO(self.args.model_path)

        self.face_recognition = FaceRecognition(args)

        # Initialize tracking variables
        self.track_crops_frame = {}
        self.track_boxes_frame = {}
        self.track_road_history = {}

        self.all_tracks = set()
        self.passed_tracks = []
        
        self.name_to_track_id = {}
        self.name_to_consistent_id = {}

        self.sift = cv2.SIFT_create()
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor(
            "models/shape_predictor_68_face_landmarks.dat")  # Download required

        self.id_mapping = {}
        self.mot_results = []

        self.current_dets = None

        self.visualize = self.args.visualize
        self.frames = {}
    
    def track(self, frame) -> None:
        detections = self.detector.track(
                frame,
                verbose=False,
                conf=self.args.detection_threshold,
                imgsz=self.args.imgsz,
                tracker=self.args.tracker,
                persist=True,
            )
        
        if detections[0].boxes.id is None:
            return None
        
        self.current_dets = detections[0].boxes.data.cpu().tolist()
        
        return detections[0]
        
    def align_face(self, face, size=160):
        # Convert to grayscale for Dlib
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Detect faces
        rects = self.dlib_detector(gray, 1)

        if len(rects) == 0:
            return None # No face detected, return None instead of original image
        
        # Get landmarks for the first detected face
        shape = self.dlib_predictor(gray, rects[0])
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        
        # Extract eye coordinates
        left_eye = landmarks[36:42].mean(axis=0).astype(int)
        right_eye = landmarks[42:48].mean(axis=0).astype(int)
        
        # Calculate angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) * -1 # Negative for correct rotation
        
        # Center of the image
        center = (face.shape[1] // 2, face.shape[0] // 2)
        
        # Compute rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        
        # Align the image
        aligned = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))

        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        x, y = eye_center[0] - size // 2, eye_center[1] - size // 2
        x, y = max(0, x), max(0, y)
        aligned_face = aligned[y:y + size, x:x + size]
        
        # Resize to desired size
        if aligned_face.shape[0] > 0 and aligned_face.shape[1] > 0:
            # Upsample to desired size

            aligned_face = cv2.resize(aligned_face, (size, size), 
                                  interpolation=cv2.INTER_CUBIC)
            
        else:
            return None
        
        return aligned_face

    def process_detections(self, frame, frame_num):
        im0 = frame.copy()
        if not self.current_dets:
            return im0
        
        for det in self.current_dets:
            x1, y1, x2, y2, track_id, conf, _ = map(int, det)
            width, height = x2 - x1, y2 - y1
            
            if not self.is_within_roi((x1, y1, width, height), self.args.roi):
                continue
                
            # Extract face with padding
            padding = int(max(width, height) * self.args.padding_ratio)
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(frame.shape[1], x2 + padding)
            y2_padded = min(frame.shape[0], y2 + padding)
            
            face = frame[y1_padded:y2_padded, x1_padded:x2_padded]
            if not face.size:
                continue
                
            # Process optional face alignment
            aligned_face = face
            if self.args.align:
                aligned_result = self.align_face(face)
                if aligned_result is not None:
                    aligned_face = aligned_result
            
            # Track management
            self.track_crops_frame.setdefault(track_id, {})[frame_num] = aligned_face
            self.track_boxes_frame.setdefault(track_id, {})[frame_num] = [x1_padded, y1_padded, width, height, conf]
            
            # Track center point for history
            center = (x1_padded + width // 2, y1_padded + height // 2)
            self.track_road_history.setdefault(track_id, []).append(center)
            self.all_tracks.add(track_id)
            
            # Visualization
            color = self.visualize.define_color(track_id)
            cv2.rectangle(im0, (x1_padded, y1_padded), (x2_padded, y2_padded), color, 2)
            cv2.putText(im0, f"id: {track_id}", (x1_padded, y1_padded), 
                    self.visualize.font, self.visualize.font_scale, color, 2)
        
        # Draw ROI if specified
        if self.args.roi is not None:
            roi = self.args.roi
            cv2.rectangle(im0, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        
        return im0
            
    def recognize_tracks(self, detections, last_frame=False) -> dict:
        persons_logged = {}
        
        # Determine which tracks to process
        removed_tracks = detections.removed_tracks.tolist() if not last_frame else list(self.all_tracks - set(self.passed_tracks))
        removed_tracks = [track_id for track_id in removed_tracks if track_id not in self.passed_tracks]
        
        for track_id in removed_tracks:
            self.passed_tracks.append(track_id)
            
            # Skip invalid tracks early
            if (not self.count_line_passing(track_id) or 
                track_id not in self.track_crops_frame or 
                not self.track_crops_frame[track_id]):
                continue
                
            # Face recognition processing
            face_embeddings = self.face_recognition.compute_embeddings(self.track_crops_frame[track_id].values())
            name, _ = self.face_recognition.recognize_face(face_embeddings)
            
            if name == "Unknown":
                continue
                
            # Consistent ID management - use existing ID or create new one
            consistent_id = self.name_to_consistent_id.setdefault(name, track_id)
            persons_logged[name] = track_id
            self.id_mapping[track_id] = consistent_id
            
            # Handle evaluation if enabled
            if self.args.eval:
                for frame_num, box in self.track_boxes_frame[track_id].items():
                    self.mot_results.append({
                        'frame': frame_num,
                        'id': consistent_id,
                        'x': box[0],
                        'y': box[1],
                        'w': box[2],
                        'h': box[3],
                        'conf': box[4],
                        'name': name
                    })
                    
            # Store name mapping and clean up memory
            self.name_to_track_id[track_id] = name
            del self.track_crops_frame[track_id]  
            
        return persons_logged
    
    def is_within_roi(self, box1, box2):
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
        
        
        