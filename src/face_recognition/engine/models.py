from typing import Any, List, Tuple
import face_alignment
import sys
import os
sys.path.append(os.curdir)
import numpy as np
from src.face_recognition.cfg import Config
import dlib
from ultralytics import YOLO
from src.face_recognition.engine import FaceRecognition
from src.face_recognition.utils import Visualization
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

        self.detector = YOLO(self.args.model_path)

        self.tracker = self.args.tracker
    
        self.conf = self.args.detection_threshold
        self.imgsz = self.args.imgsz
        self.tracker = self.args.tracker
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, face_detector='sfd')
        self.sharpness_score = []
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
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required

        self.id_mapping = {}
        self.mot_results = []

        self.current_dets = None

        self.visualize = Visualization()
    
    def calculate_blur(self, image):
        """Calculate the variance of the Laplacian to measure blur."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance  # Higher value = sharper image
    
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
        
    def align_face_dlib(self, image, detector, predictor, desired_size=150):
        # Convert to grayscale for Dlib (optional, improves performance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the cropped image
        rects = detector(gray, 1)
        if len(rects) == 0:
            # print("No faces detected by Dlib in cropped region")
            return image

        # Get landmarks for the first detected face
        shape = predictor(gray, rects[0])
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

        # Extract eye coordinates
        left_eye = landmarks[36:42].mean(axis=0).astype(int)
        right_eye = landmarks[42:48].mean(axis=0).astype(int)

        # Calculate angle and center
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) * -1  # Negative for correct rotation

        # Center of the image
        center = (image.shape[1] // 2, image.shape[0] // 2)

        # Compute rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        # Align the image
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Crop and resize to desired size, centering on eye midpoint
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        x, y = eye_center[0] - desired_size // 2, eye_center[1] - desired_size // 2
        x, y = max(0, x), max(0, y)
        aligned = aligned[y:y + desired_size, x:x + desired_size]

        # Ensure the crop is the correct size (if the crop goes out of bounds, resize the whole image)
        if aligned.shape[0] != desired_size or aligned.shape[1] != desired_size:
            aligned = cv2.resize(aligned, (desired_size, desired_size))

        return aligned
    

    def process_detections(
            self,
            frame,
            frame_num,
            roi=None,
        ): 
            im0 = frame.copy()
            
            if self.current_dets is None:
                return
            
            for idx, det in enumerate(self.current_dets):
                x1, y1, x2, y2, track_id, conf, _ = map(int, det)
                w, h = x2 - x1, y2 - y1

                # print(f"Bounding box for det {idx}: x1={x1}, y1={y1}, x2={x2}, y2={y2}, w={w}, h={h}")

                if not self.iou((x1, x2, w, h), roi):
                    continue

                # Initial crop with padding
                padding = int(max(w, h) * 0.5)  # 50% padding
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    #print("Empty face crop at:", [x1, y1, x2, y2])
                    continue

                original_height, original_width = face.shape[:2]
                # print(f"Initial crop size (face): {original_width}x{original_height}")

                # Align the face using Dlib
                aligned_face = self.align_face_dlib(face, self.dlib_detector, self.dlib_predictor)

                aligned_height, aligned_width = aligned_face.shape[:2]
                # print(f"Aligned face size: {aligned_width}x{aligned_height}")

                # Combine for comparison
                max_height = max(original_height, aligned_height)
                face_resized = cv2.resize(face, (original_width, max_height)) if original_height < max_height else face
                aligned_face_resized = cv2.resize(aligned_face, (aligned_width, max_height)) if aligned_height < max_height else aligned_face
                combined_image = np.hstack((face_resized, aligned_face_resized))
                cv2.imwrite(f"crops/face_and_aligned_{idx}.jpg", combined_image)

                if track_id not in self.track_crops_frame:
                    self.track_crops_frame[track_id], self.track_boxes_frame[track_id] = {}, {}

                self.track_crops_frame[track_id][frame_num] = aligned_face
                self.track_boxes_frame[track_id][frame_num] = [x1, y1, w, h, conf]

                center = (x1 + w // 2, y1 + h // 2)
                self.track_road_history.setdefault(track_id, []).append(center)
                self.all_tracks.add(track_id)
                color = self.visualize.define_color(track_id)

                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                cv2.putText(im0, f"id: {track_id}", (x1, y1), self.visualize.font, self.visualize.font_scale, color, 2)

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

    

    
        
        
        