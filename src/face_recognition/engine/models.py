import sys
import os
sys.path.append(os.curdir)

from src.face_recognition.engine import FaceRecognition
import numpy as np
import dlib
from ultralytics import YOLO
import cv2
from shapely.geometry import LineString
from datetime import datetime
import pytz

class FaceEngine:
    def __init__(self, args) -> None:
        self.args = args
        self.timezone = pytz.timezone(args.timezone)
        self.initialize_detectors()
        self.initialize_tracking()

    def initialize_detectors(self):
        """ Initialize the face detection and recognition models. """
        self.detector = YOLO(self.args.model_path)
        self.face_recognition = FaceRecognition(self.args)
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    def initialize_tracking(self):
        """ Initialize tracking variables. """
        self.track_crops_frame = {}
        self.track_boxes_frame = {}
        self.track_road_history = {}
        self.all_tracks = set()
        self.id_appear_time = {}
        self.passed_tracks = []
        self.name_to_consistent_id = {}
        self.id_mapping = {}
        self.mot_results = []
        self.current_dets = None
        self.visualize = self.args.visualize
        self.frames = {}

    def track(self, frame):
        """ Perform tracking on the given frame. """
        detections = self.detector.track(
            frame,
            verbose=False,
            conf=self.args.detection_threshold,
            imgsz=self.args.imgsz,
            tracker=self.args.tracker,
            persist=True,
        )
        if detections[0].boxes.id is None:
            return None, None
        
        current_dets = detections[0].boxes.data.cpu().tolist()
        removed_tracks = detections[0].removed_tracks.tolist()

        return current_dets, removed_tracks

    def align_face(self, image, desired_size=160):
        """ Align the face in the image using dlib landmarks. """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = self.dlib_detector(gray, 1)
        if not rects:
            return None
        
        shape = self.dlib_predictor(gray, rects[0])
        
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        left_eye, right_eye = landmarks[36:42].mean(axis=0).astype(int), landmarks[42:48].mean(axis=0).astype(int)
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])) * -1
        
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        x, y = max(0, eye_center[0] - desired_size // 2), max(0, eye_center[1] - desired_size // 2)
        aligned = aligned[y:y + desired_size, x:x + desired_size]
        
        if aligned.shape[0] != desired_size or aligned.shape[1] != desired_size:
            aligned = cv2.resize(aligned, (desired_size, desired_size))

        return aligned

    def process_detections(self, current_dets, frame, frame_num):
        """ Process the detections and update the tracking information. """
        im0 = frame.copy()

        if not current_dets:
            return im0

        for det in current_dets:
            self._handle_detection(det, frame, frame_num, im0)

        return im0

    def _handle_detection(self, det, frame, frame_num, im0):
        """ Process a single detection and update the tracking information. """
        now = datetime.now(self.timezone)
        
        x1, y1, x2, y2, track_id, conf, _ = map(int, det)
        width, height = x2 - x1, y2 - y1

        self.all_tracks.add(track_id)
        self.id_appear_time.setdefault(track_id, now)

        face = frame[y1:y2, x1:x2]
        padding = int(max(width, height) * self.args.padding_ratio)
        x1_padded, y1_padded = max(0, x1 - padding), max(0, y1 - padding)
        x2_padded, y2_padded = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
        padded_face = frame[y1_padded:y2_padded, x1_padded:x2_padded]

        if padded_face.size == 0 or width < self.args.min_face_size or height < self.args.min_face_size:
            return
        
        if self.args.save_crops:
            self.save_faces(frame_num, track_id, face)

        if self.args.align:
            aligned_face = self.align_face(padded_face)
        else:
            aligned_face = padded_face
        
        if aligned_face is None:
            return

        center = (x1_padded + width // 2, y1_padded + height // 2)
        self.track_road_history.setdefault(track_id, []).append(center)
        self.track_crops_frame.setdefault(track_id, {})[frame_num] = aligned_face
        self.track_boxes_frame.setdefault(track_id, {})[frame_num] = [x1_padded, y1_padded, width, height, conf]

        color = self.visualize.define_color(track_id)
        cv2.rectangle(im0, (x1_padded, y1_padded), (x2_padded, y2_padded), color, 2)
        cv2.putText(im0, f"Id: {track_id}", (x1_padded, y1_padded), 
                    self.visualize.font, self.visualize.font_scale, color, 2)

    def recognize_tracks(self, removed_tracks, last_frame=False) -> dict:
        """ Recognize faces in the tracked objects. """
        persons_logged = {}
        removed_tracks = removed_tracks if not last_frame else list(self.all_tracks - set(self.passed_tracks))

        for track_id in removed_tracks:
            self._handle_track(track_id, persons_logged)

        return persons_logged

    def _handle_track(self, track_id, persons_logged):
        """ Handle a single track and perform recognition. """
        if track_id in self.passed_tracks or not self.count_line_passing(track_id) or track_id not in self.track_crops_frame:
            if self.args.debug:
                print(f"Track {track_id} is not valid for recognition.")
            return

        self.passed_tracks.append(track_id)
        
        face_embeddings = self.face_recognition.compute_embeddings(self.track_crops_frame[track_id].values())
        name, _, best_match_idx, frame_num, recognized = self.face_recognition.recognize_face(face_embeddings, track_id)

        if self.args.save_crops:
            self.save_crops(track_id, best_match_idx, frame_num, recognized)

        if not recognized:
            return

        consistent_id = self.name_to_consistent_id.setdefault(name, track_id)
        persons_logged[name] = [track_id, self.id_appear_time[track_id]]
        self.id_mapping[track_id] = consistent_id

        if self.args.eval:
                for frame_num, box in self.track_boxes_frame[track_id].items():
                    self.mot_results.append({
                        'frame': frame_num, 'id': consistent_id,
                        'x': box[0], 'y': box[1], 'w': box[2], 'h': box[3],
                        'conf': box[4], 'name': name
                    })
    
    
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
    
    def save_crops(self, track_id, best_match_idx, frame_num, recognized) -> None:
        """
        Save the cropped image along with the ground truth image from the database concatenated horizontally.
        """
        crop_image = list(self.track_crops_frame[track_id].values())[frame_num]
        folder_rec = 'recognized' if recognized else 'not_recognized'
        save_dir = os.path.join(self.args.crops_path, folder_rec)
        
        os.makedirs(save_dir, exist_ok=True)
        
        name = self.face_recognition.db_names[best_match_idx]
        crop_save_path = os.path.join(save_dir, f"{name}_ {str(self.args.cam_type)}_{track_id}.jpg")
        
        if name in self.face_recognition.db_names:
            db_face_image = self.face_recognition.db_images[best_match_idx]

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
        