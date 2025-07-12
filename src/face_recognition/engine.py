"""Face recognition engine implementation."""

import json
import os
import contextlib
import numpy as np
import pytz
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Any
from shapely.geometry import LineString
import cv2
import gcsfs

from .recognition import FaceRecognition
from boxmot import DeepOCSORT # type: ignore
from insightface.app import FaceAnalysis # type: ignore

class FaceEngine:
    """Main face recognition engine that handles detection, tracking and recognition of faces.
    
    This class provides the core functionality for face processing, including initializing detection 
    and recognition models, tracking faces across frames, and managing face embeddings.
    """
    def __init__(self, args) -> None:
        """Initialize the face recognition engine.
        
        Args:
            args: Configuration arguments containing parameters for the face recognition system
        """
        self.args = args
        self.timezone = pytz.timezone(args.timezone)
        self.fs = gcsfs.GCSFileSystem(token=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
        self._initialize_models()
        self._initialize_tracking()
        self._setup_data_collection_folder()

    def _setup_data_collection_folder(self) -> None:
        """Set up folder for storing collected face data and recognition results."""
        self.data_collection_path = os.path.join(os.getcwd(), 'data/collection')
        if not os.path.exists(self.data_collection_path):
            os.makedirs(self.data_collection_path)

    def _initialize_models(self) -> None:
        """Initialize face detection, recognition and tracking models."""
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                self.model = FaceAnalysis(name='buffalo_l')
                self.model.prepare(ctx_id=self.args.gpu_id)

        self.tracker = DeepOCSORT(
            device=f'cuda:{self.args.gpu_id}',
            custom_features=True,
        )
        self.face_recognition = FaceRecognition(self.args)
        self.args.logger.info(f"Loaded {len(self.face_recognition.db_embs)} embeddings from {self.face_recognition.args.db_path}")


    def _initialize_tracking(self) -> None:
        """Initialize data structures for tracking face information across frames."""
        self.track_emb_frame_history: Dict[int, Dict[int, np.ndarray]] = {}
        self.track_boxes_frame: Dict[int, Dict[int, List[float]]] = {}
        self.track_road_history: Dict[int, List[Tuple[int, int]]] = {}
        self.track_crop_history = {}
       

        self.all_tracks: Set[int] = set()
        self.id_appear_time: Dict[int, datetime] = {}
        self.passed_tracks: List[int] = []

        self.mot_results: List[Dict[str, Any]] = []

    def compute_embeddings(self, image: np.ndarray, alpha=0.9) -> List[np.ndarray]:
        """Compute face embeddings for a given image.
        
        Args:
            image: Input image array
            alpha: Normalization factor for the embedding
            
        Returns:
            Face embedding vector or None if no face is detected
        """
        faces = self.model.get(image)

        if faces:
            face = faces[0]
            emb = face.embedding

                # Apply alpha normalization
            emb = alpha * emb + (1 - alpha) * emb
            emb /= np.linalg.norm(emb)
        else:
            emb = None
        
        return emb
    
    def get_emb(self, main_url):
        """Get face embeddings from images stored in cloud storage.
        
        Args:
            main_url: URL or JSON string containing URLs to face images
            
        Returns:
            List of face embeddings or None if no face is found
        """
        # Url is the str in list of 
        try:
            url_list = json.loads(main_url)

            # Check whether url_list is a list or not
            if not isinstance(url_list, list):
                url_list = [main_url]

            embeddings = []
            for url in url_list:
                prefix = "https://storage.googleapis.com/"
                if url.startswith(prefix):
                    image_path = url[len(prefix):]

                    with self.fs.open(image_path, 'rb') as f:
                        img_bytes = f.read()

                    # Decode image from bytes to OpenCV image
                    img_array = np.frombuffer(img_bytes, np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    embedding = self.compute_embeddings(image)

                    if embedding is not None:
                        embeddings.append(embedding)

            if len(embeddings) <= 0:
                self.args.logger.warning(f"No face found in the image for {url}")
                return None
            else:
                return embeddings

        except Exception as e:
            self.args.logger.warning(f"Error processing image: {e}")
            return None
    
    def update_database(self, new_users, deleted_users) -> None:
        """Update the face recognition database with new users and delete old ones.
        
        Args:
            new_users: List of new users to add to the database
            deleted_users: List of users to remove from the database
        """
        for user in new_users:
            # Convert GCS URL to local path
            embedding = self.get_emb(user['image_path'])

            if embedding is None:
                self.args.logger.warning(f"{user['name']} has no face in the image.")
                continue
            
            for emb in embedding:
                self.face_recognition.db_names.append(user['name'])
                self.face_recognition.db_embs = np.append(self.face_recognition.db_embs, [emb], axis=0)

            self.args.logger.info(f"Added {user['name']} to the database with {len(embedding)} images.")

        for name in deleted_users:
            # Now delete the user from the database even if user has two or more images
            if name in self.face_recognition.db_names:
                index = self.face_recognition.db_names.index(name)
                self.face_recognition.db_names.pop(index)
                self.face_recognition.db_embs = np.delete(self.face_recognition.db_embs, index, axis=0)
                self.args.logger.info(f"Deleted {name} from the database.")
            else:
                self.args.logger.warning(f"{name} not found in the database.")

        self.face_recognition.update_pkl()

        return
    
    def track(self, frame: np.ndarray) -> Tuple[List, List]:
        """Track faces in a given frame using the DeepOCSORT tracker.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Tuple containing lists of active tracks and removed tracks
        """
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
            self.tracker.update(np.empty((0, 6)), frame, np.empty((0, 512)))
            return self.tracker.active_tracks, self.tracker.removed_tracks
        
        boxes, features = np.array(boxes), np.array(features)
        self.tracker.update(boxes, frame, features)

        return self.tracker.active_tracks, self.tracker.removed_tracks
    
    def visualize_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Visualize tracked faces on the input frame.
        
        Args:
            frame: Input frame to visualize tracks on
            
        Returns:
            Frame with visualization of tracked faces
        """
        visualization_frame = frame.copy()
        self.tracker.plot_results(visualization_frame, show_trajectories=True)

        # Put cam_type on the top right corner
        cam_type = self.args.cam_type
        cv2.putText(visualization_frame, cam_type, (visualization_frame.shape[1] - 200, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return visualization_frame

    def process_active_tracks(self, tracks: List, frame: np.ndarray, frame_num: int) -> None:
        """Process active tracks to extract and store face data.
        
        Args:
            tracks: List of active tracks
            frame: Current frame being processed
            frame_num: Frame number in the sequence
        """
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
        """Recognize faces in tracks that are no longer active.
        
        Args:
            removed_tracks: List of track IDs that are no longer active
            last_frame: Whether this is the last frame of the video
            
        Returns:
            Dictionary mapping recognized person names to their track information
        """
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
                
            frame_nums = list(track_id_embeddings.keys())
            emb_array = np.array(list(track_id_embeddings.values()))
            recognition_info = self.face_recognition.recognize_face(emb_array, frame_nums)


            name = recognition_info['name']
            sim = recognition_info['similarity']

            self.args.logger.debug(f"{self.args.cam_type} -> {track_id} -> {name} -> {sim:.2f}.")

            try:
                self._save_face_crop(track_id, recognition_info)
            except Exception as e:
                pass

            cropped_face = self.track_crop_history.get(track_id, {}).get(recognition_info['matched_frame_num'], None)


            persons_logged[name] = [track_id, self.id_appear_time[track_id], recognition_info['recognized'], cropped_face]

            if self.args.eval:
                self._record_evaluation_results(track_id, name)
            self._delete_cache(track_id)

        return persons_logged
    
    def _delete_cache(self, track_id) -> None:
        """Delete the cache of embeddings and boxes for a specific track ID.
        
        Args:
            track_id: ID of the track to delete from cache
        """
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
        """Save a face crop of the recognized person to the data collection folder.
        
        Args:
            track_id: ID of the track
            recognition_info: Recognition information dictionary
        """
        parent_path = 'recognized' if recognition_info['recognized'] else 'unrecognized'
        if not os.path.exists(os.path.join(self.data_collection_path, parent_path)):
            os.makedirs(os.path.join(self.data_collection_path, parent_path))
        
        sim = recognition_info['similarity']

        face_crops_for_track = list(self.track_crop_history.get(track_id, {}).values())
        matched_frame_crop = face_crops_for_track[recognition_info['matched_frame_num']]

        if matched_frame_crop is not None:
            file_name = f"{recognition_info['name']}_{track_id}_{sim:.2f}.jpg"
            file_path = os.path.join(self.data_collection_path, parent_path, file_name)
            # cv2.imwrite(file_path, matched_frame_crop)
        else:
            pass
    
    def _count_line_passing(self, track_id: int) -> bool:
        """Check if a tracked face has crossed a configured counting line.
        
        Args:
            track_id: ID of the track to check
            
        Returns:
            True if the track has crossed the counting line, False otherwise
        """
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
        """Record evaluation results for a recognized face.
        
        Args:
            track_id: ID of the track
            name: Recognized name of the person
        """
        if not self.args.eval:
            return
        
        # Get first appeared frame number 
        frame_num = list(self.track_emb_frame_history[track_id].keys())[0]

        save_path = self.args.txt_path

        if not os.path.exists(save_path):
            with open(save_path, 'w') as f:
                f.write("time,name,cam_type\n")
    
        with open(save_path, 'a') as f:
            total_seconds = frame_num / self.args.fps
            
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)

            seconds = frame_num % 60
            time_str = f"{minutes}:{seconds:02d}"
            f.write(f"{time_str},{name},{self.args.cam_type}\n")

        

        self.mot_results.append({
            'name': name,
            'cam_type': self.args.cam_type,
            'time': self.id_appear_time[track_id].strftime('%Y-%m-%d %H:%M:%S'),
        })

                