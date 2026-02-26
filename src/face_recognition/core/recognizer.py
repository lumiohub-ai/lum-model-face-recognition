"""Face recognition module for comparing face embeddings and identifying people."""

import pickle
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Optional, Tuple, Any, List


class FaceRecognition:
    """Face recognition class for managing face embeddings and performing identity matching.

    This class handles loading and updating face embeddings database, and provides
    methods to recognize faces based on similarity comparison.
    """
    def __init__(self, args) -> None:
        """Initialize the face recognition system.

        Args:
            args: Configuration arguments containing parameters for face recognition
        """
        self.args = args
        self.db_names, self.db_embs = self.load_embeddings()
        self._rebuild_name_index()

    def load_embeddings(self):
        """Load face embeddings from the database file (pickle mode).

        Returns:
            Tuple containing lists of names and their corresponding face embeddings
        """
        with open(self.args.db_path, 'rb') as f:
            data = pickle.load(f)

        db_embs = data['embeddings']
        db_names = data['names']
        db_names = [name.split('_')[0] for name in db_names]

        return db_names, db_embs

    def _rebuild_name_index(self) -> None:
        """Rebuild the name-to-index mapping for O(1) lookups.

        This should be called whenever db_names is modified (add/delete operations).
        """
        self.name_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.db_names)
        }

    def get_index_by_name(self, name: str) -> Optional[int]:
        """Get the database index for a given name in O(1) time.

        Args:
            name: Person's name to look up

        Returns:
            Index in the database, or None if not found
        """
        return self.name_to_index.get(name)

    def update_pkl(self):
        """Update the pickle file with the current embeddings and names."""
        data = {
            'embeddings': self.db_embs,
            'names': self.db_names
        }
        with open(self.args.db_path, 'wb') as f:
            pickle.dump(data, f)
        self.args.logger.info(f"Updated {self.args.db_path} with {len(self.db_embs)} embeddings")

    def reload_embeddings(self):
        """Reload embeddings from pickle file into memory."""
        self.db_names, self.db_embs = self.load_embeddings()
        self._rebuild_name_index()
        self.args.logger.info(f"Reloaded {len(self.db_embs)} embeddings")

    def is_face_frontal_and_valid(self, landmarks: np.ndarray, min_frontality_threshold: float = 0.5) -> bool:
        """Check if face has all visible keypoints and meets minimum frontality requirements.

        Args:
            landmarks: Array of shape (5, 2) containing facial keypoints
            min_frontality_threshold: Minimum frontality score required

        Returns:
            True if face is valid and sufficiently frontal, False otherwise
        """
        if landmarks is None or len(landmarks) != 5:
            return False

        # Check if any keypoint is at origin or invalid (0, 0) or negative
        for point in landmarks:
            if point[0] <= 0 or point[1] <= 0:
                return False

        # Calculate frontality score
        frontality_score = self.calculate_frontality_score(landmarks)

        # Must meet minimum frontality threshold
        return frontality_score >= min_frontality_threshold

    def calculate_frontality_score(self, landmarks: np.ndarray) -> float:
        """Calculate face frontality score based on facial landmarks symmetry.

        Args:
            landmarks: Array of shape (5, 2) containing facial keypoints
                      [left_eye, right_eye, nose, left_mouth, right_mouth]

        Returns:
            Frontality score (higher means more frontal)
        """
        if landmarks is None or len(landmarks) != 5:
            return 0.0

        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]

        # Calculate eye center
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_y = (left_eye[1] + right_eye[1]) / 2

        # Calculate mouth center
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2

        # Calculate face midline (vertical line between eye center and mouth center)
        face_center_x = (eye_center_x + mouth_center_x) / 2

        # Calculate horizontal distance of nose from face midline
        nose_deviation = abs(nose[0] - face_center_x)

        # Calculate eye symmetry (distance from each eye to nose)
        left_eye_to_nose = np.linalg.norm(left_eye - nose)
        right_eye_to_nose = np.linalg.norm(right_eye - nose)
        eye_symmetry_ratio = min(left_eye_to_nose, right_eye_to_nose) / max(left_eye_to_nose, right_eye_to_nose)

        # Calculate mouth symmetry (distance from each mouth corner to nose)
        left_mouth_to_nose = np.linalg.norm(left_mouth - nose)
        right_mouth_to_nose = np.linalg.norm(right_mouth - nose)
        mouth_symmetry_ratio = min(left_mouth_to_nose, right_mouth_to_nose) / max(left_mouth_to_nose, right_mouth_to_nose)

        # Calculate inter-eye distance for normalization
        inter_eye_distance = np.linalg.norm(left_eye - right_eye)
        if inter_eye_distance == 0:
            return 0.0

        # Normalize nose deviation by inter-eye distance
        normalized_nose_deviation = nose_deviation / inter_eye_distance

        # Combine scores (higher is more frontal)
        # nose_alignment_score: 1.0 when nose is perfectly centered, decreases with deviation
        nose_alignment_score = max(0, 1.0 - normalized_nose_deviation * 2)

        # Combine all symmetry measures
        frontality_score = (nose_alignment_score * 0.4 +
                           eye_symmetry_ratio * 0.3 +
                           mouth_symmetry_ratio * 0.3)

        return frontality_score

    def recognize_face(self, face_embs, frame_nums, landmarks_dict: Optional[Dict[int, np.ndarray]] = None):
        """Recognize a face by comparing its embeddings to the database.

        Args:
            face_embs: Face embeddings to compare with the database
            frame_nums: List of frame numbers corresponding to embeddings
            landmarks_dict: Dictionary mapping frame numbers to landmark arrays

        Returns:
            Dictionary containing recognition results (name, similarity, etc.)
        """
        # Filter out non-frontal faces if landmarks are provided
        if landmarks_dict:
            valid_indices = []
            valid_frame_nums = []

            for idx, frame_num in enumerate(frame_nums):
                if frame_num in landmarks_dict:
                    landmarks = landmarks_dict[frame_num]
                    if self.is_face_frontal_and_valid(landmarks):
                        valid_indices.append(idx)
                        valid_frame_nums.append(frame_num)
                else:
                    # Keep frames without landmarks for backward compatibility
                    valid_indices.append(idx)
                    valid_frame_nums.append(frame_num)

            # If all faces are filtered out, use original data
            if len(valid_indices) == 0:
                valid_indices = list(range(len(frame_nums)))
                valid_frame_nums = frame_nums

            # Use only valid embeddings and frame numbers
            face_embs = face_embs[valid_indices]
            frame_nums = valid_frame_nums

        similarities = self.compute_similarities(face_embs)

        # Handle empty database - no embeddings to match against
        if len(self.db_embs) == 0:
            matched_frame_num = frame_nums[0] if len(frame_nums) > 0 else 0
            if landmarks_dict:
                matched_frame_num = self.get_most_frontal_frame(landmarks_dict, frame_nums)

            return {
                'name': 'unknown',
                'similarity': 0.0,
                'matched_frame_num': matched_frame_num,
                'recognized': 'unrecognized',
                'best_match_idx': -1,
            }

        best_match_idx, best_similarity = self.get_best_match(similarities)
        matched_name = self.db_names[best_match_idx].split('_')[0]

        # Get best frame for top-k analysis
        best_frame_idx = np.argmax(np.max(similarities, axis=1))

        # Get top-3 similarities from the best frame for confidence analysis
        top_k_indices = np.argsort(similarities[best_frame_idx])[-3:][::-1]
        top_k_similarities = similarities[best_frame_idx, top_k_indices].tolist()

        # Classify recognition result (RECOGNIZED/UNKNOWN/UNCERTAIN)
        status, confidence = self.classify_recognition_result(
            best_similarity, top_k_similarities
        )

        # Map status to legacy format
        if status == 'RECOGNIZED':
            recognized = 'recognized'
        else:
            # Both UNKNOWN and UNCERTAIN are treated as 'unrecognized' initially
            # UNCERTAIN will be handled differently in FaceEngine
            recognized = 'unrecognized'

        # Choose matched frame based on recognition status
        if recognized == 'unrecognized' and landmarks_dict:
            matched_frame_num = self.get_most_frontal_frame(landmarks_dict, frame_nums)
        else:
            matched_frame_num = self.get_matched_frame_number(similarities, best_match_idx, frame_nums)

        # Get top K frames with highest similarity scores to save
        top_k_frame_nums = self.get_top_k_frames(similarities, frame_nums, k=5)

        # Get frontality score for matched frame if available
        frontality_score = 0.0
        if matched_frame_num and landmarks_dict and matched_frame_num in landmarks_dict:
            frontality_score = self.calculate_frontality_score(landmarks_dict[matched_frame_num])

        recognition_info = {
            'name': matched_name,
            'similarity': best_similarity,
            'matched_frame_num': matched_frame_num,
            'recognized': recognized,
            'best_match_idx': best_match_idx,
            # New fields for enhanced filtering
            'status': status,  # RECOGNIZED/UNKNOWN/UNCERTAIN
            'confidence': confidence,
            'top3_similarities': top_k_similarities,
            'frontality_score': frontality_score,
            'top_k_frame_nums': top_k_frame_nums,  # Top K frames to save
        }

        return recognition_info

    def compute_similarities(self, face_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between input face embeddings and database embeddings.

        Args:
            face_embs: Face embeddings to compare (shape: [n_faces, embedding_dim])

        Returns:
            Matrix of similarity scores (shape: [n_faces, n_database_faces])
        """
        # Handle empty database - return empty similarity matrix
        if len(self.db_embs) == 0:
            return np.empty((len(face_embs), 0), dtype=np.float32)

        return cosine_similarity(face_embs, self.db_embs)

    def get_best_match(self, similarities: np.ndarray) -> Tuple[int, float]:
        """Find the best matching face embedding from the database.

        Args:
            similarities: Matrix of similarity scores (shape: [n_faces, n_database_faces])

        Returns:
            Tuple of (best_match_index, similarity_score)
            - best_match_index: Index of the best matching person in the database
            - similarity_score: Cosine similarity score (0.0-1.0)
        """
        max_sim_indices = np.argmax(similarities, axis=1)
        max_sim_values = np.max(similarities, axis=1)
        best_idx = np.argmax(max_sim_values)
        best_similarity = max_sim_values[best_idx]
        best_match_db_idx = max_sim_indices[best_idx]

        return best_match_db_idx, best_similarity

    def get_matched_frame_number(self, similarities: np.ndarray, best_match_idx: int, frame_nums: List[int]) -> int:
        """
        Get the frame number (from user-provided list) that had the highest similarity
        to the best matched database entry.
        """
        max_sim_indices = np.argmax(similarities, axis=1)  # DB entry index per input emb
        best_query_idx = np.argmax(np.max(similarities, axis=1))  # input embedding with best match

        return frame_nums[best_query_idx]

    def get_top_k_frames(self, similarities: np.ndarray, frame_nums: List[int], k: int = 5) -> List[int]:
        """Get the top K frame numbers with highest similarity scores.

        Args:
            similarities: Similarity matrix (num_query_embs x num_db_embs)
            frame_nums: List of frame numbers corresponding to query embeddings
            k: Number of top frames to return

        Returns:
            List of up to K frame numbers with highest similarities
        """
        # Get max similarity for each query embedding (across all DB entries)
        max_similarities = np.max(similarities, axis=1)  # Shape: (num_query_embs,)

        # Get indices of top K similarities
        k = min(k, len(max_similarities))  # Don't exceed available frames
        top_k_indices = np.argsort(max_similarities)[-k:][::-1]  # Descending order

        # Convert indices to frame numbers
        top_k_frame_nums = [frame_nums[idx] for idx in top_k_indices]

        return top_k_frame_nums

    def get_most_frontal_frame(self, landmarks_dict: Dict[int, np.ndarray], frame_nums: list) -> Optional[int]:
        """Get the frame number with the most frontal face based on landmarks.
        Only considers frames with valid frontal faces.

        Args:
            landmarks_dict: Dictionary mapping frame numbers to landmark arrays
            frame_nums: List of available frame numbers

        Returns:
            Frame number with the highest frontality score, or None if no valid frontal faces found
        """
        best_frame_num = None
        best_frontality_score = -1.0

        for frame_num in frame_nums:
            if frame_num in landmarks_dict:
                landmarks = landmarks_dict[frame_num]

                # Only consider valid frontal faces
                if self.is_face_frontal_and_valid(landmarks):
                    frontality_score = self.calculate_frontality_score(landmarks)

                    if frontality_score > best_frontality_score:
                        best_frontality_score = frontality_score
                        best_frame_num = frame_num

        return best_frame_num

    def compute_laplacian_variance(self, image_crop: np.ndarray) -> float:
        """Measure image sharpness using Laplacian variance.

        Args:
            image_crop: Face crop image

        Returns:
            Laplacian variance (higher = sharper)
        """
        if image_crop is None or image_crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def enhanced_frame_quality_check(
        self,
        face_crop: np.ndarray,
        landmarks: np.ndarray,
        bbox: np.ndarray
    ) -> Tuple[bool, float, str]:
        """Enhanced quality validation for each frame.

        Args:
            face_crop: Face crop image
            landmarks: Facial landmarks (5, 2)
            bbox: Bounding box [x1, y1, x2, y2, conf, class_id]

        Returns:
            Tuple of (is_valid, quality_score, reason)
        """
        # Get thresholds from args or use defaults
        min_frontality = getattr(self.args, 'min_frontality_score', 0.65)
        min_face_size = getattr(self.args, 'minimum_face_size', 50)
        min_laplacian = getattr(self.args, 'min_laplacian_variance', 100.0)
        min_brightness = getattr(self.args, 'min_brightness', 40)
        max_brightness = getattr(self.args, 'max_brightness', 220)
        min_landmark_spread = getattr(self.args, 'min_landmark_spread', 0.12)

        quality_checks = []

        # A. Frontality validation (stricter threshold)
        frontality_score = self.calculate_frontality_score(landmarks)
        if frontality_score < min_frontality:
            return False, 0.0, f"profile_{frontality_score:.2f}"
        quality_checks.append(('frontality', frontality_score))

        # B. Face size validation
        h, w = face_crop.shape[:2]
        if h < min_face_size or w < min_face_size:
            return False, 0.0, f"small_{w}x{h}"

        # C. Sharpness check (Laplacian variance)
        sharpness = self.compute_laplacian_variance(face_crop)
        if sharpness < min_laplacian:
            return False, 0.0, f"blurry_{sharpness:.1f}"

        # Normalize sharpness to 0-1 range
        sharpness_score = min(sharpness / 300.0, 1.0)
        quality_checks.append(('sharpness', sharpness_score))

        # D. Brightness sanity check
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if brightness < min_brightness or brightness > max_brightness:
            return False, 0.0, f"lighting_{brightness:.1f}"

        # Normalize brightness (optimal range 80-180)
        brightness_score = 1.0 - abs(brightness - 130) / 130
        brightness_score = max(0.0, min(1.0, brightness_score))
        quality_checks.append(('brightness', brightness_score))

        # E. Aspect ratio check
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:
            return False, 0.0, f"aspect_{aspect_ratio:.2f}"

        # F. Landmark spread check (detect occluded faces)
        landmarks_std = np.std(landmarks, axis=0).mean()
        bbox_size = max(w, h)
        relative_spread = landmarks_std / bbox_size if bbox_size > 0 else 0
        if relative_spread < min_landmark_spread:
            return False, 0.0, f"occluded_{relative_spread:.3f}"

        spread_score = min(relative_spread / 0.25, 1.0)
        quality_checks.append(('spread', spread_score))

        # Composite quality score
        quality_score = (
            frontality_score * 0.35 +
            sharpness_score * 0.30 +
            brightness_score * 0.20 +
            spread_score * 0.15
        )

        return True, quality_score, "pass"

    def classify_recognition_result(
        self,
        similarity: float,
        top_k_similarities: List[float]
    ) -> Tuple[str, float]:
        """Three-zone classification for recognition confidence.

        Args:
            similarity: Best match similarity (0-1)
            top_k_similarities: Top-3 match similarities

        Returns:
            Tuple of (status, confidence)
            status: 'RECOGNIZED' | 'UNKNOWN' | 'UNCERTAIN'
        """
        # Get thresholds from args or use defaults
        recognized_threshold = getattr(self.args, 'recognized_threshold', 0.40)
        true_unknown_threshold = getattr(self.args, 'true_unknown_threshold', 0.22)
        min_top2_margin = getattr(self.args, 'min_top2_margin', 0.08)

        # Zone 1: Clear recognition
        if similarity >= recognized_threshold:
            return 'RECOGNIZED', similarity

        # Zone 2: True unknown (low similarity to everyone)
        if similarity < true_unknown_threshold:
            return 'UNKNOWN', 1.0 - similarity

        # Zone 3: UNCERTAIN (between thresholds)
        # Additional checks for ambiguous cases

        # Check similarity margin between top 2 matches
        if len(top_k_similarities) >= 2:
            margin = top_k_similarities[0] - top_k_similarities[1]
            if margin < min_top2_margin:
                return 'UNCERTAIN', 0.0

        # Check variance in top-3 matches
        if len(top_k_similarities) >= 3:
            variance = float(np.var(top_k_similarities))
            if variance < 0.01:
                return 'UNCERTAIN', 0.0

        # Borderline unknown - proceed with caution
        return 'UNKNOWN', 0.5


