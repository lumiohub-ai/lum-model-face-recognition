"""Face recognition module for comparing face embeddings and identifying people."""

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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

    def load_embeddings(self):
        """Load face embeddings from the database file.
        
        Returns:
            Tuple containing lists of names and their corresponding face embeddings
        """
        with open(self.args.db_path, 'rb') as f:
            data = pickle.load(f)

        db_embs = data['embeddings']
        db_names = data['names']
        db_names = [name.split('_')[0] for name in db_names]

        self.args.logger.info(f"Loaded {len(db_embs)} embeddings from {self.args.db_path}")

        return db_names, db_embs
    
    def update_pkl(self):
        """Update the pickle file with the current embeddings and names."""
        data = {
            'embeddings': self.db_embs,
            'names': self.db_names
        }
        with open(self.args.db_path, 'wb') as f:
            pickle.dump(data, f)
        self.args.logger.info(f"Updated {self.args.db_path} with {len(self.db_embs)} embeddings")
    
    def recognize_face(self, face_embs):
        """Recognize a face by comparing its embeddings to the database.
        
        Args:
            face_embs: Face embeddings to compare with the database
            
        Returns:
            Dictionary containing recognition results (name, similarity, etc.)
        """
        similarities = self.compute_similarities(face_embs)
        best_match_idx, best_similarity = self.get_best_match(similarities)
        matched_name = self.db_names[best_match_idx].split('_')[0]
        matched_frame_num = self.get_matched_frame_number(similarities, best_match_idx)

        recognized = False
        if best_similarity >= self.args.match_threshold:
            recognized = True

        recognition_info = {
            'name': matched_name,
            'similarity': best_similarity,
            'matched_frame_num': matched_frame_num,
            'recognized': recognized,
            'best_match_idx': best_match_idx,
        }

        return recognition_info

    def compute_similarities(self, face_embs):
        """Compute cosine similarities between input face embeddings and database embeddings.
        
        Args:
            face_embs: Face embeddings to compare
            
        Returns:
            Matrix of similarity scores
        """
        return cosine_similarity(face_embs, self.db_embs)

    def get_best_match(self, similarities):
        """Find the best matching face embedding from the database.
        
        Args:
            similarities: Matrix of similarity scores
            
        Returns:
            Tuple containing the index of the best match and its similarity score
        """
        max_sim_indices = np.argmax(similarities, axis=1)
        max_sim_values = np.max(similarities, axis=1)
        best_idx = np.argmax(max_sim_values)
        best_similarity = max_sim_values[best_idx]
        best_match_db_idx = max_sim_indices[best_idx]

        return best_match_db_idx, best_similarity

    def get_matched_frame_number(self, similarities, best_match_idx):
        """Get the frame number that had the highest similarity for the best match.
        
        Args:
            similarities: Matrix of similarity scores
            best_match_idx: Index of the best matching face in the database
            
        Returns:
            Frame number with the highest similarity for the best match
        """
        frame_num_matched = np.argmax(similarities, axis=0)
        return frame_num_matched[best_match_idx]

