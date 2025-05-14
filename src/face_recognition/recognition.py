import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class FaceRecognition:
    def __init__(self, args) -> None:
        self.args = args
        self.db_names, self.db_embs = self.load_embeddings()      

    def load_embeddings(self):
        with open(self.args.db_path, 'rb') as f:
            data = pickle.load(f)

        db_embs = data['embeddings']
        db_names = data['names']
        db_names = [name.split('_')[0] for name in db_names]

        self.args.logger.info(f"Loaded {len(db_embs)} embeddings from {self.args.db_path}")

        return db_names, db_embs
    
    def update_pkl(self):
        data = {
            'embeddings': self.db_embs,
            'names': self.db_names
        }
        with open(self.args.db_path, 'wb') as f:
            pickle.dump(data, f)
        self.args.logger.info(f"Updated {self.args.db_path} with {len(self.db_embs)} embeddings")
    
    def recognize_face(self, face_embs):
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
        return cosine_similarity(face_embs, self.db_embs)

    def get_best_match(self, similarities):
        max_sim_indices = np.argmax(similarities, axis=1)
        max_sim_values = np.max(similarities, axis=1)
        best_idx = np.argmax(max_sim_values)
        best_similarity = max_sim_values[best_idx]
        best_match_db_idx = max_sim_indices[best_idx]

        return best_match_db_idx, best_similarity

    def get_matched_frame_number(self, similarities, best_match_idx):
        frame_num_matched = np.argmax(similarities, axis=0)
        return frame_num_matched[best_match_idx]

