import os

import torch
import cv2
import numpy as np

from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognition:
    def __init__(self, args) -> None:
        self.args = args
        self.resnet = (
            InceptionResnetV1(pretrained="vggface2", classify=False, layer=self.args.layer)
            .eval()
            .to(self.args.device)
        )

        self.db_names, self.db_embs, self.db_images = self.load_embeddings()      

    def compute_embeddings(self, faces):
        resized_faces = [cv2.resize(face, (160, 160)) for face in faces]
        rgb_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in resized_faces]
        
        face_tensors = torch.tensor(np.array(rgb_faces)).permute(0, 3, 1, 2).float().to(self.args.device) / 255.0
        
        with torch.no_grad():
            embeddings = self.resnet(face_tensors).cpu().numpy()
    
        return embeddings
    
    def get_face_images(self):
        face_images = []
        face_names = []

        for file in os.listdir(self.args.db_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                face_path = os.path.join(self.args.db_path, file)
                face_image = cv2.imread(face_path)

                if face_image is None:
                    continue
                
                name = file.split(".")[0]
                face_images.append(face_image)
                face_names.append(name)
        
        return face_images, face_names

    def load_embeddings(self):
        name_to_embeddings = {}
        cache_file = os.path.join(self.args.db_path, "embeddings.npz")
        
        # If cache exists, load and return embeddings
        if os.path.exists(cache_file):
            data = np.load(cache_file, allow_pickle=True)
            name_to_embeddings = data["embeddings"].item()
            db_names = list(name_to_embeddings.keys())
            db_embs = np.array(list(name_to_embeddings.values()))

            self.args.logger.info(f"Loaded {len(db_names)} cached embeddings from {cache_file}")
            self.args.logger.info(f"With shape: {db_embs.shape}")

            face_images, face_names = self.get_face_images()

            return db_names, db_embs, face_images
        
        face_images, face_names = self.get_face_images()

        # Compute embeddings if no cache exists
        face_embs = self.compute_embeddings(face_images)
        for name, face_emb in zip(face_names, face_embs):
            name_to_embeddings[name] = face_emb

        db_names = list(name_to_embeddings.keys())
        db_embs = np.array(list(name_to_embeddings.values()))

        # Save computed embeddings to cache
        np.savez(cache_file, embeddings=name_to_embeddings)
        self.args.logger.info(f"Saved {len(db_names)} embeddings to {cache_file}")
        self.args.logger.info(f"With shape: {db_embs.shape}")

        return db_names, db_embs, face_images
    
    def recognize_face(self, face_embs):
        similarities = self.compute_similarities(face_embs)
        best_match_idx, best_similarity = self.get_best_match(similarities)
        matched_name = self.db_names[best_match_idx].split('_')[0]
        matched_frame_num = self.get_matched_frame_number(similarities, best_match_idx)

        return matched_name, best_similarity, best_match_idx, matched_frame_num


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
 
    def check_new_faces(self):
        for file in os.listdir(self.args.db_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                name = file.split(".")[0].split("_")[0]

                if name not in self.db_names:
                    self.db_names.append(self.compute_embeddings(
                       [cv2.imread(os.path.join(self.db_path, file))]))

