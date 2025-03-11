import os

import torch
import cv2
import numpy as np
from PIL import Image

from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognition:
    # TODO - Set up data driven thresholds
    # TODO - Calculate AUC for reid of face recogniton model

    def __init__(self, db_path: str, match_threshold: float = 0.7, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.db_path = db_path
        self.resnet = (
            InceptionResnetV1(pretrained="vggface2", classify=False)
            .eval()
            .to(self.device)
        )
        self.match_threshold = match_threshold
        self.db_names, self.db_embs = self.load_embeddings()


    def compute_embeddings(self, faces):
        resized_faces = [cv2.resize(face, (160, 160)) for face in faces]
        rgb_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in resized_faces]
        
        face_tensors = torch.tensor(np.array(rgb_faces)).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        
        with torch.no_grad():
            embeddings = self.resnet(face_tensors).cpu().numpy()
    
        return embeddings

    def load_embeddings(self):
        name_to_embeddings = {}
        face_images = []
        face_names = []
        try:
            for file in os.listdir(self.db_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    face_path = os.path.join(self.db_path, file)
                    face_image = cv2.imread(face_path)

                    if face_image is None:
                        continue

                    face_images.append(face_image)
                    
                    name = file.split(".")[0]

                    face_names.append(name)
                    
            face_embs = self.compute_embeddings(face_images)
            for name, face_emb in zip(face_names, face_embs):
                name_to_embeddings[name] = face_emb

        except Exception as e:
            pass

        db_names = list(name_to_embeddings.keys())
        db_embs = np.array(list(name_to_embeddings.values()))

        return db_names, db_embs

    def recognize_face(self, face_embs):         
        similarities = cosine_similarity(face_embs, self.db_embs)
        
        # Get the indices of maximum similarity for each face
        max_sim_indices = np.argmax(similarities, axis=1)
        max_sim_values = np.max(similarities, axis=1)
        max_sim_names = [self.db_names[idx] for idx in max_sim_indices]

        # Sorted by similarity
        sorted_indices = np.argsort(-max_sim_values)
        sorted_names = [max_sim_names[idx] for idx in sorted_indices]

        return sorted_names[0].split('_')[0] if len(sorted_names) > 0 \
            and max_sim_values[sorted_indices[0]] > self.match_threshold else "Unknown", sorted_indices[0]

    def check_new_faces(self):
        for file in os.listdir(self.db_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                name = file.split(".")[0].split("_")[0]
                if name not in self.database:
                    self.database[name] = self.compute_embeddings(
                        cv2.imread(os.path.join(self.db_path, file)))