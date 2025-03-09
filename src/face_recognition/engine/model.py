import os
import logging

import sys
import os
from PIL import Image
sys.path.append(os.curdir)

import torch
from ultralytics import solutions
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from cfg import Config

from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognitionModel:
    def __init__(self, match_threshold=0.7):
        
        cfg = Config()

        self.device = cfg.device

        self.in_counter = solutions.ObjectCounter(
            show=False,
            model=cfg.model_path,
            classes=[0],
            show_in=True, 
            show_out=True,
            verbose=False,
            tracker=cfg.tracker,
            conf=cfg.detection_threshold,
            imgsz=cfg.imgsz,
        )
        
        self.out_counter = solutions.ObjectCounter(
            show=False,
            region=cfg.out_region_points,
            model=cfg.model_path,
            classes=[0],
            show_in=True, 
            show_out=True,
            verbose=False,
            tracker=cfg.tracker,
            conf=cfg.detection_threshold,
            imgsz=cfg.imgsz,
        )  

        self.resnet = (
            InceptionResnetV1(pretrained="vggface2", classify=False)
            .eval()
            .to(self.device)
        )
        
        self.face_crops_path = cfg.face_crops_path
        
        self.match_threshold = match_threshold
        self.database = self.load_embeddings()


    def compute_embeddings(self, faces):
        resized_faces = [cv2.resize(face, (160, 160)) for face in faces]
        resized_faces = [np.array(Image.fromarray(face).convert("RGB")) for face in resized_faces]

        face_tensors = torch.tensor(np.array(resized_faces)).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        
        with torch.no_grad():
            embeddings = self.resnet(face_tensors).cpu().numpy()
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= norms  # Normalize each embedding
        
        return embeddings

    def load_embeddings(self):
        name_to_embeddings = {}
        face_images = []
        face_names = []
        try:
            for file in os.listdir(self.face_crops_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    face_path = os.path.join(self.face_crops_path, file)
                    face_image = cv2.imread(face_path)
                    if face_image is None:
                        logger.warning(f"Could not read image file: {file}")
                        continue
                    face_images.append(face_image)
                    
                    name = file.split(".")[0]

                    face_names.append(name)
                    
            face_embs = self.compute_embeddings(face_images)
            for name, face_emb in zip(face_names, face_embs):
                name_to_embeddings[name] = face_emb

        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")

        return name_to_embeddings

    def recognize_face(self, face_embs):     
        db_names = list(self.database.keys())
        db_embs = np.array(list(self.database.values()))
    
        similarities = cosine_similarity(face_embs, db_embs)
        
        # Get the indices of maximum similarity for each face
        max_sim_indices = np.argmax(similarities, axis=1)
        max_sim_values = np.max(similarities, axis=1)
        max_sim_names = [db_names[idx] for idx in max_sim_indices]

        # Sorted by similarity
        sorted_indices = np.argsort(-max_sim_values)
        sorted_names = [max_sim_names[idx] for idx in sorted_indices]

        return sorted_names[0].split('_')[0] if len(sorted_names) > 0 \
            and max_sim_values[sorted_indices[0]] > self.match_threshold else "Unknown"

    def check_new_faces(self):
        for file in os.listdir(self.face_crops_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                name = file.split(".")[0].split("_")[0]
                if name not in self.database:
                    self.database[name] = self.compute_embeddings(
                        cv2.imread(os.path.join(self.face_crops_path, file)))