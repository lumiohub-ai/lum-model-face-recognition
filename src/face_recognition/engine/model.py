import os
import logging

import sys
import os
sys.path.append(os.curdir)

import torch
from ultralytics import solutions
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognitionModel:
    def __init__(self, device, face_crops_path, in_region_points, out_region_points, 
                match_threshold=0.7, 
                model_path='yolov8m-face.pt'):
        self.device = device
        self.model_path=model_path

        self.in_region_points = in_region_points
        self.out_region_points = out_region_points

        self.in_counter = solutions.ObjectCounter(
            show=False,
            region=in_region_points,
            model="yolov8m-face.pt",
            classes=[0],
            show_in=True, 
            show_out=True,
            line_width=2,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        
        self.out_counter = solutions.ObjectCounter(
            show=False,
            region=out_region_points,
            model="yolov8m-face.pt",
            classes=[0],
            show_in=True, 
            show_out=True,
            line_width=2,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )  

        self.resnet = (
            InceptionResnetV1(pretrained="vggface2", classify=False)
            .eval()
            .to(self.device)
        )
        
        self.face_crops_path = face_crops_path
        self.match_threshold = match_threshold
        self.database = self.load_embeddings()

    def compute_embeddings(self, face):
        try:
            resized_face = cv2.resize(face, (160, 160))
            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

            face_tensor = (
                torch.tensor(resized_face).permute(2, 0, 1).float().to(self.device)
                / 255.0
            )
            face_tensor = face_tensor.unsqueeze(0)

            with torch.no_grad():
                emb = self.resnet(face_tensor).cpu().numpy().flatten()
            emb /= np.linalg.norm(emb)

            return emb
        except Exception:
            return None

    def load_embeddings(self):
        name_to_embeddings = {}
        try:
            for file in os.listdir(self.face_crops_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    face_path = os.path.join(self.face_crops_path, file)
                    face_image = cv2.imread(face_path)
                    if face_image is None:
                        logger.warning(f"Could not read image file: {file}")
                        continue

                    face_emb = self.compute_embeddings(face_image)
                    if face_emb is None:
                        logger.warning(f"Could not compute embeddings for {file}")
                        continue

                    name = file.split(".")[0].split("_")[0]
                    name_to_embeddings[name] = face_emb
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")

        return name_to_embeddings

    def recognize_face(self, face_emb):
        if face_emb is None or not self.database:
            return "Detecting..."

        sims = [cosine_similarity([face_emb], [emb]) for emb in self.database.values()]
        max_sim = max(sims)
        if max_sim > self.match_threshold:
            name = list(self.database.keys())[sims.index(max_sim)]

            return name

        return "Detecting..."

    def check_new_faces(self):
        for file in os.listdir(self.face_crops_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                name = file.split(".")[0].split("_")[0]
                if name not in self.database:
                    self.database[name] = self.compute_embeddings(
                        cv2.imread(os.path.join(self.face_crops_path, file))
                    )
