import os

import torch
import cv2
import numpy as np
import joblib
import pickle

from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

class FaceRecognition:
    def __init__(self, args) -> None:
        self.args = args
        self.resnet = (
            InceptionResnetV1(pretrained="vggface2", classify=False, layer=self.args.layer)
            .eval()
            .to(self.args.device)
        )

        self.db_names, self.db_embs = self.load_embeddings()      

    def compute_embeddings(self, faces):
        resized_faces = [cv2.resize(face, (160, 160)) for face in faces]
        rgb_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in resized_faces]
        
        face_tensors = torch.tensor(np.array(rgb_faces)).permute(0, 3, 1, 2).float().to(self.args.device) / 255.0
        
        with torch.no_grad():
            embeddings = self.resnet(face_tensors).cpu().numpy()
    
        return embeddings

    def load_embeddings(self):
        name_to_embeddings = {}
        face_images = []
        face_names = []

        for file in os.listdir(self.args.db_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                face_path = os.path.join(self.args.db_path, file)
                face_image = cv2.imread(face_path)

                if face_image is None:
                    continue
                       
                face_images.append(face_image)
                name = file.split(".")[0]             

                face_names.append(name)
                
        face_embs = self.compute_embeddings(face_images)
        for name, face_emb in zip(face_names, face_embs):
            name_to_embeddings[name] = face_emb

        db_names = list(name_to_embeddings.keys())
        db_embs = np.array(list(name_to_embeddings.values()))

        self.args.logger.info(f"Loaded {len(db_names)} embeddings from {self.args.db_path}")
        self.args.logger.info(f"Embedding shape: {db_embs.shape}")

        return db_names, db_embs
    
    def recognize_face(self, face_embs):  
        similarities = cosine_similarity(face_embs, self.db_embs)
             
        # Get the indices of maximum similarity for each face
        max_sim_indices = np.argmax(similarities, axis=1)
        max_sim_values = np.max(similarities, axis=1)
        max_sim_names = [self.db_names[idx] for idx in max_sim_indices]

        # Sorted by similarity
        sorted_indices = np.argsort(-max_sim_values)
        sorted_similarities = max_sim_values[sorted_indices]
        sorted_names = [max_sim_names[idx] for idx in sorted_indices]
        
        return sorted_names[0].split('_')[0] if len(sorted_names) > 0 \
            and max_sim_values[sorted_indices[0]] > self.args.match_threshold else "Unknown", sorted_similarities[0]

    def check_new_faces(self):
        for file in os.listdir(self.args.db_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                name = file.split(".")[0].split("_")[0]

                if name not in self.db_names:
                    self.db_names.append(self.compute_embeddings(
                       [cv2.imread(os.path.join(self.db_path, file))]))
                    
    # def recognize_face(self, face_embs):        
    #     if self.svm_classifier is None:
    #         return "Unknown", 0.0
        
    #     # Make predictions for each frame
    #     if len(face_embs.shape) > 1 and face_embs.shape[0] > 1:
    #         # Multiple frames
    #         predictions = []
    #         confidences = []
            
    #         for i in range(face_embs.shape[0]):
    #             # Get embedding for this frame
    #             frame_embedding = face_embs[i].reshape(1, -1)
                
    #             # Predict class
    #             predicted_class = self.svm_classifier.predict(frame_embedding)[0]
    #             class_probabilities = self.svm_classifier.predict_proba(frame_embedding)[0]
                
    #             # Get confidence for this prediction
    #             confidence = class_probabilities[predicted_class]
                
    #             predictions.append(predicted_class)
    #             confidences.append(confidence)
            
    #         # Find the index of the prediction with highest confidence
    #         best_idx = np.argmax(confidences)
    #         predicted_class = predictions[best_idx]
    #         confidence = confidences[best_idx]
    #     else:
    #         # Single embedding
    #         frame_embedding = face_embs.reshape(1, -1)
    #         predicted_class = self.svm_classifier.predict(frame_embedding)[0]
    #         class_probabilities = self.svm_classifier.predict_proba(frame_embedding)[0]
    #         confidence = class_probabilities[predicted_class]
        
    #     print(f"Predicted class: {self.id_to_name[predicted_class]}, Confidence: {confidence}")
        
    #     # Apply threshold
    #     if confidence > self.args.match_threshold:
    #         return self.id_to_name[predicted_class], confidence
    #     else:
    #         return "Unknown", confidence

