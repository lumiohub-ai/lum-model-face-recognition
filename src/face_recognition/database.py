import os
import cv2
import numpy as np
import pickle
import argparse
from insightface.app import FaceAnalysis

class Database:
    def __init__(self, device='gpu', alpha=0.9, det_thresh=0.1, det_size=(160, 160)):
        """
        Initialize the FaceDatabaseCreator with device and alpha parameter
        
        Args:
            device (str): 'gpu' or 'cpu' for face detection
            alpha (float): Weight factor for face embedding normalization
        """
        self.alpha = alpha
        self.ctx_id = 0 if device == 'gpu' else -1 # GPU or CPU
        self.model = FaceAnalysis(name='buffalo_l')
        self.model.prepare(ctx_id=self.ctx_id, det_thresh=det_thresh, det_size=det_size)
    
    def generate(self, input_dir, output_file):
        """
        Create a face database from images in input_dir and save to output_file
        
        Args:
            input_dir (str): Directory containing face images
            output_file (str): Output pickle file path to save embeddings
        
        Returns:
            dict: Dictionary containing embeddings and names
        """
        known_embeddings = []
        class_names = []

        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"❌ Input directory '{input_dir}' does not exist")
            return None
        
        # Process each image in the directory
        for file in os.listdir(input_dir):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(input_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Cannot read {file}")
                continue

            # Convert to RGB for InsightFace
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.model.get(img_rgb)

            if faces:
                face = faces[0]
                emb = face.embedding

                # Apply alpha normalization
                emb = self.alpha * emb + (1 - self.alpha) * emb
                emb /= np.linalg.norm(emb)

                name = os.path.splitext(file)[0]
                known_embeddings.append(emb)
                class_names.append(name)
                print(f"✅ Detected {name} in {file}")

            else:
                print(f"⚠️ No face detected in {file}")

        if not known_embeddings:
            print("⚠️ No faces detected in any image")
            return None

        # Create data dictionary
        data = {
            'embeddings': np.array(known_embeddings),
            'names': class_names
        }

        # Save to output file
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Saved face embeddings to {output_file}")
        
        return data