import os
import cv2
import numpy as np
import pickle
import argparse
from insightface.app import FaceAnalysis

def create_face_database(training_path, output_path, device, alpha=0.9):
    # 🚀 Initialize model
    ctx_id = 0 if device == 'gpu' else -1
    model = FaceAnalysis(name='buffalo_l')
    model.prepare(ctx_id=ctx_id, det_thresh=0.1, det_size=(160, 160))

    known_embeddings = []
    classNames = []

    for file in os.listdir(training_path):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(training_path, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Cannot read {file}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = model.get(img_rgb)

        if faces:
            face = faces[0]
            emb = face.embedding

            emb = alpha * emb + (1 - alpha) * emb
            emb /= np.linalg.norm(emb)

            name = os.path.splitext(file)[0]
            known_embeddings.append(emb)
            classNames.append(name)
            print(f"✅ Detected {name} in {file}")

        else:
            print(f"⚠️ No face detected in {file}")

    data = {
        'embeddings': np.array(known_embeddings),
        'names': classNames
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Saved face embeddings to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate face embedding database using InsightFace.")
    parser.add_argument('--input_dir', required=True, help='Directory with training face images')
    parser.add_argument('--output_file', default='face_data.pkl', help='Output pickle file name')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Device to run the model')

    args = parser.parse_args()
    create_face_database(args.input_dir, args.output_file, args.device)
