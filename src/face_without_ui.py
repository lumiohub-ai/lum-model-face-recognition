import torch
import cv2
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import os
import pickle
import numpy as np
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity

face_embeddings_path = 'data/embeddings'
face_crops_path = 'data/images'

# Initialize YOLO and FaceNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolov8m-face.pt")
resnet = InceptionResnetV1(pretrained='vggface2').eval()

cap = cv2.VideoCapture('rtsp://admin:hbai2024@172.30.1.44:554/Streaming/Channels/1')

# Load existing face embeddings
embeddings = defaultdict(np.array)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Load stored face embeddings with names
    for file in os.listdir(face_embeddings_path):
        with open(os.path.join(face_embeddings_path, file), 'rb') as f:
            embeddings[file.split('.')[0]] = pickle.load(f)

    results = yolo_model.predict(source=frame, conf=0.5)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])\
        # declare must be resized to 160x160
        resized_w, resized_h = 160, 160

        face = frame[y1:y2, x1:x2]

        if face.shape[0] < 10 or face.shape[1] < 10:
            continue

        face_tensor = torch.tensor(cv2.resize(face, (resized_h, resized_h))).permute(2, 0, 1).float().to(device)
        face_tensor = (face_tensor / 255.0).unsqueeze(0)

        with torch.no_grad():
            emb = resnet(face_tensor).cpu().numpy().flatten()

        max_sim = 0
        name = None

        if len(embeddings) != 0:
            sims = [cosine_similarity([emb], [vector]) for vector in embeddings.values()]
            max_sim = max(sims)

        if max_sim > 0.7:
            match_index = sims.index(max_sim)
            name = list(embeddings.keys())[match_index]

        # else:
        #     name = input("Enter the name of the person: ")
        #     embeddings[name] = emb

        #     # save embedding
        #     with open(os.path.join(face_embeddings_path, f"{name}.pkl"), 'wb') as f:
        #         pickle.dump(emb, f)
        #     # save crop
        #     cv2.imwrite(os.path.join(face_crops_path, f"{name}.jpg"), face)


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()