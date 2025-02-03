import os
import pickle
import numpy as np
from collections import defaultdict

# Flask imports
from flask import Flask, Response, render_template, request, redirect

# Vision imports
import torch
import cv2
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO

from sklearn.metrics.pairwise import cosine_similarity

# Flask app initialization
app = Flask(__name__)


face_embeddings_path = 'data/embeddings'
face_crops_path = 'data/images'

# Initialize YOLO and FaceNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolov8m-face.pt")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

video_capture = None
stop_stream = False

# Load the Web UI
@app.route('/')
def index():
    return render_template('src/web/index.html')

@app.route('/start', methods=['POST'])
def start_processing():
    global video_capture, stop_stream
    process_type = request.form['process_type']

    if process_type == "real_time":
        rtsp_link = request.form['rtsp_link']
        video_capture = cv2.VideoCapture(rtsp_link if rtsp_link else 0)
    elif process_type == "video":
        uploaded_file = request.files['video_file']
        video_path = "uploaded_video.mp4"
        uploaded_file.save(video_path)
        video_capture = cv2.VideoCapture(video_path)

    stop_stream = False
    return redirect('/video_feed')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop_streaming():
    global stop_stream
    stop_stream = True
    return redirect('/')

def process_frames():
    global video_capture, stop_stream

    # Load existing face embeddings
    embeddings = defaultdict(np.array)

    # Load stored face embeddings with names
    for file in os.listdir(face_embeddings_path):
        with open(os.path.join(face_embeddings_path, file), 'rb') as f:
            embeddings[file.split('.')[0]] = pickle.load(f)

    while True:
        if stop_stream:
            break

        ret, frame = video_capture.read()
        if not ret:
            continue

        results = yolo_model.predict(source=frame, conf=0.5)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            face = frame[y1:y2, x1:x2]

            if face.shape[0] < 10 or face.shape[1] < 10:
                continue

            face_tensor = torch.tensor(cv2.resize(face, (160, 160))).permute(2, 0, 1).float().to(device)
            face_tensor = (face_tensor / 255.0).unsqueeze(0)
            emb =  resnet(face_tensor).cpu().numpy().flatten()

            # Normalize the embedding
            emb = emb / np.linalg.norm(emb)

            sims = cosine_similarity((emb, vector) for vector in embeddings.values())
            max_sim = max(sims) if sims else float('inf')

            name = None
            if max_sim > 0.5:
                match_index = sims.index(max_sim)
                name = list(embeddings.keys())[match_index]
            else:
                name = input("Enter the name of the person: ")
                embeddings[name] = emb
                # save embedding
                with open(os.path.join(face_embeddings_path, f"{name}.pkl"), 'wb') as f:
                    pickle.dump(emb, f)
                # save crop
                cv2.imwrite(os.path.join(face_crops_path, f"{name}.jpg"), face)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
