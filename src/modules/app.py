from flask import Flask, Response, render_template, request, redirect, jsonify
import cv2
import torch
import numpy as np
import pickle
import os
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from datetime import datetime, timedelta

# Flask app initialization
app = Flask(__name__)

# Load stored face encodings and names
if os.path.exists("face_data.pkl"):
    with open("face_data.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

# Initialize YOLO and FaceNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("/home/bahodir/Bahodir/face_recognition/yolov8m-face.pt")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

video_capture = None
stop_stream = False

@app.route('/')
def index():
    return render_template('index.html')

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
    while True:
        if stop_stream:
            break

        ret, frame = video_capture.read()
        if not ret:
            continue

        results = yolo_model.predict(source=frame, conf=0.5)
        face_tensors = []
        face_locations = []

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            face = frame[y1:y2, x1:x2]
            if face.shape[0] < 10 or face.shape[1] < 10:
                continue
            face_tensor = torch.tensor(cv2.resize(face, (160, 160))).permute(2, 0, 1).float().to(device)
            face_tensor = (face_tensor / 255.0).unsqueeze(0)
            face_tensors.append(face_tensor)
            face_locations.append((x1, y1, x2, y2))

        if face_tensors:
            with torch.no_grad():
                face_encodings = [resnet(face_tensor).cpu().numpy().flatten() for face_tensor in face_tensors]

            for face_encoding, (x1, y1, x2, y2) in zip(face_encodings, face_locations):
                distances = [np.linalg.norm(face_encoding - enc) for enc in known_face_encodings]
                min_distance = min(distances) if distances else float('inf')
                name = "Unknown"
                bbox_color = (0, 0, 255)

                if min_distance < 0.5:
                    match_index = distances.index(min_distance)
                    name = known_face_names[match_index]
                    bbox_color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
