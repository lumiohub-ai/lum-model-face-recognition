import os
from flask import Flask, Response, render_template_string, request, redirect, jsonify
import cv2
import torch
import numpy as np
import pickle
import threading
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from datetime import datetime, timedelta
import time

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
yolo_model = YOLO("yolov8m-face.pt")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

detection_log = {}  # To track detections and time intervals
video_capture = None  # Global video capture object
stop_stream = False  # To control stopping the stream
unknown_name_queue = []  # Queue for unknown faces

@app.route('/', methods=['GET', 'POST'])
def index():
    global video_capture, stop_stream
    if request.method == 'POST':
        action = request.form['action']
        if action == "real_time":
            rtsp_link = request.form['rtsp_link']
            video_capture = cv2.VideoCapture(rtsp_link if rtsp_link else 0)
            stop_stream = False  # Reset stop flag
            return redirect('/video_feed')
        elif action == "video":
            uploaded_file = request.files['video_file']
            if uploaded_file:
                video_path = "uploaded_video.mp4"
                uploaded_file.save(video_path)
                video_capture = cv2.VideoCapture(video_path)
                stop_stream = False  # Reset stop flag
                return redirect('/video_feed')

    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Recognition Dashboard</title>
        <style>
            video { width: 640px; height: 360px; }
            .full-screen { width: 100%; height: 100vh; }
            .overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                background-color: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 20px;
                display: none;
            }
        </style>
    </head>
    <body>
        <h1>Face Recognition Dashboard</h1>
        <form method="POST">
            <label for="rtsp_link">Enter RTSP Camera Link (Leave empty to use webcam):</label>
            <input type="text" id="rtsp_link" name="rtsp_link">
            <button name="action" value="real_time">Real-Time</button>
        </form>
        <form method="POST" enctype="multipart/form-data">
            <label for="video_file">Upload Video:</label>
            <input type="file" id="video_file" name="video_file" required>
            <button name="action" value="video">Process Video</button>
        </form>
        <h2>Live Video Feed</h2>
        <video id="videoPlayer" autoplay></video>
        <button id="toggleFullScreen">Toggle Full Screen</button>
        <form method="POST" action="/stop">
            <button type="submit">Stop</button>
        </form>

        <div id="unknownOverlay" class="overlay">
            <h2>Unknown Face Detected</h2>
            <label for="unknownName">Who is this? (Leave empty to skip):</label>
            <input type="text" id="unknownName">
            <button onclick="submitUnknownName()">Submit</button>
        </div>

        <script>
            const videoPlayer = document.getElementById('videoPlayer');
            videoPlayer.src = "/video_feed";
            videoPlayer.setAttribute("playsinline", "true");

            document.getElementById('toggleFullScreen').addEventListener('click', () => {
                if (videoPlayer.classList.contains('full-screen')) {
                    videoPlayer.classList.remove('full-screen');
                } else {
                    videoPlayer.classList.add('full-screen');
                }
            });

            function showUnknownOverlay() {
                document.getElementById('unknownOverlay').style.display = 'block';
            }

            function submitUnknownName() {
                const name = document.getElementById('unknownName').value;
                fetch('/submit_unknown', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
                document.getElementById('unknownOverlay').style.display = 'none';
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/submit_unknown', methods=['POST'])
def submit_unknown():
    global unknown_name_queue
    data = request.get_json()
    unknown_name_queue.append(data['name'])
    return jsonify(success=True)

# Function to process frames and stream them
@app.route('/video_feed')
def video_feed():
    def generate_frames():
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
                        if name not in detection_log or datetime.now() - detection_log[name] > timedelta(hours=4):
                            detection_log[name] = datetime.now()
                        bbox_color = (0, 255, 0)
                    else:
                        show_unknown = True  # Unknown person detected
                        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        if show_unknown:
                            threading.Thread(target=handle_unknown_face, args=(face_encoding,)).start()

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def handle_unknown_face(face_encoding):
    global unknown_name_queue
    while not unknown_name_queue:
        time.sleep(1)  # Wait until user provides a name

    name = unknown_name_queue.pop(0)
    if name:
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
        with open("face_data.pkl", "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
