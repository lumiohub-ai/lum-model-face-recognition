import os
import numpy as np
from collections import defaultdict, deque
import torch
import cv2
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from ultralytics import solutions
from sklearn.metrics.pairwise import cosine_similarity
import random
import logging
from pathlib import Path
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;10485760"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

face_crops_path = 'data/images'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

yolo_model = YOLO("yolov8m-face.pt")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

rtsp_link = 'rtsp://admin:hbai2024@172.30.1.86:554/Streaming/Channels/1?subtype=roi'

track_frame_count = defaultdict(int)
name_to_track_id = {}
name_to_color = {}  # Store color for each person's name
person_entry_log = {}  # Store the time and date of the first entry
recent_entries = deque(maxlen=3)  # Store the last 3 entries

roi_coordinates = (528, 41, 714, 659)

def generate_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def compute_embeddings(face):
    try:
        resized_face = cv2.resize(face, (160, 160))
        resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

        face_tensor = torch.tensor(resized_face).permute(2, 0, 1).float().to(device) / 255.0
        face_tensor = face_tensor.unsqueeze(0)

        with torch.no_grad():
            emb = resnet(face_tensor).cpu().numpy().flatten()
        emb /= np.linalg.norm(emb)

        return emb
    except Exception as e:
        logger.error(f"Error computing embeddings: {str(e)}")
        return None

def load_embeddings():
    name_to_embeddings = {}
    try:
        for file in os.listdir(face_crops_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                face_path = os.path.join(face_crops_path, file)
                face_image = cv2.imread(face_path)
                if face_image is not None:
                    face_emb = compute_embeddings(face_image)
                    if face_emb is not None:
                        name = file.split('.')[0].split('_')[0]
                        name_to_embeddings[name] = face_emb
                    else:
                        logger.warning(f"Could not compute embeddings for {file}")
                else:
                    logger.warning(f"Could not read image file: {file}")
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")

    return name_to_embeddings

def recognize_face(face_emb, name_to_embeddings):
    if face_emb is None or not name_to_embeddings:
        return "Detecting..."

    sims = [cosine_similarity([face_emb], [emb]) for emb in name_to_embeddings.values()]
    max_sim = max(sims)
    if max_sim > 0.7:
        name = list(name_to_embeddings.keys())[sims.index(max_sim)]
        if max_sim > 0.9:
            random_number = random.randint(0, 10000)
            cv2.imwrite(f"data/images/{name}_{random_number}.jpg", face)
            name_to_embeddings[name] = face_emb
        return name

    return "Detecting..."

def log_person_entry(name):
    if name != "Detecting..." and name not in person_entry_log:
        now = datetime.now().strftime("%H:%M:%S on %d.%m.%Y")
        log_entry = f"{name} entered at {now}"
        person_entry_log[name] = now
        recent_entries.append(log_entry)  # Add to the recent entries deque
        # logger.info(log_entry)
        print(log_entry)  # Display the entry log in the console

cap = cv2.VideoCapture(rtsp_link, cv2.CAP_FFMPEG)
name_to_embeddings = load_embeddings()

# Object Counter
line_points = [(20, 400), (1080, 400)]
# counter = solutions.ObjectCounter(show=True, region=line_points, model_path='yolov8m-face.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        logger.error("Error reading frame")
        break

    x, y, w, h = roi_coordinates
    frame = frame[y:y + h, x:x + w]

    detections = yolo_model.track(frame, conf=0.5, verbose=False, imgsz=800, persist=True,)
    # frame = counter.count(frame)

    if detections[0].boxes.id is None:
        cv2.imshow('frame', frame)
        cv2.imwrite('frame.jpg', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = detections[0].boxes.data.cpu().numpy()
    track_ids = detections[0].boxes.id.cpu().tolist()



    for det, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2, conf, cls, _ = map(int, det)
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        face = frame[y1:y2, x1:x2]

        if face.shape[0] < 10 or face.shape[1] < 10:
            continue

        track_frame_count[track_id] += 1

        if track_frame_count[track_id] >= 30:
            if track_id in name_to_track_id:
                del name_to_track_id[track_id]  # Remove the previous association
            track_frame_count[track_id] = 0  # Reset the frame counter

        if track_id in name_to_track_id:
            name = name_to_track_id[track_id]
        else:
            face_emb = compute_embeddings(face)
            name = recognize_face(face_emb, name_to_embeddings)
            if name != "Detecting...":
                name_to_track_id[track_id] = name

        # Log the person's entry
        log_person_entry(name)

        # Assign a color to the name
        if name not in name_to_color:
            if name == "Detecting...":
                name_to_color[name] = (0, 0, 0)  # Black for "Detecting..."
            else:
                name_to_color[name] = generate_random_color()

        color = name_to_color[name]

        # Calculate adaptive font size based on the bounding box width
        box_width = x2 - x1
        font_scale = max(0.5, box_width / 300)  # Adjust font size based on width, min 0.5
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(name, font, font_scale, 1)[0]

        # Place name at the top of the bounding box
        text_x = x1 + 5
        text_y = max(0, y1 - 10)  # Ensure the text does not go out of frame

        # Ensure text stays within the bounding box
        if text_x + text_size[0] > x2:
            text_x = x2 - text_size[0] - 5  # Adjust text position if it exceeds the box

        # Draw bounding box and name label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Bounding box
        cv2.putText(frame, name, (text_x, text_y), font, font_scale, (255, 255, 255), 2)

    # Display the last 3 entries at the top-right corner of the screen
    base_y = 30
    padding = 10
    max_text_width = 0  # Track the widest text width to draw the background

    # Calculate the widest text for the background width
    for entry in recent_entries:
        text_size = cv2.getTextSize(entry, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        max_text_width = max(max_text_width, text_size[0])

    for i, entry in enumerate(recent_entries):
        # Position the text and the background
        text_size = cv2.getTextSize(entry, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        top_right_x = frame.shape[1] - max_text_width - padding * 2  # Right side position

        # Draw a black background for the text
        cv2.rectangle(frame, (top_right_x, base_y - 25 + i * 35),
                      (frame.shape[1] - 10, base_y + i * 35 + 5), (0, 0, 0), -1)

        # Draw the text on top of the black background
        cv2.putText(frame, entry, (top_right_x + 5, base_y + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
