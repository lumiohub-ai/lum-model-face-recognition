import os
import base64
from typing import Dict, Set
import asyncio
from fastapi import FastAPI, Request, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolov8m-face.pt")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

face_crops_path = 'data/images'
os.makedirs(face_crops_path, exist_ok=True)

# Create a thread pool for video processing
thread_pool = ThreadPoolExecutor(max_workers=4)

# Store active connections and their video captures
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.video_captures: Dict[str, cv2.VideoCapture] = {}
        self.processing_flags: Dict[str, bool] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.processing_flags[client_id] = True
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.video_captures:
            self.processing_flags[client_id] = False
            cap = self.video_captures[client_id]
            cap.release()
            del self.video_captures[client_id]
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.processing_flags:
            del self.processing_flags[client_id]
        logger.info(f"Client {client_id} disconnected")

manager = ConnectionManager()

def compute_embeddings(face):
    try:
        resized_face = cv2.resize(face, (160, 160))
        face_tensor = torch.tensor(resized_face).permute(2, 0, 1).float().to(device) / 255.0
        face_tensor = face_tensor.unsqueeze(0)

        with torch.no_grad():
            emb = resnet(face_tensor).cpu().numpy().flatten()

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
                        name_to_embeddings[file.split('.')[0]] = face_emb
                    else:
                        logger.warning(f"Could not compute embeddings for {file}")
                else:
                    logger.warning(f"Could not read image file: {file}")
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
    return name_to_embeddings

embeddings = load_embeddings()

def process_frame(frame, embeddings):
    try:
        if frame is None:
            return None

        # Resize frame before processing to improve performance
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            frame = cv2.resize(frame, (800, int(height * scale)))

        results = yolo_model.predict(source=frame, conf=0.5, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return frame

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])

            # Ensure coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face = frame[y1:y2, x1:x2]

            if face.shape[0] < 30 or face.shape[1] < 30:
                continue

            emb = compute_embeddings(face)
            if emb is None:
                continue

            name = "Unknown"
            max_sim = 0

            if embeddings:
                sims = [cosine_similarity([emb], [vector]) for vector in embeddings.values()]
                if sims:
                    max_sim = max(sims)
                    if max_sim > 0.7:
                        match_index = sims.index(max_sim)
                        name = list(embeddings.keys())[match_index]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({max_sim:.2f})"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1_label = max(y1, label_size[1])
            cv2.rectangle(frame, (x1, y1_label - label_size[1] - baseline),
                        (x1 + label_size[0], y1_label + baseline),
                        (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1_label),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return None

async def process_video_stream(client_id: str, rtsp_link: str):
    try:
        # Configure capture
        cap = cv2.VideoCapture(rtsp_link)
        if not cap.isOpened():
            logger.error(f"Failed to open video stream for client {client_id}")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        manager.video_captures[client_id] = cap

        frame_count = 0
        last_time = time.time()
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 5

        while manager.processing_flags.get(client_id, False):
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                logger.warning(f"Failed to read frame for client {client_id}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(f"Too many consecutive failures for client {client_id}")
                    break
                await asyncio.sleep(0.1)
                continue

            consecutive_failures = 0
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - last_time

            # Process every 3rd frame to reduce load
            if frame_count % 3 == 0:
                try:
                    processed_frame = await asyncio.get_event_loop().run_in_executor(
                        thread_pool, process_frame, frame.copy(), embeddings
                    )

                    if processed_frame is not None:
                        # Compress frame to reduce bandwidth
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        _, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')

                        try:
                            websocket = manager.active_connections.get(client_id)
                            if websocket:
                                await websocket.send_json({
                                    "type": "frame",
                                    "frame": frame_base64
                                })
                        except Exception as e:
                            logger.error(f"Error sending frame to client {client_id}: {str(e)}")
                            break
                except Exception as e:
                    logger.error(f"Error processing frame for client {client_id}: {str(e)}")
                    continue

            # Maintain around 10 FPS
            if elapsed_time < 0.1:
                await asyncio.sleep(0.1 - elapsed_time)

            if frame_count >= 30:  # Reset counter every 30 frames
                frame_count = 0
                last_time = current_time

    except Exception as e:
        logger.error(f"Error in video stream processing for client {client_id}: {str(e)}")
    finally:
        manager.disconnect(client_id)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    try:
        await manager.connect(client_id, websocket)

        while True:
            data = await websocket.receive_json()
            if data.get("type") == "start_stream":
                rtsp_link = data.get("rtsp_link")
                if rtsp_link:
                    asyncio.create_task(process_video_stream(client_id, rtsp_link))
                else:
                    await websocket.send_json({"type": "error", "message": "Invalid RTSP link"})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        manager.disconnect(client_id)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/add_face")
async def add_face(name: str = Form(...), image: UploadFile = File(None), camera: bool = Form(False)):
    try:
        if camera:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return {"message": "Failed to capture image from camera."}
        else:
            if not image:
                return {"message": "No image file provided."}
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"message": "Failed to decode image file."}

        face_results = yolo_model.predict(source=frame, conf=0.5, verbose=False)

        if not face_results or len(face_results[0].boxes.xyxy) == 0:
            return {"message": "No face detected in the image."}

        # Get the first face detected
        x1, y1, x2, y2 = map(int, face_results[0].boxes.xyxy[0][:4])

        # Ensure coordinates are within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return {"message": "Invalid face coordinates."}

        first_face = frame[y1:y2, x1:x2]

        # Compute embeddings
        emb = compute_embeddings(first_face)
        if emb is None:
            return {"message": "Failed to compute face embeddings."}

        embeddings[name] = emb

        # Save face crop
        output_path = os.path.join(face_crops_path, f"{name}.jpg")
        if not cv2.imwrite(output_path, first_face):
            return {"message": "Failed to save face image."}

        return {"message": f"Face of {name} added successfully."}
    except Exception as e:
        logger.error(f"Error adding face: {str(e)}")
        return {"message": "Error adding face. Please try again."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8075, log_level="info")