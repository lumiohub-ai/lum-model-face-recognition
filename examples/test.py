import sys
import os

# Set timezone to Asia/Tashkent
os.environ['TZ'] = 'Asia/Tashkent'
import time
time.tzset()

from datetime import datetime
sys.path.append(os.curdir)

from src.face_recognition.hbface import HBFace

# RTSP streams for IN and OUT cameras
in_camera = 'rtsp://admin:Namhbai01@192.168.13.17:554'
out_camera = 'rtsp://admin:Namhbai01@192.168.13.16:554'

print("Starting face recognition...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    roi=[(383, 53, 1015, 709), (639, 1, 1276, 717)],
    line_points=[[(0, 273), (631, 264)], [(1, 2), (636, 712)]],
    multi_camera=True,
    show=True,  # Disable display, just process and save
    match_threshold=0.6,
    detection_threshold=0.7,
    imgsz=1280,
    padding_ratio=0.2,
    db_path='data/hb-uzb',
)

# Run the face recognition system
face_engine_multi.run()

print("Face recognition completed.")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
