import sys
import os
sys.path.append(os.curdir)
from src.face_recognition.hbface import HBFace


in_camera = '' 
out_camera = ''

# Multiple cameras
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    multi_camera=True,
    show=True,
    match_threshold=0.7,
    db_path='data/hb-kor',
)
face_engine_multi.run()