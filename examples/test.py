import sys
import os
sys.path.append(os.curdir)

from src.face_recognition.hbface import HBFace

# RTSP streams for IN and OUT cameras
in_camera = ''


# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types="IN",
    video_path=in_camera,
    #roi=(4, 30, 543, 706), 
    multi_camera=False,
    show=True,  # Disable display, just process and save
    match_threshold=0.7,
    detection_threshold=0.25,
    imgsz=1280,
    padding_ratio=0.15,
    db_path='data/hb-kor-aligned-aug',
    save_crops=True,
    log_file='logs/ilhan_in_test.log',
    backend_url='',
    
)

# Run the face recognition system
face_engine_multi.run()