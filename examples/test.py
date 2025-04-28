import sys
import os
sys.path.append(os.curdir)

from src.face_recognition.hbface import HBFace

# RTSP streams for IN and OUT cameras
in_camera = 'rtsp://admin:Namhbai01@192.168.13.17:554'
out_camera ='rtsp://admin:Namhbai01@192.168.13.16:554'

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN","OUT"],
    video_path=[in_camera, out_camera],
    roi=[(326, 26, 1006, 719), (0, 0, 1280, 720)], # must be format of [(x1, y1, x2, y2), (x1, y1, x2, y2)]
    # line_points = [[(2217, 886), (941, 912)],[(609, 558), (1618, 601)]],
    multi_camera=True,
    show=True,  # Disable display, just process and save
    match_threshold=0.7,
    detection_threshold=0.25,
    imgsz=1280,
    padding_ratio=0.15,
    db_path='data/hb-uzb',
    save_crops=True,
    log_file='logs/ilhan_in_test.log',
    debug=True)


# Run the face recognition system
face_engine_multi.run()
