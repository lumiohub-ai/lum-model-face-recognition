import sys
import os
sys.path.append(os.curdir)

from src.face_recognition.hbface import HBFace

# RTSP streams for IN and OUT cameras
in_camera = 'client/pred_videos/videoa1-1_eval.mp4'

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types="IN",
    video_path=in_camera,
    roi=(302, 82, 986, 976),
    line_points=[(129, 241), (1799, 267)],
    multi_camera=False,
    show=True,  # Disable display, just process and save
    match_threshold=0.3,
    detection_threshold=0.25,
    imgsz=1280,
    padding_ratio=0.15,
    db_path='/home/hbvision/mirsaid/smart-office/notebooks/face_data_ilhan.pkl',
    save_crops=True,
    log_file='logs/ilhan_in_test.log',
    debug=True)


# Run the face recognition system
face_engine_multi.run()
