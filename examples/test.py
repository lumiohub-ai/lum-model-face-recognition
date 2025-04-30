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
    show=True,
    match_threshold=0.3,
    db_path='data/embeddings/ilhan.pkl',
    debug=False,
    log_file='logs/ilhan.log',
    )

# Run the face recognition system
face_engine_multi.run()
