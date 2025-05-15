from face_recognition import HBFace
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# RTSP streams for IN and OUT cameras
in_camera = os.getenv("ILHAN_IN")
out_camera = os.getenv("ILHAN_OUT")

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN","OUT"],
    roi=[(788, 8, 2556, 1426), (30, 18, 1740, 1418)],
    line_points = [[(9, 267), (1760, 710)], [(278, 313), (1709, 561)]],
    video_path=[in_camera, out_camera],
    multi_camera=True,
    show=False,  # Disable display, just process and save
    log_file='data/logs/ilhan.log',
    db_path='data/embeddings/ilhan.pkl',
    production=True,
    record_always=False,
    debug=True)


# Run the face recognition system
face_engine_multi.run()