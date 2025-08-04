from face_recognition import HBFace
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# RTSP streams for IN and OUT cameras

in_camera = os.getenv("HB_IN")
out_camera = os.getenv("HB_OUT")

username = os.getenv("EXPERIMENT_USERNAME")
password = os.getenv("EXPERIMENT_PASSWORD")
client_slug = os.getenv("EXPERIMENT_CLIENTSLUG")

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    # roi=[(858, 41, 1660, 903), (527, 62, 1316, 864)], 
    # line_points=[[(724, 497), (1206, 830)], [(525, 1069), (1522, 1065)]],
    multi_camera=True,
    show=False,  # Disable display, just process and save
    match_threshold=0.3,
    partial_match_threshold=0.17,  # Threshold for partial matches
    db_path='data/embeddings/hb_korea.pkl',
    production=True,
    debug=True,
    record_always=False,
    save_video=False,
    username=username,
    password=password,
    client_slug=client_slug,
    save_recognized_frame=True
)

# Run the face recognition system
face_engine_multi.run()