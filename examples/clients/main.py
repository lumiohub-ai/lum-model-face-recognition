from face_recognition import HBFace
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# RTSP streams for IN and OUT cameras

in_camera = os.getenv("HB_IN")
# in_camera_management = os.getenv("HB_IN_MANAGEMENT")
out_camera = os.getenv("HB_OUT")

email = os.getenv("SA_EMAIL")
password = os.getenv("SA_PASSWORD")
client_slug = os.getenv("HB_CLIENTSLUG")

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    camera_names=["Dev Camera 2", "Dev Camera 3"],
    camera_ids = [3,4]
    # roi=[(858, 41, 1660, 903), (527, 62, 1316, 864)],
    # line_points=[[(724, 497), (1206, 830)], [(525, 1069), (1522, 1065)]],
    multi_camera=True,
    show=False,
    match_threshold=[0.15, 0.3],
    partial_match_threshold=0.17,
    db_path='volumes/src/embeddings/main.pkl',
    production=True,
    debug=True,
    record_always=True,
    save_video=True,
    email=email,
    password=password,
    client_slug=client_slug,
    save_recognized_frame=True
)

# Run the face recognition system
face_engine_multi.run()