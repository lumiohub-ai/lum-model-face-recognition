from face_recognition import HBFace  # Main entry point - now using refactored modular architecture
import os
from dotenv import load_dotenv

load_dotenv(override=True)

in_camera = os.getenv("HB_IN")
# in_camera_management = os.getenv("HB_IN_MANAGEMENT")
out_camera = os.getenv("HB_OUT")

email = os.getenv("SA_EMAIL")
password = os.getenv("SA_PASSWORD")
client_slug = os.getenv("HB_CLIENTSLUG")

face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],                                                  # camera
    video_path=[in_camera, out_camera],                                       # camera
    camera_name=["Dev Camera 2", "Dev Camera 3"],                             # camera
    camera_id=[2, 3],                                                         # camera
    # roi=[(858, 41, 1660, 903), (527, 62, 1316, 864)],                       # camera
    # line_points=[[(724, 497), (1206, 830)], [(525, 1069), (1522, 1065)]],   # camera
    match_threshold=[0.3, 0.3],                                               # camera

    db_path='volumes/src/embeddings/main.pkl',                   # will be removed
    email=email,
    password=password,
    client_slug=client_slug,
)

# Run the face recognition system
face_engine_multi.run()
