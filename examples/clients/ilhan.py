from face_recognition import HBFace
import os
from dotenv import load_dotenv

load_dotenv()

# RTSP streams for IN and OUT cameras
in_camera = os.getenv("ILHAN_IN")
out_camera = os.getenv("ILHAN_OUT")

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN","OUT"],
    roi=[(994, 110, 1914, 962), (820, 87, 1703, 945)],
    line_points = [[(4, 340), (919, 315)],[(10, 246), (873, 275)]],
    video_path=[in_camera, out_camera],
    multi_camera=True,
    show=False,  # Disable display, just process and save
    log_file='logs/ilhan.log',
    db_path='data/embeddings/ilhan.pkl',
    debug=True)


# Run the face recognition system
face_engine_multi.run()