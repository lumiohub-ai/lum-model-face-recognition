import sys
import os
sys.path.append(os.curdir)
from src.face_recognition.hbface import HBFace

# Get values from environment variables
in_camera = os.getenv("IN_CAMERA")
out_camera = os.getenv("OUT_CAMERA")

roi_in = tuple(map(int, os.getenv("ROI_IN", "844,2,1462,855").split(',')))
roi_out = tuple(map(int, os.getenv("ROI_OUT", "475,61,1339,871").split(',')))

line_in = [tuple(map(int, point.split(','))) for point in os.getenv("LINE_IN", "0,137;615,340").split(';')]
line_out = [tuple(map(int, point.split(','))) for point in os.getenv("LINE_OUT", "0,103;863,222").split(';')]

match_threshold = float(os.getenv("MATCH_THRESHOLD", "0.7"))
detection_threshold = float(os.getenv("DETECTION_THRESHOLD", "0.7"))
imgsz = int(os.getenv("IMGSZ", "1280"))
padding_ratio = float(os.getenv("PADDING_RATIO", "0.2"))
db_path = os.getenv("DB_PATH", "data/hb-kor")
show = os.getenv("SHOW", "False").lower() == "true"

# Initialize face recognition engine
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    roi=[roi_in, roi_out],
    line_points=[line_in, line_out],
    multi_camera=True,
    show=show,
    match_threshold=match_threshold,
    detection_threshold=detection_threshold,
    imgsz=imgsz,
    padding_ratio=padding_ratio,
    db_path=db_path,
)

face_engine_multi.run()
