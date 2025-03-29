import sys
import os
sys.path.append(os.curdir)
from src.face_recognition.hbface import HBFace

in_camera = '' 
out_camera = ''

# Multiple cameras
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    roi=[(383, 53, 1015, 709), (639, 1, 1276, 717)],
    line_points=[[(0, 273), (631, 264)], [(1, 2), (636, 712)]],
    multi_camera=True,
    show=True,
    match_threshold=0.7,
    detection_threshold=0.7,
    imgsz=1280,
    padding_ratio=0.2,
    db_path='data/hb-uzb',
    save_video=True,
)
face_engine_multi.run()


# OUT Namangan
# Line: [(1, 2), (636, 712)] ROI: (639, 1, 1276, 717)

# IN Namangan
# Line: [(0, 273), (631, 264)] ROI: (383, 53, 1015, 709)
