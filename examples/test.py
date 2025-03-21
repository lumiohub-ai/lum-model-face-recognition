import sys
import os
sys.path.append(os.curdir)
from src.face_recognition.hbface import HBFace

video_path = 'in.mp4'

streamer = HBFace(video_path=video_path, cam_type='IN', show=True, db_path='data/hb-kor',
                  match_threshold=0.6, detection_threshold=0.5, imgsz=1920, eval=False)

streamer.run()

print(streamer.recognized_names)
