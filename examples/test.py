import sys
import os
sys.path.append(os.curdir)
from src.face_recognition.hbface import HBFace

video_path = 'client/pred_videos/videoa1-1_eval.mp4'

streamer = HBFace(video_path=video_path, cam_type='IN', annot=True, 
                  roi=(302, 82, 986, 976), line_points=[(129, 241), (1799, 267)])
streamer.run()

print(streamer.recognized_names)
