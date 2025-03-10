import sys
import os
sys.path.append(os.curdir)

from src.face_recognition.hbface import HBFace

video_path = 'videos/output_10.mkv'

streamer = HBFace(video_path=video_path, cam_type='IN', annot=True)
streamer.run()

print(streamer.recognized_names)
