from hbface import HBFace

video_path = 'client/pred_videos/videoa1-1_eval.mp4'

streamer = HBFace(video_path=video_path, cam_type='IN', annot=True)
streamer.run()

print(streamer.recognized_names)
