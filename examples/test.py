from face_recognition import HBFace

# RTSP streams for IN and OUT cameras
in_camera = 'data/tests/ilhan/videoa1-1.mp4' 
out_camera = 'data/tests/ilhan/videoa1-2.mp4'

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    multi_camera=True,
    show=True,  # Disable display, just process and save
    match_threshold=0.3,
    db_path='data/embeddings/ilhan.pkl',
)

# Run the face recognition system
face_engine_multi.run()