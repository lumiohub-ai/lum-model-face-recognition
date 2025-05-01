from face_recognition import HBFace

# RTSP streams for IN and OUT cameras
in_camera = 'rtsp://admin:bHfthUmGVXxtuXTu@192.168.217.152:554' 
out_camera = 'rtsp://admin:hbai2024@192.168.217.150:554'

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    multi_camera=True,
    show=True,  # Disable display, just process and save
    match_threshold=0.3,
    db_path='data/embeddings/hb-kor-camera.pkl',
)

# Run the face recognition system
face_engine_multi.run()