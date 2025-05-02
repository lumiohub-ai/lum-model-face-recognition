from face_recognition import HBFace

# RTSP streams for IN and OUT cameras
in_camera = 'rtsp://admin:bHfthUmGVXxtuXTu@192.168.217.151:554' 
out_camera = 'rtsp://admin:hbai2024@192.168.217.150:554'

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    roi=[(383,53,1015,709), (639,1,1276,717)], 
    multi_camera=True,
    show=False,  # Disable display, just process and save
    match_threshold=0.3,
    log_file='logs/korea.log',
    db_path='data/embeddings/hb-kor-camera.pkl',
)

# Run the face recognition system
face_engine_multi.run()