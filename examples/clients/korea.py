from face_recognition import HBFace

# RTSP streams for IN and OUT cameras
out_camera = 'rtsp://admin:bHfthUmGVXxtuXTu@192.168.217.151:554' 
in_camera = 'rtsp://admin:hbai2024@192.168.217.150:554'

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    # roi=[(858, 41, 1660, 903), (527, 62, 1316, 864)], 
    # line_points=[[(724, 497), (1206, 830)], [(525, 1069), (1522, 1065)]],
    multi_camera=True,
    show=True,  # Disable display, just process and save
    match_threshold=0.3,
    log_file='logs/korea.log',
    db_path='data/embeddings/hb-kor-camera.pkl',
    debug=True,
)

# Run the face recognition system
face_engine_multi.run()