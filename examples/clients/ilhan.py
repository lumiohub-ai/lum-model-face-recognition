from face_recognition import HBFace

# RTSP streams for IN and OUT cameras
in_camera = 'rtsp://admin:qwerty12.@192.168.1.68/Streaming/Channels/101'
out_camera ='rtsp://admin:qwerty12.@192.168.1.69/Streaming/Channels/101'

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN","OUT"],
    video_path=[in_camera, out_camera],
    roi=[(994, 110, 1914, 962), (820, 87, 1703, 945)],
    # line_points = [[(2217, 886), (941, 912)],[(609, 558), (1618, 601)]],
    multi_camera=True,
    show=False,  # Disable display, just process and save
    log_file='logs/ilhan.log',
    db_path='data/embeddings/ilhan.pkl',
    debug=True)


# Run the face recognition system
face_engine_multi.run()