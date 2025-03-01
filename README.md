For running Face-Recognition app:

```
python src/face_recognition/main.py
```

Please change configurations from src/face_recognition/config.yaml

### For saving the test video use the command:

```
ffmpeg -rtsp_transport tcp -i "rtsp://admin:bHfthUmGVXxtuXTu@192.168.218.201:554/Streaming/Channels/1" -vcodec copy -acodec copy -f matroska output_3.mkv -vf "scale=1280:720" -preset ultrafast
```