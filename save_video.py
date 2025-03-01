import cv2
import numpy as np


out_video_path = 'rtsp://admin:bHfthUmGVXxtuXTu@192.168.218.201:554/Streaming/Channels/1'
cap = cv2.VideoCapture(out_video_path, cv2.CAP_FFMPEG)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cap.set(cv2.CAP_PROP_FPS, 20)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    

    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
