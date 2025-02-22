import cv2

cap = cv2.VideoCapture('rtsp://admin:bHfthUmGVXxtuXTu@192.168.218.200:554/Streaming/Channels/2')

ret, frame = cap.read()

if ret:
    cv2.imwrite("frame.jpg", frame)
    print("Frame saved as frame.jpg")
else:
    print("Failed to capture frame")

cap.release()