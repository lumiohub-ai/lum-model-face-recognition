import cv2

cap = cv2.VideoCapture("rtsp://admin:hbai2024@192.168.217.150:554'")

ret, frame = cap.read()

if ret:
    cv2.imwrite("frame_in.jpg", frame)
    print("Frame saved as frame.jpg")
else:
    print("Failed to capture frame")

cap.release()