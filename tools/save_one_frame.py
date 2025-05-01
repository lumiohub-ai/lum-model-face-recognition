import cv2

cap = cv2.VideoCapture("rtsp://admin:Namhbai01@192.168.13.16:554")

ret, frame = cap.read()

if ret:
    cv2.imwrite("frame_out.jpg", frame)
    print("Frame saved as frame.jpg")
else:
    print("Failed to capture frame")

cap.release()