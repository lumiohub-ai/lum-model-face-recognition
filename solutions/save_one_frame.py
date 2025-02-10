import cv2

cap = cv2.VideoCapture('/home/hbvision/mirsaid/smart-office/IMG_7121.mov')

ret, frame = cap.read()

if ret:
    cv2.imwrite("frame.jpg", frame)
    print("Frame saved as frame.jpg")
else:
    print("Failed to capture frame")

cap.release()