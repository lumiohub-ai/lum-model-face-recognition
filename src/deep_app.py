from deepface import DeepFace
import cv2

models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet"
]


cap = cv2.VideoCapture('rtsp://admin:hbai2024@172.30.1.44:554/Streaming/Channels/1')


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Perform face search using DeepFace
    dfs = DeepFace.find(
        img_path=frame,
        db_path="/home/hbvision/mirsaid/face-recognition/user/database",
        model_name=models[1],
        detector_backend='yolov8',
    )

    # If matches found, visualize results
    if len(dfs) > 0:
        for i, df in enumerate(dfs):
            person_name = df.identity[0].split("/")[-2]
            # Draw rectangle on the face

            x, y, w, h = df.source_x[0], df.source_y[0], df.source_w[0], df.source_h[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{person_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
